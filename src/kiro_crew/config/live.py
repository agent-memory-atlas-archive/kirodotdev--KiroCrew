"""Live config: one poll, one reload, one dispatch.

The gateway holds many long-lived objects that copy a config value at
construction (the session manager's idle timeout, the subagent manager's
concurrency cap, a channel transport's allow-list). A write to ``config.json``
reaches those copies only if something pushes the new value at them, and only
the writer that happens to know about a copy can push -- which left every
``kirocrew config set`` and every hand edit silently inert for those fields.

This module is the single mechanism that closes the gap:

* **One poll.** :class:`ConfigWatch` runs a single background task that
  compares the loader's file fingerprint (two ``stat`` calls, off the event
  loop) on an interval. Nothing else in the gateway polls ``config.json``.
* **One reload.** When the fingerprint moves -- or an in-process writer calls
  :func:`notify_config_written` -- the watcher performs ONE
  :meth:`KiroCrewConfig.load` off the loop. That load is the same call the rest
  of the tree makes, so the ``publish_*`` snapshots the loader maintains
  (compaction threshold, timezone, alias table, MCP path dirs) ride along for
  free rather than needing a second reader.
* **One dispatch.** The old and new documents are flattened to dotted leaf paths
  and diffed; every subscriber whose prefixes intersect the changed set is
  called, sequentially, on the loop, each guarded so one failing applier cannot
  starve the rest. Subscribers receive a :class:`ConfigChange` carrying both
  configs and the exact changed paths, so an applier can be as narrow as one
  field or as wide as a section.

Design rules this module keeps (see ``docs/system-specs/modules/config.md``):

* No filesystem I/O on the event loop: the fingerprint and the load run in
  ``asyncio.to_thread``. :meth:`snapshot` is a plain attribute read.
* Loads are serialized by the watcher, so the diff is always old-vs-newer and
  an applier never sees an out-of-order pair. Other ``load()`` callers are
  unaffected; the loader's own ticket ordering still governs its snapshots.
* Values are never logged, only changed PATHS -- ``to_dict()`` carries channel
  tokens and the diff sees them.
* A subscriber bound to an object holds it weakly, so a manager that is
  discarded (tests, provider reloads) falls out of the registry on its own.
* Failure is loud, never silent: a load that raises is logged at WARNING and the
  previous snapshot is kept; the next tick retries.

Anything a gateway does in response to a config write -- from the dashboard,
the CLI or ``$EDITOR`` -- belongs behind :func:`subscribe`, not in a request
handler. A handler that mutates ``config.json`` should end with
:func:`notify_config_written` (the loader's writers do this themselves) and
let the watcher apply, so all three writers behave identically.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import threading
import weakref
from collections.abc import Awaitable, Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import cycle with the loader at runtime
    from kiro_crew.config.loader import KiroCrewConfig

logger = logging.getLogger(__name__)

#: Default seconds between fingerprint checks. Two ``stat`` calls per tick on a
#: worker thread; low enough that a CLI write lands before the operator has
#: switched windows, high enough to be invisible in CPU profiles.
DEFAULT_POLL_INTERVAL_SECS = 2.0

#: Floor for the interval so a misconfigured test or embedder cannot spin the
#: worker pool on stats.
MIN_POLL_INTERVAL_SECS = 0.05

Applier = Callable[["ConfigChange"], Awaitable[None] | None]


def flatten_config(doc: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested config document to ``{dotted.leaf.path: value}``.

    Non-empty dicts recurse; everything else (scalars, lists, empty dicts) is a
    leaf. Lists are leaves because every list-typed field in the schema is a
    whole value (an allow-list, a set of roots) whose consumers rebuild from the
    full list, so a per-index diff would only fragment one change into many.
    """
    out: dict[str, Any] = {}
    for key, value in doc.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping) and value:
            out.update(flatten_config(value, path))
        else:
            out[path] = value
    return out


def diff_config_docs(old: Mapping[str, Any] | None, new: Mapping[str, Any]) -> frozenset[str]:
    """Return the dotted leaf paths whose value differs between *old* and *new*.

    A path present on only one side counts as changed. ``old=None`` (no prior
    document) reports every leaf of *new* -- the shape a first-time subscriber
    wants when it asks to be brought up to date.
    """
    new_flat = flatten_config(new)
    if old is None:
        return frozenset(new_flat)
    old_flat = flatten_config(old)
    changed = {
        path
        for path in set(old_flat) | set(new_flat)
        if old_flat.get(path, _MISSING) != new_flat.get(path, _MISSING)
    }
    return frozenset(changed)


_MISSING = object()


def _path_matches(path: str, prefix: str) -> bool:
    """Whether dotted *path* is *prefix* itself or lies under it."""
    return path == prefix or path.startswith(prefix + ".")


@dataclass(frozen=True)
class ConfigChange:
    """What one reload observed.

    A real reload always supplies ``old``: the watcher diffs the prior document
    against the new one. ``old`` is ``None`` only when a caller synthesizes a
    change by hand (tests do) and has no prior document to offer.
    """

    old: "KiroCrewConfig | None"
    new: "KiroCrewConfig"
    changed: frozenset[str]

    def touched(self, *prefixes: str) -> bool:
        """Whether any changed path is one of *prefixes* or lies under one."""
        return any(_path_matches(p, pre) for p in self.changed for pre in prefixes)

    def under(self, prefix: str) -> frozenset[str]:
        """The changed paths that are *prefix* or lie under it."""
        return frozenset(p for p in self.changed if _path_matches(p, prefix))


@dataclass
class Subscription:
    """A registered applier. ``cancel()`` removes it; a dead weak target removes itself."""

    name: str
    prefixes: tuple[str, ...]
    _ref: Any = field(repr=False)
    _watch: "ConfigWatch | None" = field(default=None, repr=False)

    def callback(self) -> Applier | None:
        """Resolve the applier, or ``None`` if its bound object was collected."""
        ref = self._ref
        if isinstance(ref, (weakref.WeakMethod, weakref.ref)):
            return ref()
        return ref

    def cancel(self) -> None:
        if self._watch is not None:
            self._watch._remove(self)
            self._watch = None


def _hold(callback: Applier) -> Any:
    """Hold a bound method weakly so a subscriber object can be collected."""
    if inspect.ismethod(callback):
        return weakref.WeakMethod(callback)
    return callback


class ConfigWatch:
    """The process's config poller and hot-apply dispatcher.

    Construct once per gateway (``watch()`` hands out the process singleton),
    :meth:`start` it after the event loop is running, and :meth:`stop` it on
    shutdown. Subscribers may register before ``start`` -- a boot-time
    constructor is the natural place -- and are dispatched only after it.
    """

    def __init__(self, *, poll_interval_secs: float = DEFAULT_POLL_INTERVAL_SECS) -> None:
        self._interval = max(MIN_POLL_INTERVAL_SECS, float(poll_interval_secs))
        self._subs: list[Subscription] = []
        self._subs_lock = threading.Lock()
        self._cfg: KiroCrewConfig | None = None
        self._doc: dict[str, Any] | None = None
        self._fingerprint: tuple | None = None
        self._task: asyncio.Task[None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._wake: asyncio.Event | None = None
        self._force = False
        self._cycle_lock: asyncio.Lock | None = None
        self.last_error: str | None = None

    # ── registry ──────────────────────────────────────────────────────

    def subscribe(
        self,
        *prefixes: str,
        callback: Applier,
        name: str | None = None,
    ) -> Subscription:
        """Register *callback* for changes at or under any of *prefixes*.

        A prefix is a dotted config path (``"agent.max_subagents"``,
        ``"session"``, ``"agents"``). With no prefixes the applier fires on every
        reload. *callback* may be sync or async; it receives a
        :class:`ConfigChange`. A bound method is held weakly. Order of dispatch is
        order of registration.
        """
        if not callable(callback):
            raise TypeError("callback must be callable")
        sub = Subscription(
            name=name or str(getattr(callback, "__qualname__", repr(callback))),
            prefixes=tuple(prefixes),
            _ref=_hold(callback),
            _watch=self,
        )
        with self._subs_lock:
            self._subs.append(sub)
        return sub

    def _remove(self, sub: Subscription) -> None:
        with self._subs_lock:
            try:
                self._subs.remove(sub)
            except ValueError:
                pass

    def _live_subscriptions(self) -> list[Subscription]:
        """Snapshot the registry, dropping entries whose target was collected."""
        with self._subs_lock:
            alive = [s for s in self._subs if s.callback() is not None]
            if len(alive) != len(self._subs):
                self._subs = alive
            return list(alive)

    def subscriptions(self) -> Iterator[Subscription]:
        return iter(self._live_subscriptions())

    # ── state ─────────────────────────────────────────────────────────

    def snapshot(self) -> "KiroCrewConfig | None":
        """The config as of the last applied reload -- a plain attribute read.

        ``None`` before :meth:`start` (or :meth:`prime`) has run. Callers on the
        event loop should prefer this over ``KiroCrewConfig.load()`` when they
        only need the value the rest of the gateway has already adopted.
        """
        return self._cfg

    @property
    def started(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def poll_interval_secs(self) -> float:
        return self._interval

    def prime(self, cfg: "KiroCrewConfig", fingerprint: tuple | None = None) -> None:
        """Adopt *cfg* as the current snapshot without dispatching.

        Boot calls this with the config it already loaded, so the first tick
        after :meth:`start` diffs against what the gateway actually booted with
        rather than replaying every leaf. Safe from any thread; nothing here
        touches the filesystem.
        """
        self._cfg = cfg
        self._doc = cfg.to_dict()
        if fingerprint is not None:
            self._fingerprint = fingerprint

    # ── lifecycle ─────────────────────────────────────────────────────

    async def start(self, initial: "KiroCrewConfig | None" = None) -> None:
        """Arm the poll task on the running loop.

        *initial* is the config the caller booted with. It is primed as the
        baseline but deliberately WITHOUT the file's current fingerprint: the
        gateway loaded it some time before this call, and an edit made in that
        window (a CLI ``config set`` while the gateway was booting) would
        otherwise be paired with a fingerprint that already reflects it and
        never be applied. Leaving the fingerprint unset makes the first cycle
        reload the file and diff it against the boot config, so that edit is
        dispatched like any other. Without *initial* the watcher loads once
        (off-loop) and that load IS the baseline, fingerprint included.
        Idempotent: a second call while running is a no-op.
        """
        if self.started:
            return
        loop = asyncio.get_running_loop()
        self._loop = loop
        self._wake = asyncio.Event()
        self._cycle_lock = asyncio.Lock()
        if initial is not None:
            self.prime(initial)
            self._fingerprint = None
        elif self._cfg is None:
            fingerprint = await asyncio.to_thread(self._current_fingerprint)
            cfg = await asyncio.to_thread(self._load)
            self.prime(cfg, fingerprint)
        self._task = loop.create_task(self._run(), name="config-watch")
        self._task.add_done_callback(self._on_task_done)

    async def stop(self) -> None:
        task = self._task
        self._task = None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    def notify_written(self) -> None:
        """Wake the poll immediately after an in-process config write.

        Safe from any thread (writers run in ``asyncio.to_thread``). Forces a
        reload on the next cycle even if the fingerprint reads equal, because
        the fingerprint is mtime-based and a coarse filesystem clock can make a
        write invisible to it. A no-op before :meth:`start`.
        """
        self._force = True
        loop, wake = self._loop, self._wake
        if loop is None or wake is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(wake.set)
        except RuntimeError:
            # Loop shut down between the check and the call: nothing to wake.
            pass

    async def refresh_now(self) -> ConfigChange | None:
        """Run one reload-and-dispatch cycle and return what it applied.

        For request handlers that must answer only after the new value is in
        force, and for tests. Forces the load regardless of fingerprint.
        Returns ``None`` when nothing changed or the load failed (the failure is
        logged and kept in :attr:`last_error`).
        """
        self._force = True
        return await self._cycle()

    # ── internals ─────────────────────────────────────────────────────

    def _on_task_done(self, task: "asyncio.Task[None]") -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error("config watch task exited unexpectedly", exc_info=exc)

    async def _run(self) -> None:
        assert self._wake is not None
        while True:
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=self._interval)
            except asyncio.TimeoutError:
                pass
            self._wake.clear()
            try:
                await self._cycle()
            except asyncio.CancelledError:
                raise
            except Exception:
                # _cycle contains its own failures; this is the last line of
                # defence so the poll never dies on an unexpected error.
                logger.exception("config watch cycle failed")

    async def _cycle(self) -> ConfigChange | None:
        if self._cycle_lock is None:
            # Not started: a direct refresh_now() from a test or a CLI path.
            self._cycle_lock = asyncio.Lock()
        async with self._cycle_lock:
            force, self._force = self._force, False
            fingerprint = await asyncio.to_thread(self._current_fingerprint)
            if not force and fingerprint == self._fingerprint:
                return None
            try:
                cfg, doc = await asyncio.to_thread(self._load_with_doc)
            except Exception as e:  # noqa: BLE001 - keep the previous snapshot
                self.last_error = f"{type(e).__name__}: {e}"
                logger.warning("config reload failed; keeping previous snapshot: %s", e)
                return None
            changed = diff_config_docs(self._doc, doc)
            old = self._cfg
            # Adopt before dispatch so an applier that reads snapshot() sees the
            # new config, and record the PRE-load fingerprint: a write landing
            # mid-read leaves it unequal to the file, so the next tick reloads.
            self._cfg, self._doc, self._fingerprint = cfg, doc, fingerprint
            self.last_error = None
            if not changed:
                return None
            change = ConfigChange(old=old, new=cfg, changed=changed)
            logger.info(
                "config reloaded: %d field(s) changed (%s)",
                len(changed),
                ", ".join(sorted(changed)[:12]) + (" …" if len(changed) > 12 else ""),
            )
            await self._dispatch(change)
            return change

    async def _dispatch(self, change: ConfigChange) -> None:
        for sub in self._live_subscriptions():
            if sub.prefixes and not change.touched(*sub.prefixes):
                continue
            cb = sub.callback()
            if cb is None:
                continue
            try:
                result = cb(change)
                if inspect.isawaitable(result):
                    await result
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("config applier %r failed; continuing", sub.name)

    @staticmethod
    def _current_fingerprint() -> tuple:
        from kiro_crew.config import loader

        return loader._config_fingerprint()

    @staticmethod
    def _load() -> "KiroCrewConfig":
        from kiro_crew.config.loader import KiroCrewConfig

        return KiroCrewConfig.load()

    @classmethod
    def _load_with_doc(cls) -> tuple["KiroCrewConfig", dict[str, Any]]:
        cfg = cls._load()
        return cfg, cfg.to_dict()


# ── process singleton ────────────────────────────────────────────────

_WATCH: ConfigWatch | None = None
_WATCH_LOCK = threading.Lock()


def watch() -> ConfigWatch:
    """The process-global watcher, created on first use (never started here)."""
    global _WATCH
    with _WATCH_LOCK:
        if _WATCH is None:
            _WATCH = ConfigWatch()
        return _WATCH


def subscribe(*prefixes: str, callback: Applier, name: str | None = None) -> Subscription:
    """Register an applier on the process watcher. See :meth:`ConfigWatch.subscribe`."""
    return watch().subscribe(*prefixes, callback=callback, name=name)


def snapshot() -> "KiroCrewConfig | None":
    """The last applied config on the process watcher, or ``None`` if unstarted."""
    w = _WATCH
    return w.snapshot() if w is not None else None


def notify_config_written() -> None:
    """Tell the process watcher a config write just landed.

    Called by the loader's own writers, so every in-process writer wakes the
    watcher without knowing it exists. A no-op in processes with no watcher
    (the CLI, tests that never started one).
    """
    w = _WATCH
    if w is not None:
        w.notify_written()


def reset_for_tests() -> None:
    """Drop the process watcher so a test gets a fresh, unstarted one."""
    global _WATCH
    with _WATCH_LOCK:
        w = _WATCH
        _WATCH = None
    if w is not None and w._task is not None:
        w._task.cancel()


__all__ = [
    "Applier",
    "ConfigChange",
    "ConfigWatch",
    "DEFAULT_POLL_INTERVAL_SECS",
    "Subscription",
    "diff_config_docs",
    "flatten_config",
    "notify_config_written",
    "reset_for_tests",
    "snapshot",
    "subscribe",
    "watch",
]
