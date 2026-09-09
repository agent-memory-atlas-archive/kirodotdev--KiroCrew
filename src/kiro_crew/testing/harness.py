"""Test harness for spawning isolated KiroCrew gateways.

Companion to the ``--test-mode`` / ``--json-ready`` / ``--port`` / ``--approval``
CLI flags on ``kirocrew gateway``. Provides a context manager that spins up
an isolated, headless gateway from the current workspace's source tree (not
the system-installed ``kirocrew``), reads the ``KIROCREW_READY:{...}`` line
off stdout, and tears down cleanly on exit.

Transport-agnostic: ``GatewayHandle`` exposes the URL plus a few metadata
fields. The caller chooses the driver (Playwright via DSO Frontend MCP is
the recommended one; plain HTTP for backend-only smoke tests is fine but
not the recommended path).

Usage:

    from kiro_crew.testing.harness import spawn_feature_gateway

    with spawn_feature_gateway() as handle:
        # drive Playwright / urllib / etc against handle.url
        ...
    # subprocess and tmp KIROCREW_HOME are gone by here

Reusable primitives:

    ``parse_ready_line`` and ``terminate_pgid`` are public, I/O-agnostic
    building blocks. Out-of-process supervisors that drive a *detached*
    gateway (tail its log for the READY line, terminate it later by pid
    without holding the ``Popen``) can reuse them instead of re-implementing
    the wire-contract parse and the SIGTERM→SIGKILL group kill.

Cross-platform: the harness runs on Linux, macOS and Windows. Two things had
to be platform-neutral for that, and both are load-bearing rather than
cosmetic:

    * The READY wait reads the child's stdout through a daemon reader thread
      feeding a ``queue.Queue`` (:class:`_StdoutPump`). ``selectors`` cannot
      poll a PIPE on Windows -- ``DefaultSelector`` there is select()-based and
      accepts sockets only -- so a selector-driven read is not merely slower on
      Windows, it raises. The queue's bounded ``get`` preserves what the
      selector poll bought: the deadline is enforced even while the child is
      alive and silent.
    * Teardown routes through ``platform_compat.kill_process_tree`` on Windows
      (``taskkill /T /F``) instead of ``terminate_pgid``; there is no
      ``setsid`` / ``killpg`` there. See
      ``docs/system-specs/common/platform-compat.md``.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import queue
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Callable, Iterator, Optional

from kiro_crew import platform_compat
from kiro_crew.kiro_prerequisite import FAKE_ACP_TEST_MODE_ENV

_LOGGER = logging.getLogger(__name__)

# 60 s default — config init + MCP probe + dashboard bind takes meaningful
# time on slow machines. 60 s gives headroom without masking real hangs.
# Override via ``KIROCREW_HARNESS_READY_TIMEOUT``.
DEFAULT_READY_TIMEOUT = 60.0

# How long to wait between SIGTERM and SIGKILL during teardown. Gateway's
# graceful-shutdown budget is 10 s internally; 5 s here is enough for the
# common case (no pending tool calls, no hung MCP servers) and bounds the
# pytest teardown latency for tests that exercise multiple invocations.
TERMINATE_GRACE_SECONDS = 5.0

# How long the READY wait keeps reading stdout after the child has exited, so a
# READY line already written but not yet handed over by the reader thread is not
# misreported as "exited before READY". The child's write end is closed by then,
# so EOF arrives in milliseconds; this is a ceiling, not a sleep.
EXIT_DRAIN_SECONDS = 2.0

# Sentinel prefix the gateway prints to stdout once the dashboard is bound.
# Owned by ``slack/gateway.py``; if you change it there, update here too.
READY_PREFIX = "KIROCREW_READY:"


@dataclass(frozen=True)
class GatewayHandle:
    """Handle on a spawned gateway.

    Intentionally minimal — exposes the URL plus a few metadata fields and
    leaves all I/O to the caller. Adding HTTP / WebSocket / MCP helpers
    here would couple every consumer to a specific driver; keeping it
    transport-agnostic lets each test pick its own (Playwright, plain
    axios, urllib, websockets — whatever fits).

    Attributes:
        url: Authenticated dashboard URL with token query param. Safe to
            ``urllib.request.urlopen`` directly or feed to a browser via
            Playwright.
        port: OS-assigned ephemeral port the dashboard is bound to.
        token: Session token embedded in ``url``. Exposed separately for
            clients that build their own URLs (e.g. WebSocket connectors).
        home: Path to the throwaway ``KIROCREW_HOME`` directory the gateway
            is using. Useful for tests that need to inspect on-disk state
            (sessions, memory, lessons) after exercising the gateway.
        proc: Underlying ``subprocess.Popen`` handle. Most tests should
            never touch this; the context manager owns its lifecycle.
    """

    url: str
    port: int
    token: str
    home: Path
    proc: subprocess.Popen[bytes]
    #: Bound diagnostics provider (exit status, stderr tail, stdout tail).
    #: ``None`` for handles built outside :func:`spawn_feature_gateway`.
    _diagnostics: Optional[Callable[[], str]] = None

    def diagnostics(self) -> str:
        """Exit status plus the stderr and stdout tails of the child, for a
        failure AFTER the READY line.

        The READY wait already attaches stderr to a spawn failure. A test's own
        first request failing (connection reset, refused, a 5xx) is the other
        common shape, and on a platform nobody can reproduce locally the child's
        stderr is the only evidence there is. Call this in the assertion message.
        """
        if self._diagnostics is None:
            return f"pid {self.proc.pid} exit={self.proc.poll()!r} (no captured output)"
        return self._diagnostics()


class GatewaySpawnError(RuntimeError):
    """Raised when the harness can't spin up a working gateway.

    Wraps the underlying cause (timeout, early exit, bad fixture name)
    with the subprocess's stderr so failures produce useful diagnostics
    instead of a bare ``TimeoutError`` from a buried ``readline()``.
    """


def _resolve_workspace_src() -> Path:
    """Locate the in-repo ``src/`` so PYTHONPATH points at feature-branch code.

    Walks up from this file's location
    (``<pkg>/src/kiro_crew/testing/harness.py``) to find the package's
    ``src/`` directory. We deliberately avoid using the system-installed
    ``kirocrew`` for two reasons:

    1. We're testing the *current* code, not whatever the developer has
       on PATH. A stale Toolbox install would silently mask regressions.
    2. The system install may not have ``--test-mode`` yet, so we must run
       the in-repo code that does.
    """
    here = Path(__file__).resolve()
    # <pkg>/src/kiro_crew/testing/harness.py -> <pkg>/src
    src = here.parent.parent.parent
    if not (src / "kiro_crew" / "__init__.py").exists():
        raise GatewaySpawnError(
            f"Could not locate kiro_crew package at {src}. "
            f"Harness expects harness.py at <pkg>/src/kiro_crew/testing/, "
            f"with the source tree at <pkg>/src/kiro_crew/."
        )
    return src


def parse_ready_line(line: str) -> dict[str, Any]:
    """Parse and validate a single ``KIROCREW_READY:{...}`` line.

    Pure (no I/O), so both the in-process pipe reader
    (``_wait_for_ready_line``) and out-of-process consumers that tail a log
    file (e.g. a detached-gateway supervisor) can share one source of truth
    for the wire contract.

    ``line`` must start with ``READY_PREFIX``; the primitive verifies this
    itself so it owns the whole wire contract (callers just hand it candidate
    lines). Returns the parsed payload dict.

    Raises:
        ``GatewaySpawnError`` on a missing prefix, malformed JSON, a non-dict
        payload, or a missing required key (``port`` / ``token``). Surfacing
        these as the harness's own error type keeps the
        always-``GatewaySpawnError`` contract callers rely on instead of
        leaking ``JSONDecodeError`` / ``KeyError``. A gateway version drift
        that drops one of these keys is exactly the protocol shift the harness
        should surface clearly.
    """
    if not line.startswith(READY_PREFIX):
        raise GatewaySpawnError(f"line does not start with {READY_PREFIX}: {line!r}")
    try:
        payload = json.loads(line[len(READY_PREFIX) :])
    except json.JSONDecodeError as exc:
        raise GatewaySpawnError(f"malformed {READY_PREFIX} line: {line!r} ({exc})") from exc
    if not isinstance(payload, dict):
        raise GatewaySpawnError(
            f"{READY_PREFIX} payload was {type(payload).__name__}, " f"expected dict: {line!r}"
        )
    for required_key in ("port", "token"):
        if required_key not in payload:
            raise GatewaySpawnError(
                f"{READY_PREFIX} payload missing required key " f"{required_key!r}: {line!r}"
            )
    return payload


class _StdoutPump:
    """One daemon reader over the child's stdout, for the whole run.

    Why a thread and not ``selectors``: ``selectors.DefaultSelector()`` is
    select()-based on Windows and accepts SOCKETS only, so registering a
    subprocess PIPE there raises rather than polling it. A reader thread behaves
    identically on all three platforms.

    Why the queue: the thread would otherwise read ahead without bound, and the
    READY waiter needs a *bounded* wait so the deadline is checked even while
    the child is alive and silent -- the one guarantee the old selector poll
    provided. ``queue.Queue.get(timeout=...)`` gives exactly that, without a
    busy loop, and it does not spin on EOF the way a selector does (EOF is
    permanently readable).

    Why one thread for the whole run and not one per phase: two readers on one
    pipe race for the same bytes. Before READY, chunks go to the queue for the
    waiter. After :meth:`handoff`, the same thread keeps draining -- which is
    what stops the gateway blocking on a full stdout pipe partway through a long
    run -- into a bounded ring buffer, so a multi-minute run's whole stdout is
    not retained.
    """

    #: Chunks retained after handoff, for post-mortem diagnostics of a failed
    #: run. Matches the previous ``deque(maxlen=256)`` drain.
    TAIL_CHUNKS = 256

    def __init__(self, stdout: IO[bytes]) -> None:
        self._stdout = stdout
        self._queue: queue.Queue[Optional[bytes]] = queue.Queue()
        self._tail: deque[bytes] = deque(maxlen=self.TAIL_CHUNKS)
        self._handed_off = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        try:
            while True:
                chunk = self._stdout.read1(4096)  # type: ignore[attr-defined]
                if not chunk:
                    return
                if self._handed_off.is_set():
                    self._tail.append(chunk)
                else:
                    self._queue.put(chunk)
        except (OSError, ValueError):
            # Pipe closed under us (teardown races the read). Nothing to
            # report: the waiter's deadline and ``proc.poll()`` own the verdict.
            return
        finally:
            # EOF sentinel, so the waiter can stop expecting bytes without
            # having to poll the thread's liveness.
            self._queue.put(None)

    def get(self, timeout: float) -> Optional[bytes]:
        """Next stdout chunk, ``None`` at EOF, raising ``queue.Empty`` on timeout."""
        return self._queue.get(timeout=timeout)

    def drain_until_eof(self, timeout: float) -> bytes:
        """Everything the reader captures up to EOF, bounded by ``timeout``.

        Used on child exit. The child is gone, so its write end is closed and
        the pipe EOFs promptly -- but the reader thread may not have handed the
        last chunk over yet, and a READY line already in the pipe must not be
        reported as "exited before READY". A plain non-blocking drain loses that
        race; waiting for the sentinel does not.
        """
        deadline = time.monotonic() + timeout
        chunks: list[bytes] = []
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                chunk = self._queue.get(timeout=min(remaining, 0.05))
            except queue.Empty:
                continue
            if chunk is None:
                break
            chunks.append(chunk)
        return b"".join(chunks)

    def handoff(self) -> None:
        """Stop queueing; keep draining the pipe into the bounded tail."""
        self._handed_off.set()

    def tail_text(self) -> str:
        """The post-handoff stdout tail, decoded for a diagnostic message."""
        return b"".join(self._tail).decode("utf-8", errors="replace")


def fake_acp_backend_launcher(directory: Path) -> Path:
    """The path to hand ``KIROCREW_KIRO_BIN`` so the gateway spawns the packaged
    fake ACP backend, on every platform.

    On POSIX that is ``fake_acp_backend.__file__`` itself: the module carries a
    shebang and is exec'd in place. Windows has no shebang, so ``CreateProcess``
    refuses a ``.py`` path outright; it DOES run a ``.cmd`` (handed to the
    command interpreter), and ``.cmd`` is in ``platform_compat``'s
    runnable-suffix set, so the candidate also survives ``is_executable_file``.
    ``build.yml`` stages its Windows backend payload as ``kirocrew.cmd`` for the
    same reason.

    Named ``kiro-backend.cmd`` ON PURPOSE, not ``kiro-cli.exe``: delegation to
    Kiro CLI's own sandbox must come from the reviewed ACP call site passing
    ``is_kiro_cli=True``, never from the basename, so a launcher whose name
    cannot satisfy ``_spawns_kiro_cli`` proves the classification is what grants
    it. ``directory`` must outlive the gateway that runs the shim.
    """
    from kiro_crew.testing import fake_acp_backend

    backend = Path(str(fake_acp_backend.__file__))
    if not platform_compat.IS_WINDOWS:
        return backend
    launcher = Path(directory) / "kiro-backend.cmd"
    launcher.write_text(
        "@echo off\r\n" f'"{sys.executable}" -m kiro_crew.testing.fake_acp_backend %*\r\n',
        encoding="ascii",
    )
    return launcher


def _wait_for_ready_line(
    proc: subprocess.Popen[bytes],
    *,
    timeout: float,
    stderr_buffer: list[bytes],
    pump: Optional[_StdoutPump] = None,
) -> dict[str, Any]:
    """Read stdout until we see ``KIROCREW_READY:{...}`` or hit the timeout.

    Consumes chunks from a :class:`_StdoutPump` rather than reading the pipe
    directly, so the deadline is enforced even when the subprocess is alive but
    silent. A blocking ``stdout.readline()`` on a gateway stuck in a network
    call -- writing nothing and not exiting -- would hang the harness
    indefinitely, contradicting the documented 60s guarantee.

    Child exit during the wait is an IMMEDIATE failure carrying the stderr tail
    (populated by the caller's drain thread), not a wait to the deadline. The
    already-queued stdout is drained and scanned for READY first: a child that
    printed READY and exited in the same breath did emit it, and reporting that
    as "exited before READY" would be a wall-clock race rather than a verdict.

    ``pump`` lets the caller keep the reader alive past this function (see
    ``spawn_feature_gateway``, which hands it off to the bounded drain). When
    omitted, one is created and owned here.

    Raises ``GatewaySpawnError`` on timeout, early exit, or malformed payload.
    Returns the parsed READY-line dict.
    """
    deadline = time.monotonic() + timeout
    stdout = proc.stdout
    if stdout is None:  # defensive — Popen always wires stdout when PIPE
        raise GatewaySpawnError("subprocess stdout is not piped")

    if pump is None:
        pump = _StdoutPump(stdout)
        pump.start()

    def _stderr_tail() -> str:
        return b"".join(stderr_buffer).decode("utf-8", errors="replace")[-4000:]

    def _scan(buffer: bytes) -> tuple[Optional[dict[str, Any]], bytes]:
        """Pull whole lines off ``buffer``; return the READY payload if seen."""
        while b"\n" in buffer:
            line_bytes, buffer = buffer.split(b"\n", 1)
            line = line_bytes.decode("utf-8", errors="replace").strip()
            if line.startswith(READY_PREFIX):
                return parse_ready_line(line), buffer
        return None, buffer

    buf = b""
    while True:
        if proc.poll() is not None:
            # Scan whatever the reader captures up to EOF before declaring the
            # exit premature (see the docstring's race note). Bounded: the child
            # is already gone, so its write end is closed and EOF is imminent.
            payload, buf = _scan(buf + pump.drain_until_eof(EXIT_DRAIN_SECONDS))
            if payload is not None:
                return payload
            raise GatewaySpawnError(
                f"gateway subprocess exited with code {proc.returncode} "
                f"before emitting {READY_PREFIX} line.\n"
                f"--- stderr (last) ---\n{_stderr_tail()}"
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise GatewaySpawnError(
                f"gateway did not emit {READY_PREFIX} line within "
                f"{timeout:.1f}s. Override with KIROCREW_HARNESS_READY_TIMEOUT.\n"
                f"--- stderr (last) ---\n{_stderr_tail()}"
            )
        # Cap each wait at 0.5s so the deadline + poll() checks above run
        # frequently even if the subprocess goes silent.
        try:
            chunk = pump.get(timeout=min(remaining, 0.5))
        except queue.Empty:
            continue  # poll interval elapsed — re-check deadline & proc
        if chunk is None:
            # EOF on stdout without READY. The child may still be alive (it can
            # close stdout and keep running), so keep honouring the deadline and
            # the exit check rather than failing here; the queue is empty from
            # now on, so each iteration sleeps its 0.5s instead of spinning.
            continue
        payload, buf = _scan(buf + chunk)
        if payload is not None:
            return payload


def _drain_stderr(stderr: IO[bytes], buffer: list[bytes]) -> None:
    """Continuously read from stderr into ``buffer``.

    The gateway emits ~30+ WARNING lines on startup (config-loader meta-key
    spam, MCP probe failures for unconfigured servers). They all go to
    stderr, so without a drainer the pipe fills, the subprocess blocks on
    write, and we deadlock. Buffer the contents so we can surface them
    on failure.
    """
    while True:
        chunk = stderr.read1(4096)  # type: ignore[attr-defined]
        if not chunk:
            return
        buffer.append(chunk)


def terminate_pgid(
    pid: int,
    *,
    grace: Optional[float] = None,
    wait: Optional[Callable[[float], object]] = None,
) -> None:
    """SIGTERM a process group by pid, escalate to SIGKILL after ``grace``.

    POSIX ONLY. Process groups, ``setsid`` and ``killpg`` do not exist on
    Windows; ``_terminate_process_group`` routes that platform through
    ``platform_compat.kill_process_tree`` instead. Calling this on Windows
    raises ``AttributeError`` from ``os.getpgid`` -- deliberately not guarded,
    because a caller reaching here on Windows has the wrong teardown primitive
    and should say so loudly.

    pid-based (no ``Popen`` handle required) so out-of-process supervisors —
    e.g. a ``stop`` script tearing down a detached gateway it did not itself
    spawn — share the same teardown semantics as the in-process harness. The
    target must be a process-group leader (spawn with
    ``start_new_session=True``).

    Args:
        pid: Process-group leader pid.
        grace: Seconds between SIGTERM and SIGKILL. Defaults to the module
            constant ``TERMINATE_GRACE_SECONDS``, resolved at CALL time (not
            import time) so tests that patch the constant are honored.
        wait: Optional exit-detection hook, called as ``wait(grace)``; it
            should return when the target has exited or raise
            ``subprocess.TimeoutExpired`` when the grace window elapses (i.e.
            ``proc.wait``'s contract). Callers that own the ``Popen`` SHOULD
            pass ``proc.wait`` — handle-based detection sees the exit
            immediately even while the child is an unreaped zombie, which the
            default pid poll cannot distinguish from a live process. Without a
            hook, liveness is polled on the whole GROUP (``os.killpg(pgid,
            0)``) so a fast-exiting leader with lingering children still
            escalates to the group SIGKILL.

    Does NOT reap: a supervisor that owns the ``Popen`` should ``proc.wait()``
    afterwards; an out-of-process caller relies on init reaping the reparented
    child. No-op if the process is already gone. ``PermissionError`` (pid
    recycled to another user, or reduced privilege) aborts the teardown with a
    logged diagnostic rather than crashing or silently no-opping.
    """
    if grace is None:
        grace = TERMINATE_GRACE_SECONDS
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return  # already gone
    except PermissionError:
        _LOGGER.warning(
            "terminate_pgid(%d): getpgid denied (pid recycled to another "
            "user?) — aborting teardown",
            pid,
        )
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except PermissionError:
        _LOGGER.warning(
            "terminate_pgid(%d): SIGTERM to pgid %d denied — target not "
            "signalable by this user, teardown skipped",
            pid,
            pgid,
        )
        return

    if wait is not None:
        # Handle-based detection: returns the moment the child exits (even as
        # an unreaped zombie), so graceful teardowns don't burn the full grace
        # window the way a pid poll would.
        leader_exited = False
        with contextlib.suppress(subprocess.TimeoutExpired):
            wait(grace)
            leader_exited = True
        if leader_exited:
            # Leader is gone, but group children may linger (holding the
            # ephemeral port). One group probe; sweep if anything remains.
            try:
                os.killpg(pgid, 0)
            except (ProcessLookupError, PermissionError):
                return  # whole group gone (or no longer ours) — done
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(pgid, signal.SIGKILL)
            return
    else:
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            try:
                # Group liveness, not leader liveness: a reparented leader can
                # exit (and be reaped by init) while a SIGTERM-ignoring child
                # keeps the ephemeral port open — the exact tree-outlives-
                # parent case the group kill exists for.
                os.killpg(pgid, 0)
            except ProcessLookupError:
                return  # whole group exited within the grace window
            except PermissionError:
                # pgid recycled to another user mid-window: our group is gone
                # and SIGKILLing would hit an unrelated group. Stop here.
                _LOGGER.warning(
                    "terminate_pgid(%d): pgid %d no longer signalable during "
                    "grace poll (recycled?) — skipping SIGKILL escalation",
                    pid,
                    pgid,
                )
                return
            time.sleep(0.05)

    # Still alive after the grace window — escalate.
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except PermissionError:
        _LOGGER.warning(
            "terminate_pgid(%d): SIGKILL to pgid %d denied — group may " "still be running",
            pid,
            pgid,
        )


def _terminate_process_tree_windows(proc: subprocess.Popen[bytes]) -> None:
    """Tear down the gateway's tree on Windows via ``taskkill /T /F``.

    Windows has no ``setsid`` / ``killpg``, so ``terminate_pgid``'s whole
    mechanism is unavailable; ``platform_compat.kill_process_tree`` is the
    repo's one helper for this (see platform-compat.md). It walks the real
    parent/child tree, which is what reaches the gateway's MCP servers and
    kiro-cli children -- ``proc.terminate()`` would signal only the parent and
    leave them holding the ephemeral port.

    There is no graceful step to attempt first: ``taskkill /F`` is a hard
    terminate, and Windows offers no SIGTERM equivalent deliverable to a child
    that shares no console. The child therefore gets no shutdown budget on this
    platform, which is acceptable for a throwaway test gateway and is why the
    grace constant is only used for the reap below.
    """
    try:
        platform_compat.kill_process_tree(proc.pid, platform_compat.SIGTERM)
    except ProcessLookupError:
        pass  # already gone between the poll above and here
    except (PermissionError, OSError) as exc:
        # A protected descendant or a transient access denial. Log rather than
        # raise: teardown must not turn a passing test red, and the reap below
        # still reports whether the parent actually died.
        _LOGGER.warning("kill_process_tree(%d) failed during teardown: %s", proc.pid, exc)
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=TERMINATE_GRACE_SECONDS)


def _terminate_process_group(proc: subprocess.Popen[bytes]) -> None:
    """Tear down the gateway's whole process tree, then reap.

    The gateway spawns child processes (MCP servers, kiro-cli sessions,
    secretary). ``proc.terminate()`` only signals the parent;
    children can outlive it and hold the ephemeral port or cache files open.

    POSIX delegates the SIGTERM→SIGKILL group kill to ``terminate_pgid`` (shared
    with out-of-process supervisors), passing ``proc.wait`` as the
    exit-detection hook — handle-based detection returns the instant the
    child exits (even before it's reaped), so a graceful SIGTERM teardown
    doesn't burn the full grace window the way a pid poll would. Then
    ``proc.wait()`` reaps so the child doesn't linger as a zombie.
    ``TERMINATE_GRACE_SECONDS`` is read at call time so tests can patch it.

    Windows takes ``_terminate_process_tree_windows`` instead: process groups
    are a POSIX concept and the group primitives do not exist there.
    """
    if proc.poll() is not None:
        return  # already exited
    if platform_compat.IS_WINDOWS:
        _terminate_process_tree_windows(proc)
        return
    terminate_pgid(
        proc.pid,
        grace=TERMINATE_GRACE_SECONDS,
        wait=lambda timeout: proc.wait(timeout=timeout),
    )
    # The hook reaps on graceful exit; after a SIGKILL escalation the child
    # still needs reaping so it doesn't linger as a zombie.
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=TERMINATE_GRACE_SECONDS)


@contextlib.contextmanager
def spawn_feature_gateway(
    fixture: str = "minimal",
    approval: str = "reads",
    *,
    crons: bool = False,
    timeout: Optional[float] = None,
) -> Iterator[GatewayHandle]:
    """Spin up an isolated gateway from the current workspace checkout.

    Args:
        fixture: Named fixture (``empty`` / ``minimal`` / ``rich``) passed
            through to ``kirocrew gateway --seed`` so the tmp KIROCREW_HOME
            is populated atomically with gateway startup. A bad name causes
            the gateway to exit before READY; the readline loop surfaces
            seed's stderr in a ``GatewaySpawnError``.
        approval: Approval mode to pass through ``--approval``. ``"reads"``
            (default) auto-approves a conservative set of read verbs;
            ``"yolo"`` auto-approves everything but requires an isolated
            ``KIROCREW_HOME`` (the harness always provides one); pass
            ``"interactive"`` only if the test will drive the approval
            UI itself.
        crons: When ``False`` (default) the harness passes ``--no-crons``
            to suppress all scheduled jobs — the safe default since stray
            cron fires can pollute unrelated tests' state. Set ``True`` to
            keep cron scheduling enabled (e.g. tests that exercise
            ``cron_add`` end-to-end and need the scheduler thread alive).
        timeout: Override the ready-line timeout in seconds. Falls back to
            ``KIROCREW_HARNESS_READY_TIMEOUT`` env var, then to
            ``DEFAULT_READY_TIMEOUT``.

    Yields:
        ``GatewayHandle`` once the gateway has bound its dashboard port
        and emitted the ``KIROCREW_READY:{...}`` line. The handle is only
        valid inside the ``with`` block; on exit the subprocess is
        terminated and ``handle.home`` is removed.

    Raises:
        ``GatewaySpawnError`` if the gateway exits before READY, doesn't
        emit READY within the timeout, or can't seed the fixture.
    """
    src = _resolve_workspace_src()
    home = Path(tempfile.mkdtemp(prefix="kirocrew-harness-"))
    # Outer try/finally so ``home`` is always cleaned up, even if
    # ``subprocess.Popen`` raises before we reach the inner block (bad
    # ``sys.executable``, fd exhaustion, fork failure, ...).
    try:
        if timeout is None:
            env_timeout = os.environ.get("KIROCREW_HARNESS_READY_TIMEOUT")
            timeout = float(env_timeout) if env_timeout else DEFAULT_READY_TIMEOUT

        env = {
            **os.environ,
            "PYTHONPATH": str(src) + os.pathsep + os.environ.get("PYTHONPATH", ""),
            "KIROCREW_HOME": str(home),
            # Isolate the AGENT-SPEC home too, not just the data home (issue #4912).
            # ``kirocrew gateway`` boot runs ``rebuild_agent_config``, which writes the
            # managed MCP specs into ``kiro_agents_dir()``. Left at the default that
            # resolves the operator's real machine-wide ``~/.kiro/agents`` -- and the
            # spawned gateway is an ordinary (non-worktree) install, so ``agent.py``'s
            # write guard takes the "writing its own shared home" branch and does NOT
            # decline. It would then stamp this throwaway checkout's venv + tmp data
            # home into the real install's specs, breaking every managed MCP call on
            # the machine until the next gateway restart heals it.
            #
            # Pointing ``KIRO_HOME`` at ``<home>/kiro`` makes ``kiro_agents_dir()``
            # resolve to ``<home>/kiro/agents`` -- which is EXACTLY
            # ``isolated_agents_dir(<home>)``, the dedicated dir the write guard's
            # private-target exemption already lets this instance own. The whole tree
            # is removed with ``home`` on teardown, so the gateway writes only its own
            # specs and leaves the shared install untouched.
            "KIRO_HOME": str(home / "kiro"),
            # Marks this gateway as a test rig. It grants no launch privilege —
            # the packaged fake backend is exec'd by the ordinary in-place path
            # like any other runnable executable.
            FAKE_ACP_TEST_MODE_ENV: "1",
            # Force unbuffered Python so we see READY without waiting for the
            # next flush. ``--json-ready`` already calls ``flush=True`` on the
            # READY print itself, but other prints leading up to it (e.g.
            # "Created default config") would block-buffer when stdout is a
            # pipe and could mask early failures.
            "PYTHONUNBUFFERED": "1",
            # Embeddings are default-on; never let a harness-spawned gateway
            # kick the 610MB embedding-model download during a test run.
            "KIROCREW_SKIP_MODEL_DOWNLOAD": "1",
        }

        cmd = [
            sys.executable,
            "-m",
            "kiro_crew",
            "gateway",
            "--test-mode",
            # ``--seed`` populates the (empty) tmp KIROCREW_HOME from the named
            # fixture (empty / minimal / rich) before binding the dashboard.
            # Atomic with the gateway start: a bad fixture name → gateway exits
            # with seed's exit code before READY, which the readline loop below
            # surfaces as a GatewaySpawnError with stderr.
            "--seed",
            fixture,
            "--approval",
            approval,
        ]
        if not crons:
            # Suppress scheduled jobs by default — a stray cron firing during
            # an unrelated test is a hard-to-diagnose source of flakes. Tests
            # that specifically exercise the cron path opt back in via
            # ``crons=True``.
            cmd.append("--no-crons")
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # Process-group isolation so teardown can reap the whole tree and
            # child MCP servers / kiro-cli sessions don't outlive their parent
            # holding ports open. The two kwargs are passed EXPLICITLY (never
            # **unpacked) per platform-compat.md: on POSIX start_new_session
            # calls setsid so killpg reaps the group and creationflags=0 is a
            # no-op; on Windows there is no setsid and CREATE_NEW_PROCESS_GROUP
            # is what makes the tree taskkill /T-reapable.
            start_new_session=platform_compat.IS_POSIX,
            creationflags=platform_compat.CREATE_NEW_PROCESS_GROUP,
        )

        # Drain stderr asynchronously into a list so the buffer can't fill
        # and deadlock the subprocess. Using a daemon thread is fine: it
        # exits when the process closes its stderr.
        stderr_buffer: list[bytes] = []
        if proc.stderr is not None:
            drainer = threading.Thread(
                target=_drain_stderr, args=(proc.stderr, stderr_buffer), daemon=True
            )
            drainer.start()

        # ONE reader owns stdout for the whole run. It feeds the READY waiter
        # first, then keeps draining into a bounded tail -- the READY loop stops
        # consuming once it sees the sentinel, and without a continuing drain the
        # gateway blocks on a full stdout pipe partway through a long run
        # (per-turn logging fills the ~64KB buffer), stalling the event loop into
        # connection-refused for every later request. A second thread on the same
        # pipe would race this one for the same bytes, so the pump switches
        # destinations instead of being replaced.
        pump: Optional[_StdoutPump] = None
        if proc.stdout is not None:
            pump = _StdoutPump(proc.stdout)
            pump.start()

        try:
            ready = _wait_for_ready_line(
                proc, timeout=timeout, stderr_buffer=stderr_buffer, pump=pump
            )
            if pump is not None:
                pump.handoff()
            port = int(ready["port"])
            token = str(ready["token"])
            url = f"http://localhost:{port}/?token={token}"

            def _diagnostics() -> str:
                err = b"".join(stderr_buffer).decode("utf-8", errors="replace")[-4000:]
                out = ""
                if pump is not None:
                    out = pump.tail_text()[-2000:]
                return (
                    f"gateway pid {proc.pid} exit={proc.poll()!r}\n"
                    f"--- stderr (last) ---\n{err}\n"
                    f"--- stdout after READY (last) ---\n{out}"
                )

            handle = GatewayHandle(
                url=url,
                port=port,
                token=token,
                home=home,
                proc=proc,
                _diagnostics=_diagnostics,
            )
            yield handle
        finally:
            _terminate_process_group(proc)
    finally:
        # Clean up tmp home regardless of how we exited. ``ignore_errors``
        # because pytest cleanup races with nested file handles on macOS
        # and the test harness shouldn't fail teardown on stale FDs.
        shutil.rmtree(home, ignore_errors=True)
