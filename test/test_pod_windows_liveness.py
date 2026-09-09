"""Two fail-OPEN holes the Windows pod backend closes.

Both are the same shape: a Windows-only path answered "nothing is running here"
when it could not tell, and teardown then deleted state a live gateway owned.
They are asserted on every platform because both fixes are platform-neutral
Python over one primitive, and the primitive's own per-platform behaviour is
already covered by ``test_platform_compat``.
"""

from __future__ import annotations

import os
import subprocess
import threading
import types

import pytest

from kiro_crew import platform_compat
from kiro_crew.pod import runtime as rt
from kiro_crew.pod import windows as win
from kiro_crew.pod.config import EXIT_REFUSED_UNRECOVERABLE, PodConfig


def _cp(stdout: str = "", returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


@pytest.fixture
def cfg(tmp_path, monkeypatch) -> PodConfig:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("KIROCREW_POD_ROOT", str(tmp_path / "pods"))
    monkeypatch.setenv("KIROCREW_POD_ENV_DIR", str(tmp_path / "pods-env"))
    monkeypatch.setenv("KIROCREW_POD_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    c = PodConfig.load()
    c.pods_dir.mkdir(parents=True, exist_ok=True)
    return c


# --------------------------------------------------------------------------
# The per-name mutex is a real lock on every platform
# --------------------------------------------------------------------------
def test_the_mutex_locks_through_the_shared_cross_platform_helper(cfg, monkeypatch):
    """Pinned at the primitive, because that is what makes Windows serialize.

    A pod plane on Windows is live, so an unguarded `pod up` racing a `pod down`
    on one name lets teardown stop the replacement and delete its home. The lock
    has to come from the helper that implements both platforms, and the fd has to
    come from the non-truncating open: `open(path, "w")` empties the file before
    any lock is held, which on Windows leaves a contender locking an emptied file.
    """
    seen: dict[str, object] = {}
    real_open = rt.open_lock_file
    real_lock = rt.file_lock

    def spy_open(path):
        seen["path"] = str(path)
        return real_open(path)

    def spy_lock(fd, **kwargs):
        seen["fd"] = fd
        seen["exclusive"] = kwargs.get("exclusive")
        return real_lock(fd, **kwargs)

    monkeypatch.setattr(rt, "open_lock_file", spy_open)
    monkeypatch.setattr(rt, "file_lock", spy_lock)
    with rt.pod_name_mutex(cfg, "demo"):
        pass
    assert seen["path"] == str(cfg.pods_dir / f"{cfg.unit_prefix}@demo.lock")
    assert isinstance(seen["fd"], int)
    assert seen["exclusive"] is True


def test_two_contenders_on_one_name_serialize_under_windows_semantics(cfg, monkeypatch):
    """The property the fix exists for, driven through the Windows branch.

    ``file_lock`` chooses its implementation from ``IS_POSIX``, so forcing that
    False runs the win32 acquire path: a spin on the non-blocking primitive that
    fails CLOSED. Two threads cross a barrier before either enters, and the
    critical sections must not interleave.
    """
    monkeypatch.setattr(platform_compat, "IS_POSIX", False)
    holders: list[int] = []
    events: list[str] = []
    barrier = threading.Barrier(2)
    lock = threading.Lock()

    def fake_win_acquire(fd, timeout=None):
        # Stand in for msvcrt.locking's byte-range lock, which Linux cannot run.
        # The argv shape is pinned: the acquire takes the fd it was handed.
        assert isinstance(fd, int)
        deadline = 200
        while deadline:
            if lock.acquire(blocking=False):
                return True
            deadline -= 1
            import time as _t

            _t.sleep(0.005)
        return False

    monkeypatch.setattr(platform_compat, "_win_acquire_blocking", fake_win_acquire)
    # msvcrt is imported only on win32, so the release path needs a stand-in here.
    monkeypatch.setattr(
        platform_compat,
        "msvcrt",
        types.SimpleNamespace(locking=lambda *a: None, LK_UNLCK=0),
        raising=False,
    )

    def contend(tag: str) -> None:
        barrier.wait(timeout=10)
        with rt.pod_name_mutex(cfg, "demo"):
            events.append(f"enter:{tag}")
            holders.append(len(holders) + 1)
            import time as _t

            _t.sleep(0.05)
            events.append(f"exit:{tag}")
        lock.release()

    threads = [threading.Thread(target=contend, args=(t,)) for t in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)
    assert not any(t.is_alive() for t in threads), "both contenders must finish"
    # No interleaving: every enter is immediately followed by its own exit.
    assert [e.split(":")[0] for e in events] == ["enter", "exit", "enter", "exit"], events


def test_the_mutex_stays_reentrant_within_one_thread(cfg):
    """The CLI holds it across a transaction while start_pod re-acquires inside."""
    with rt.pod_name_mutex(cfg, "demo"):
        with rt.pod_name_mutex(cfg, "demo"):
            pass
    assert (cfg.pods_dir / f"{cfg.unit_prefix}@demo.lock").exists()


def test_a_stuck_holder_refuses_rather_than_running_unserialized(cfg, monkeypatch):
    """``file_lock`` fails CLOSED, and the mutex must not swallow that."""
    monkeypatch.setattr(platform_compat, "IS_POSIX", False)
    monkeypatch.setattr(platform_compat, "_win_acquire_blocking", lambda fd, timeout=None: False)
    with pytest.raises(OSError, match="refusing to proceed unserialized"):
        with rt.pod_name_mutex(cfg, "demo"):
            pytest.fail("the critical section must not be entered without the lock")


# --------------------------------------------------------------------------
# A pid record that cannot be written is a boot failure
# --------------------------------------------------------------------------
def test_an_unwritable_pid_record_terminates_the_child_and_exits_nonzero(
    cfg, monkeypatch, tmp_path, capsys
):
    """The record IS the pod's liveness, so a pod without one must not run.

    With no record every reader calls the pod stopped, and the first ``pod down``
    deletes its task and isolated HOME while the gateway serves, reporting rc=0.
    """
    killed: list[tuple[int, str]] = []
    reaped: list[str] = []

    class FakeProc:
        pid = os.getpid()

        def kill(self):
            killed.append((self.pid, "popen"))

        def wait(self, timeout=None):
            reaped.append("wait")
            return 0

    monkeypatch.setattr(win.subprocess, "Popen", lambda argv, **kw: FakeProc())
    # Pinned alongside the fake Popen: on macOS ``process_start_time`` shells out
    # to ``ps`` through the same ``subprocess.Popen`` and would receive FakeProc.
    monkeypatch.setattr(win, "process_start_time", lambda pid: "1234567")
    monkeypatch.setattr(win, "apply_windows_resource_ceiling", lambda pid: True)
    monkeypatch.setattr(win, "resume_process_main_thread", lambda pid: True)
    monkeypatch.setattr(
        win,
        "record_supervised_pid",
        lambda c, n, p: (_ for _ in ()).throw(OSError("read-only file system")),
    )
    monkeypatch.setattr(
        win,
        "kill_process_tree_pinned",
        lambda pid, token, sig=None: killed.append((pid, "tree")) or True,
    )
    monkeypatch.setattr(
        win, "stop", lambda *a, **k: pytest.fail("the boot path must not delete the task")
    )

    rc = win.supervise_gateway(cfg, "demo", tmp_path / "kirocrew", ["gateway"], {})

    assert rc == EXIT_REFUSED_UNRECOVERABLE
    assert any(kind == "tree" for _pid, kind in killed), "the tree kill must be pinned, not bare"
    assert reaped, "the terminated child must be reaped"
    out = capsys.readouterr().out
    assert str(win.pid_record_path(cfg, "demo")) in out, "the message must name the record path"
    assert "read-only file system" in out
    # No record is left for a reader to trust, and nothing was torn down.
    assert win.supervised_pid(cfg, "demo") is None
    assert not win.pid_record_path(cfg, "demo").exists()


def test_a_fragment_left_by_a_failed_record_write_is_cleared(cfg, monkeypatch, tmp_path):
    """A partial record is worse than none: it names a pid nothing can attribute."""
    fragment = win.pid_record_path(cfg, "demo")

    class FakeProc:
        pid = os.getpid()

        def kill(self):
            return None

        def wait(self, timeout=None):
            return 0

    def half_write(c, n, p):
        fragment.parent.mkdir(parents=True, exist_ok=True)
        fragment.write_text(f"{p}\n")
        raise OSError("no space left on device")

    monkeypatch.setattr(win.subprocess, "Popen", lambda argv, **kw: FakeProc())
    # Pinned alongside the fake Popen: on macOS ``process_start_time`` shells out
    # to ``ps`` through the same ``subprocess.Popen`` and would receive FakeProc.
    monkeypatch.setattr(win, "process_start_time", lambda pid: "1234567")
    monkeypatch.setattr(win, "apply_windows_resource_ceiling", lambda pid: True)
    monkeypatch.setattr(win, "resume_process_main_thread", lambda pid: True)
    monkeypatch.setattr(win, "record_supervised_pid", half_write)
    monkeypatch.setattr(win, "kill_process_tree_pinned", lambda pid, token, sig=None: True)

    assert win.supervise_gateway(cfg, "demo", tmp_path / "kirocrew", ["gateway"], {}) != 0
    assert not fragment.exists()


def test_record_supervised_pid_raises_instead_of_swallowing(cfg, monkeypatch):
    """The primitive itself must surface the failure to its one caller."""

    def boom(*a, **k):
        raise OSError("access is denied")

    monkeypatch.setattr(win.Path, "write_text", boom)
    with pytest.raises(OSError, match="access is denied"):
        win.record_supervised_pid(cfg, "demo", os.getpid())


# --------------------------------------------------------------------------
# stop(): the symmetric case
# --------------------------------------------------------------------------
def test_stop_refuses_when_a_recorded_pid_is_alive_but_unattributable(cfg, monkeypatch):
    """ "Cannot tell" must not be rendered as "not running" on the teardown path.

    A record naming a LIVE pid whose creation-time identity does not match is
    what a fragment or a recycled pid looks like. Deleting the task there hands
    the caller an rc=0 it reclaims the HOME on, out from under a live process.
    """
    win.pid_record_path(cfg, "demo").write_text(f"{os.getpid()}\nnot-the-real-token\n")
    calls: list[str] = []
    monkeypatch.setattr(win, "schtasks", lambda *a: calls.append(a[0]) or _cp())
    win.write_task_script(cfg, "demo")

    cp = win.stop(cfg, "demo")

    assert cp.returncode == 1
    assert "does not carry the creation-time identity" in cp.stderr
    assert str(win.pid_record_path(cfg, "demo")) in cp.stderr
    assert str(os.getpid()) in cp.stderr
    assert "/Delete" not in calls, "the task must not be deleted on an unprovable stop"
    assert win.task_script_path(cfg, "demo").exists(), "per-pod state must be preserved"


def test_stop_still_reclaims_a_stale_record_for_a_dead_pid(cfg, monkeypatch):
    """The ordinary hard-stop leftover must keep passing through.

    A ``/End`` reaps the wrapper before its cleanup runs, so a record naming a
    pid that is GONE is routine. Refusing on that would block every `pod down`
    after a hard stop.
    """
    win.pid_record_path(cfg, "demo").write_text("999999999\nstale-token\n")
    monkeypatch.setattr(win, "schtasks", lambda *a: _cp())
    monkeypatch.setattr(win, "pid_exists", lambda pid: False)
    win.write_task_script(cfg, "demo")

    cp = win.stop(cfg, "demo")

    assert cp.returncode == 0
    assert not win.task_script_path(cfg, "demo").exists()
    assert not win.pid_record_path(cfg, "demo").exists()


def test_stop_with_no_record_at_all_is_the_plain_stopped_path(cfg, monkeypatch):
    """A cleanly stopped pod unlinks its record, so absence is not ambiguity."""
    monkeypatch.setattr(win, "schtasks", lambda *a: _cp())
    win.write_task_script(cfg, "demo")
    assert win.stop(cfg, "demo").returncode == 0


def test_the_unattributable_probe_ignores_a_provable_record(cfg):
    """A record that PROVES itself is the live-pod path, handled before this."""
    win.record_supervised_pid(cfg, "demo", os.getpid())
    assert win._unattributable_live_pid(cfg, "demo") is None
    assert win.supervised_pid(cfg, "demo") == os.getpid()


def test_the_unattributable_probe_ignores_a_junk_record(cfg):
    win.pid_record_path(cfg, "demo").write_text("not-a-pid\n")
    assert win._unattributable_live_pid(cfg, "demo") is None
