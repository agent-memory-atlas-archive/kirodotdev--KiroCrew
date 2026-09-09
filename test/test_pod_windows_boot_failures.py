"""Three ways the Windows pod boot answered wrongly about its own gateway.

Each is a fact the canary proved on windows-latest and no fake could: the pid
that binds the port is not the pid the supervisor records, a POSIX shell script
reached ``bash`` and got WSL, and the agent-backend pin never left the plane.
All three fixes are platform-neutral Python, so these run everywhere with the
platform flag pinned.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from kiro_crew import cli as kc_cli
from kiro_crew import platform_compat
from kiro_crew.pod import runtime as rt
from kiro_crew.pod import windows as win
from kiro_crew.pod.config import PodConfig, environment_vars


@pytest.fixture
def cfg(tmp_path, monkeypatch) -> PodConfig:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("KIROCREW_POD_ROOT", str(tmp_path / "pods"))
    monkeypatch.setenv("KIROCREW_POD_ENV_DIR", str(tmp_path / "pods-env"))
    monkeypatch.setenv("KIROCREW_POD_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.delenv("KIROCREW_POD_KIRO_BIN", raising=False)
    c = PodConfig.load()
    c.pods_dir.mkdir(parents=True, exist_ok=True)
    return c


def _listener(pid: int):
    return platform_compat.PortListener(pid=pid, address="127.0.0.1", family="4")


# --------------------------------------------------------------------------
# 1. The binding pid is a CHILD of the recorded pid on Windows
# --------------------------------------------------------------------------
def test_a_descendant_holding_the_port_is_this_pod_on_windows(cfg, monkeypatch):
    """The canary's HEALTH_FOREIGN: both pids are the pod.

    A pip console script is an ``.exe`` launcher stub that starts the interpreter
    as a child and waits, so ``supervise_gateway`` records the stub while the
    child binds the port. Reporting that as foreign made `pod up` refuse a
    gateway that had booted and was answering.
    """
    monkeypatch.setattr(rt, "IS_WINDOWS", True)
    monkeypatch.setattr(rt, "_pod_recorded_pid", lambda c, n, p: 4242)
    monkeypatch.setattr(rt, "main_pid", lambda c, n: 4242)
    monkeypatch.setattr(rt, "listening_pid_tool_available", lambda: True)
    monkeypatch.setattr(rt, "find_port_listeners", lambda port: [_listener(4243)])
    monkeypatch.setattr(rt, "process_descendants", lambda pid: [4243] if pid == 4242 else [])

    assert rt.port_owner(cfg, "demo", 7999) == rt.OWNER_POD


def test_the_descendant_widening_reads_one_snapshot_of_the_recorded_pid(cfg, monkeypatch):
    """Pinned at the helper and its argument: no new matcher, no re-walk per pid."""
    monkeypatch.setattr(rt, "IS_WINDOWS", True)
    monkeypatch.setattr(rt, "_pod_recorded_pid", lambda c, n, p: 4242)
    monkeypatch.setattr(rt, "main_pid", lambda c, n: 4242)
    monkeypatch.setattr(rt, "listening_pid_tool_available", lambda: True)
    monkeypatch.setattr(rt, "find_port_listeners", lambda port: [_listener(4243)])
    asked: list[int] = []
    monkeypatch.setattr(rt, "process_descendants", lambda pid: asked.append(pid) or [4243])

    rt.port_owner(cfg, "demo", 7999)
    assert asked == [4242], "the tree is read once, from the recorded pid"


def test_a_listener_outside_the_pods_tree_is_still_foreign_on_windows(cfg, monkeypatch):
    """The widening goes DOWNWARD only; the foreign verdict keeps its teeth."""
    monkeypatch.setattr(rt, "IS_WINDOWS", True)
    monkeypatch.setattr(rt, "_pod_recorded_pid", lambda c, n, p: 4242)
    monkeypatch.setattr(rt, "main_pid", lambda c, n: 4242)
    monkeypatch.setattr(rt, "listening_pid_tool_available", lambda: True)
    monkeypatch.setattr(rt, "find_port_listeners", lambda port: [_listener(9999)])
    monkeypatch.setattr(rt, "process_descendants", lambda pid: [4243])

    assert rt.port_owner(cfg, "demo", 7999) == rt.OWNER_FOREIGN


def test_a_descendant_does_not_rescue_a_pod_with_no_supervised_pid(cfg, monkeypatch):
    """No recorded pid means no tree to widen, so the listener stays foreign."""
    monkeypatch.setattr(rt, "IS_WINDOWS", True)
    monkeypatch.setattr(rt, "_pod_recorded_pid", lambda c, n, p: None)
    monkeypatch.setattr(rt, "main_pid", lambda c, n: None)
    monkeypatch.setattr(rt, "listening_pid_tool_available", lambda: True)
    monkeypatch.setattr(rt, "find_port_listeners", lambda port: [_listener(9999)])
    monkeypatch.setattr(rt, "process_descendants", lambda pid: pytest.fail("nothing to walk from"))

    assert rt.port_owner(cfg, "demo", 7999) == rt.OWNER_FOREIGN


def test_posix_does_not_accept_a_descendant(cfg, monkeypatch):
    """POSIX execs the gateway in place, so the recorded pid IS the binder.

    Widening there would accept a child the pod spawned as the port's owner, which
    on that platform is a real foreign responder rather than the launcher artifact
    this fix exists for.
    """
    monkeypatch.setattr(rt, "IS_WINDOWS", False)
    monkeypatch.setattr(rt, "_pod_recorded_pid", lambda c, n, p: 4242)
    monkeypatch.setattr(rt, "main_pid", lambda c, n: 4242)
    monkeypatch.setattr(rt, "listening_pid_tool_available", lambda: True)
    monkeypatch.setattr(rt, "find_port_listeners", lambda port: [_listener(4243)])
    monkeypatch.setattr(rt, "process_descendants", lambda pid: pytest.fail("POSIX must not widen"))

    assert rt.port_owner(cfg, "demo", 7999) == rt.OWNER_FOREIGN


# --------------------------------------------------------------------------
# 2. No POSIX shell on the Windows gateway boot path
# --------------------------------------------------------------------------
def test_ensure_node_spawns_no_shell_on_windows(monkeypatch, tmp_path):
    """``bash`` on Windows is WSL's launcher, so this spawn printed a WSL banner.

    The pod's gateway inherits its wrapper's redirected stdout, so that UTF-16
    banner landed in the pod's own log and read as pod output while installing
    nothing. The script is POSIX-only and the repo ships no Windows twin.
    """
    script = tmp_path / "ensure-node.sh"
    script.write_text("#!/bin/sh\nexit 0\n")
    monkeypatch.setenv("KIROCREW_PROJECT_DIR", str(tmp_path))
    monkeypatch.setattr(kc_cli.platform_compat, "IS_WINDOWS", True)
    monkeypatch.setattr(kc_cli, "_node_ok", lambda: True)
    monkeypatch.setattr(
        kc_cli.subprocess, "run", lambda *a, **k: pytest.fail(f"spawned {a!r} on win32")
    )

    assert kc_cli._ensure_node(str(tmp_path)) is True


def test_ensure_node_reports_the_hosts_own_node_answer_on_windows(monkeypatch, tmp_path):
    """Skipping is not a degraded answer: `_node_ok` already sees a real Node."""
    (tmp_path / "ensure-node.sh").write_text("#!/bin/sh\nexit 0\n")
    monkeypatch.setenv("KIROCREW_PROJECT_DIR", str(tmp_path))
    monkeypatch.setattr(kc_cli.platform_compat, "IS_WINDOWS", True)
    monkeypatch.setattr(kc_cli, "_node_ok", lambda: False)
    monkeypatch.setattr(kc_cli.subprocess, "run", lambda *a, **k: pytest.fail("no spawn on win32"))

    assert kc_cli._ensure_node(str(tmp_path)) is False


def test_posix_still_runs_the_script(monkeypatch, tmp_path):
    """The POSIX behaviour is unchanged, so the guard is win32-only."""
    script = tmp_path / "ensure-node.sh"
    script.write_text("#!/bin/sh\nexit 0\n")
    monkeypatch.setenv("KIROCREW_PROJECT_DIR", str(tmp_path))
    monkeypatch.setattr(kc_cli.platform_compat, "IS_WINDOWS", False)
    seen: list[list[str]] = []

    def fake_run(argv, **kwargs):
        seen.append(list(argv))
        return subprocess.CompletedProcess(args=argv, returncode=0)

    monkeypatch.setattr(kc_cli.subprocess, "run", fake_run)
    assert kc_cli._ensure_node(str(tmp_path)) is True
    assert seen == [["bash", str(script)]]


def test_the_pod_wrapper_spawns_no_posix_shell(cfg):
    """The generated .cmd itself must name no POSIX shell.

    The wrapper is the only shell-shaped artifact the pod boot ships, so a `sh` or
    `bash` in it would put WSL on the boot path by construction rather than by a
    library call.
    """
    body = win.render_task_script(cfg, "demo")
    lowered = body.lower()
    for shell in ("bash", "wsl", "/bin/sh", " sh ", "sh -c"):
        assert shell not in lowered, f"the wrapper must not name {shell!r}"


# --------------------------------------------------------------------------
# 3. The agent-backend pin reaches the pod gateway on Windows
# --------------------------------------------------------------------------
def test_the_plane_translates_the_pod_pin_to_the_gateway_variable(monkeypatch, tmp_path):
    """The knob is `KIROCREW_POD_KIRO_BIN`; the gateway reads `KIROCREW_KIRO_BIN`."""
    fake = tmp_path / "fake-kiro.cmd"
    fake.write_text("@echo off\r\n", newline="")
    monkeypatch.setenv("KIROCREW_POD_KIRO_BIN", str(fake))
    selected = environment_vars(PodConfig.load())
    assert selected["KIROCREW_KIRO_BIN"] == str(fake)
    assert "KIROCREW_POD_KIRO_BIN" not in selected


def test_the_windows_wrapper_pins_the_agent_backend(monkeypatch, tmp_path):
    """The task carries no environment block, so the wrapper is what sets it.

    Without this the pod boots and answers health while its background session
    dies with "kiro-cli not found", which is a gateway that cannot run one agent
    turn reported as a healthy pod.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("KIROCREW_POD_ROOT", str(tmp_path / "pods"))
    monkeypatch.setenv("KIROCREW_POD_ENV_DIR", str(tmp_path / "pods-env"))
    monkeypatch.setenv("KIROCREW_POD_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    fake = tmp_path / "fake-kiro.cmd"
    fake.write_text("@echo off\r\n", newline="")
    monkeypatch.setenv("KIROCREW_POD_KIRO_BIN", str(fake))
    c = PodConfig.load()
    c.pods_dir.mkdir(parents=True, exist_ok=True)

    body = win.render_task_script(c, "demo")
    assert f'set "KIROCREW_KIRO_BIN={fake}"' in body


def test_the_wrapper_omits_the_pin_when_the_plane_did_not_ask(cfg):
    """A default plane stays byte-identical, so the seam costs nothing unused."""
    assert "KIROCREW_KIRO_BIN" not in win.render_task_script(cfg, "demo")


def test_the_booted_gateway_environment_carries_the_pin(cfg, monkeypatch, tmp_path):
    """End of the chain: what `supervise_gateway` hands the child.

    The wrapper sets the variable in the task's process environment and
    ``build_pod_env`` starts from that environment, so the gateway receives it.
    Asserted at the spawn boundary rather than in the middle, because the middle
    is where the canary's chain looked fine and the end is where it broke.
    """
    monkeypatch.setenv("KIROCREW_KIRO_BIN", str(tmp_path / "fake-kiro.cmd"))
    env = rt.build_pod_env(cfg, cfg.home_dir("demo"), 8610, tmp_path / "worktree")
    assert env["KIROCREW_KIRO_BIN"] == str(tmp_path / "fake-kiro.cmd")


def test_supervise_gateway_hands_the_pin_to_the_child(cfg, monkeypatch, tmp_path):
    """Pinned at the CreateProcess boundary: argv plus the env the child gets."""
    seen: dict[str, object] = {}

    class FakeProc:
        pid = os.getpid()

        def wait(self, timeout=None):
            return 0

    def fake_popen(argv, **kwargs):
        seen["argv"] = list(argv)
        seen["env"] = dict(kwargs.get("env") or {})
        return FakeProc()

    monkeypatch.setattr(win.subprocess, "Popen", fake_popen)
    # Pinned alongside the fake Popen: on macOS ``process_start_time`` shells out
    # to ``ps`` through the same ``subprocess.Popen`` and would receive FakeProc.
    monkeypatch.setattr(win, "process_start_time", lambda pid: "1234567")
    monkeypatch.setattr(win, "apply_windows_resource_ceiling", lambda pid: True)
    monkeypatch.setattr(win, "resume_process_main_thread", lambda pid: True)
    binary = tmp_path / "kirocrew.exe"
    fake = tmp_path / "fake-kiro.cmd"

    win.supervise_gateway(
        cfg, "demo", binary, ["gateway", "--no-crons"], {"KIROCREW_KIRO_BIN": str(fake)}
    )

    assert seen["argv"] == [str(binary), "gateway", "--no-crons"]
    assert dict(seen["env"])["KIROCREW_KIRO_BIN"] == str(fake)


def test_the_canary_pins_a_fake_backend(monkeypatch):
    """The canary must set the seam, or it asserts a pod that cannot serve a turn."""
    source = Path(__file__).resolve().parent / "test_pod_windows_boot.py"
    text = source.read_text()
    assert "KIROCREW_POD_KIRO_BIN" in text
    assert "fake_acp_backend_launcher" in text
