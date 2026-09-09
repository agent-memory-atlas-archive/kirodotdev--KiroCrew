"""Boot a REAL pod on Windows, through the real Task Scheduler, and reclaim it.

Everything in ``test_pod_windows.py`` fakes ``schtasks`` so the backend has
coverage on the Linux matrix. This file is the other half: it creates an actual
scheduled task, boots the worktree's actual gateway under it, waits for
``/api/health``, and then proves ``pod down`` leaves neither a task nor a
process. Nothing here can be established by a fake, which is why it is a
separate file with its own preconditions.

**It never skips.** A skip is how a boot canary rots: the job stays green while
the thing it was written to prove stops running. Every precondition this test
needs -- ``schtasks`` usable by the runner user, a built venv in the checkout --
is asserted with ``pytest.fail`` naming the cause, so the CI step's
``grep "1 passed"`` cannot pass on a test that did not really run.

**Not collected by the sharded Windows job.** It is listed in
``windows-collect-ignore.txt`` because it needs a built ``.venv`` that the
shards' ``uv pip install --system`` does not create, and because a five-minute
boot does not belong in a duration-balanced shard. The dedicated canary step in
``ci.yml`` names the file explicitly on the pytest command line, which bypasses
``collect_ignore`` by design (see that file's header).

Health goes over **TCP** here, unlike the Linux path's private AF_UNIX socket:
CPython on Windows has no ``AF_UNIX``. So the port is read back from ``pod up
--json`` rather than re-derived, because allocation can move a pod off its
first-preference slot and the recorded claim is the only authority.

The pod plane is rooted at a SHORT absolute path (``C:\\kcpb<hex>``) rather than
under pytest's ``tmp_path``. A deep temporary directory is how the sibling
POSIX work hit an AF_UNIX path-length ceiling; here the risk is Windows'
``MAX_PATH`` reaching the pod's own nested run/session directories, and the
mitigation is the same one: keep the root short.
"""

from __future__ import annotations

import http.client
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

from kiro_crew.platform_compat import IS_WINDOWS, process_start_time
from kiro_crew.pod import windows as win
from kiro_crew.pod.config import PodConfig
from kiro_crew.subprocess_utf8 import UTF8_TEXT
from kiro_crew.testing.harness import fake_acp_backend_launcher

# The whole module is win32-only (there is no Task Scheduler to drive elsewhere)
# AND opt-in: it registers a real scheduled task and boots a real gateway, which
# no developer should get from a bare `pytest` on a Windows box. The canary job
# sets KIROCREW_E2E_POD_WINDOWS=1 and asserts the test ran rather than skipped,
# mirroring KIROCREW_E2E_SCENARIOS for the pod scenario suite. The module is
# also on the root conftest's _HOST_SERVICE_EXEC_ALLOWED_MODULES for the same
# reason, with the rationale recorded there.
pytestmark = [
    pytest.mark.skipif(
        not IS_WINDOWS, reason="drives the real schtasks.exe and boots a real pod on win32"
    ),
    pytest.mark.skipif(
        os.environ.get("KIROCREW_E2E_POD_WINDOWS") != "1",
        reason="set KIROCREW_E2E_POD_WINDOWS=1 to register a real Task Scheduler task and boot a pod",
    ),
]

# How long the gateway gets to bind and answer. A cold first boot on a
# windows-latest runner pays for interpreter start, config seeding and the SPA
# mount, and subprocess latency there is both slow and highly variable.
HEALTH_TIMEOUT_SECS = 180.0

# The repository checkout under test. `test/` sits directly under it.
CHECKOUT = Path(__file__).resolve().parent.parent


def _venv_kirocrew() -> Path:
    """The checkout's own ``kirocrew`` entry point -- what a pod actually boots.

    A pod is deliberately the WORKTREE's binary, not the control plane's, so an
    absent venv is a missing precondition rather than something to work around.
    """
    return CHECKOUT / ".venv" / "Scripts" / "kirocrew.exe"


def _short_plane_root() -> Path:
    """A short, absolute, unique plane root on the same drive as the interpreter."""
    drive = Path(sys.executable).drive or "C:"
    return Path(f"{drive}\\kcpb{uuid.uuid4().hex[:6]}")


def _probe_health(port: int) -> int:
    """HTTP status of the pod's ``/api/health``, or 0 when nothing answers."""
    # http.client against a fixed loopback host: only the port varies, so there
    # is no scheme a dynamic value could redirect to.
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
    try:
        conn.request("GET", "/api/health")
        return int(conn.getresponse().status)
    except Exception:
        return 0
    finally:
        conn.close()


@pytest.fixture
def plane(tmp_path_factory) -> dict[str, str]:
    """A hermetic pod plane the developer's real pods cannot collide with.

    The unit prefix is unique per run, so the task lands in its own Task
    Scheduler folder and ``active_names`` cannot see another plane's pods. The
    base port is pushed above the default band for the same reason.

    ``KIROCREW_POD_KIRO_BIN`` pins the agent backend to a fake launcher, which is
    what lets the pod's gateway finish booting on a runner with no kiro-cli:
    ``config.environment_vars`` translates it to the ``KIROCREW_KIRO_BIN`` the
    gateway reads, and the Windows wrapper sets that in the task's environment.
    Without it the pod boots, answers health, and its background session dies with
    "kiro-cli not found", so the canary would be asserting a gateway that cannot
    run a single agent turn.
    """
    root = _short_plane_root()
    (root / "pods").mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        # The control plane's own data home, pinned under the plane root so the
        # `kirocrew pod` verbs' `ensure_data_home()` never touches the real one.
        "KIROCREW_HOME": str(root / "kh"),
        "KIROCREW_WORKSPACE": str(root / "kw"),
        "KIROCREW_POD_ROOT": str(root / "pods"),
        "KIROCREW_POD_ENV_DIR": str(root / "env"),
        "KIROCREW_POD_ARTIFACTS_DIR": str(root / "artifacts"),
        "KIROCREW_POD_UNIT_PREFIX": f"kcboot{uuid.uuid4().hex[:6]}",
        "KIROCREW_POD_BASE_PORT": "8610",
        "KIROCREW_POD_REPO": str(CHECKOUT),
        "KIROCREW_POD_KIRO_BIN": str(fake_acp_backend_launcher(root)),
    }
    try:
        yield env
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
def served_dist():
    """Require a served SPA bundle; never write one into the checkout.

    ``pod up`` fails loud when ``static/dist`` is missing, which is correct
    product behaviour and the wrong thing for this test to re-litigate: what is
    under test is Task Scheduler supervision, not the Vite build. The CI job
    places a one-file stand-in BEFORE pytest runs (see the canary job in
    ``ci.yml``); a developer runs ``npm run build`` or drops the same stand-in.
    The test itself writes nothing under ``src/``: a hard timeout or a worker
    kill would bypass any cleanup it promised.
    """
    index = CHECKOUT / "src" / "kiro_crew" / "static" / "dist" / "index.html"
    if not index.exists():
        pytest.fail(
            f"no served SPA bundle at {index}; build the SPA or place a one-file "
            "stand-in there before running the canary (the CI job does this)"
        )
    yield


def _pod(env: dict[str, str], *args: str, timeout: int = 300) -> subprocess.CompletedProcess:
    """Run one ``kirocrew pod`` verb through the CONTROL plane interpreter.

    The control plane is deliberately this test's own interpreter rather than the
    worktree's venv: that is the real split (a stable global ``kirocrew`` drives
    pods whose payload is the worktree's build), and it keeps a broken worktree
    from breaking the verbs that would diagnose it.
    """
    return subprocess.run(
        [sys.executable, "-m", "kiro_crew", "pod", *args],
        capture_output=True,
        timeout=timeout,
        check=False,
        env=env,
        cwd=str(CHECKOUT),
        **UTF8_TEXT,
    )


def test_a_real_pod_boots_under_task_scheduler_and_leaves_nothing_behind(plane, served_dist):
    """The end-to-end contract: up, healthy, listed, then provably gone.

    One test rather than several, because the phases are not independent: a
    booted pod is a live scheduled task plus a live gateway plus an isolated
    HOME, and splitting the assertions would leave one phase's residue to be
    reclaimed by another phase's fixture. The teardown IS half the contract.
    """
    # Precondition 1: this user can really create a task. Failing here rather
    # than skipping is the whole point -- a policy that forbids user task
    # creation makes pods impossible on the host, which is a red canary, not an
    # absent one.
    try:
        win.require_backend()
    except win.WindowsTaskError as exc:
        pytest.fail(
            "the pod boot canary cannot run because this user cannot drive Task "
            f"Scheduler: {exc}"
        )

    # Precondition 2: a pod boots the WORKTREE's own binary, so the checkout has
    # to have been built. The CI step builds it explicitly before this runs.
    if not _venv_kirocrew().exists():
        pytest.fail(
            f"no {_venv_kirocrew()} -- a pod boots the worktree's own kirocrew, so "
            "the checkout must have a built venv. Create it with "
            "`python -m venv .venv` plus an editable install before this test."
        )

    # A pod is named after a git WORKTREE: `pod up <name>` resolves the checkout
    # through `git worktree list` on KIROCREW_POD_REPO and matches the basename
    # (pod/runtime.py resolve_checkout). The checkout itself is the main
    # worktree, so its own basename is the one name guaranteed to resolve on a
    # fresh runner. A made-up name failed here on the first Windows run with
    # "no git worktree 'boot6e85ff'". The plane env above keeps the task name,
    # root and ports unique, so the fixed pod name collides with nothing.
    name = CHECKOUT.name
    cfg_env = {**plane, "KIROCREW_POD_BIN": str(_venv_kirocrew())}
    cfg = _load_cfg(cfg_env)

    # `up` sits INSIDE the teardown guard: the task is registered and the
    # gateway spawned partway through it, so a boot that times out or fails its
    # own validation must still reach `down`, or a scheduled task and a live
    # gateway outlive the test (rmtree of the plane root reclaims neither).
    try:
        up = _pod(cfg_env, "up", name, "--seed", "minimal", "--json")
        assert up.returncode == 0, f"pod up failed:\n{up.stdout}\n{up.stderr}"
        payload = json.loads(up.stdout)
        # Read the port BACK rather than re-deriving it: allocation can move a pod
        # off its first-preference slot, and the recorded claim is the authority.
        port = int(payload["port"])

        deadline = time.monotonic() + HEALTH_TIMEOUT_SECS
        code = 0
        while time.monotonic() < deadline:
            code = _probe_health(port)
            # 401/403 mean serving but gated, which still proves it is up.
            if code in (200, 401, 403):
                break
            time.sleep(2)
        assert code in (200, 401, 403), (
            f"the pod never answered /api/health on 127.0.0.1:{port} within "
            f"{HEALTH_TIMEOUT_SECS:.0f}s (last status {code}). Its own log:\n"
            f"{win.recent_journal(cfg, name, 80)}"
        )

        # The service manager's own view: a task really exists for this pod.
        assert win.task_exists(cfg, name), (
            f"the pod is serving but Task Scheduler has no task at " f"{win.task_name(cfg, name)}"
        )

        listed = _pod(cfg_env, "ls")
        assert listed.returncode == 0, listed.stderr
        assert name in listed.stdout, f"pod ls did not list the running pod:\n{listed.stdout}"

        status = _pod(cfg_env, "status", name, "--json")
        assert status.returncode == 0, status.stderr
        state = json.loads(status.stdout)
        assert state["status"] == "up", state
        assert state["port"] == port, state

        # Capture the identity teardown will be judged against, WHILE it is
        # still live: a pid alone is not an identity, and after the process is
        # gone its creation time cannot be read.
        pid = win.main_pid(cfg, name)
        assert pid is not None, "a serving pod must have a supervised pid"
        start_token = process_start_time(pid)
        assert start_token, "this host must report a creation time for the pod's gateway"
    finally:
        down = _pod(cfg_env, "down", name)

    assert down.returncode == 0, f"pod down failed:\n{down.stdout}\n{down.stderr}"

    # The task is gone. Judged by /Query's EXIT CODE, never its output, which is
    # localized.
    assert not win.task_exists(cfg, name), (
        f"pod down returned 0 but the scheduled task at {win.task_name(cfg, name)} "
        "is still registered"
    )

    # The gateway is gone, and "gone" means by pid PLUS creation-time identity:
    # a recycled pid answering with its own creation time must not read as the
    # pod still running.
    assert process_start_time(pid) != start_token, (
        f"pod down returned 0 but pid {pid} still carries the pod gateway's "
        "creation-time identity, so the process was never reaped"
    )
    # And the record itself is cleared, so nothing can attest for this name.
    assert win.main_pid(cfg, name) is None
    assert not win.task_script_path(cfg, name).exists()
    assert not cfg.home_dir(name).exists(), "the isolated HOME must be reclaimed"


def _load_cfg(env: dict[str, str]) -> PodConfig:
    """A PodConfig for the plane *env* describes, without mutating this process."""
    saved = {k: os.environ.get(k) for k in env if k.startswith("KIROCREW_POD_")}
    os.environ.update({k: v for k, v in env.items() if k.startswith("KIROCREW_POD_")})
    try:
        return PodConfig.load()
    finally:
        for key, old in saved.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old
