"""``agent_template_pane`` must survive load() -> masked GET -> frontend predicate.

The crew definition panel (private fork-on-edit) is gated behind
``agent_template_pane: true`` as a TOP-LEVEL key in the running instance's
``config.json``. The frontend reads that flag live off ``GET
/api/config/kirocrew`` (``website/src/hooks/useAgentTemplatePane.ts`` requires
the value to be exactly ``true``), so the whole feature turns on the key
reaching the browser in that response body — the same launch-blocker shape
``connections_ui`` had (see test_connections_ui_flag.py for the full rationale
on why an unmodelled key never gets there).
"""

from __future__ import annotations

import json

from kiro_crew.config import loader as L
from kiro_crew.config.loader import KiroCrewConfig

# The one spelling the frontend hook and the feature map use.
FLAG = "agent_template_pane"


def _point_loader_at(tmp_path, monkeypatch, data: dict) -> None:
    cfgp = tmp_path / "config.json"
    cfgp.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(L, "config_path", lambda: cfgp)
    monkeypatch.setattr(L, "config_dir", lambda: tmp_path)
    monkeypatch.setattr(L, "config_local_path", lambda: tmp_path / "config.local.json")


def _masked(cfg: KiroCrewConfig) -> dict:
    from kiro_crew.dashboard.handlers.core import _masked_config_dict

    return _masked_config_dict(cfg)


def test_flag_set_true_reaches_the_masked_get(tmp_path, monkeypatch):
    """The launch blocker: ``true`` on disk must arrive in the browser's copy."""
    _point_loader_at(tmp_path, monkeypatch, {FLAG: True})
    cfg = KiroCrewConfig.load()

    assert cfg.agent_template_pane is True
    # Modelled, therefore NOT swept up as an unknown section...
    assert FLAG not in cfg._extra_sections
    # ...and therefore present in the browser-facing view, strict-true.
    assert _masked(cfg).get(FLAG) is True


def test_flag_defaults_off_and_rejects_truthy_strings(tmp_path, monkeypatch):
    """Default-off; a string "true" must NOT coerce (frontend is === true)."""
    _point_loader_at(tmp_path, monkeypatch, {})
    assert KiroCrewConfig.load().agent_template_pane is False

    _point_loader_at(tmp_path, monkeypatch, {FLAG: "true"})
    cfg = KiroCrewConfig.load()
    assert cfg.agent_template_pane is False
    assert _masked(cfg).get(FLAG) is False


def test_flag_round_trips_through_save(tmp_path, monkeypatch):
    """save() must serialize the field so an operator's opt-in survives."""
    _point_loader_at(tmp_path, monkeypatch, {FLAG: True})
    cfg = KiroCrewConfig.load()
    cfg.save()
    on_disk = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert on_disk[FLAG] is True
