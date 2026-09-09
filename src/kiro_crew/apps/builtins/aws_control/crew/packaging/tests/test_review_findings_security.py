"""The five findings the GPT lane raised once its adjudication could run.

All five are the same shape of defect and it is worth naming: the builder exists to stop
untrusted crew content reaching a place it should not, and each of these was a path or a
field it trusted on the way. Each test below reddens if its fix is reverted, and each
mutation is pointed at the exact construct rather than at a substring that also appears
elsewhere.

F1 ``_write_marker_exclusive`` -- the staging marker was written with ``write_text``, so a
   symlink pre-planted at ``<out>.staging.owned`` was followed and its target truncated.

F2 ``_validated_crew_name`` -- ``source / "agents" / f"{name}.json"`` let ``--crew`` carry
   separators, ``..`` or an absolute path, so the spec read came from outside the source.
   Operator-supplied rather than attacker-supplied, so hardening rather than a breach.

F3 ``_open_root_nofollow`` -- the anchor root of the per-component ``O_NOFOLLOW`` walk was
   itself opened following links, so swapping ``<source>/agents`` for a link made every
   check below verify the wrong tree carefully.

F4 ``_marker_is_ours`` -- ownership was ``staging_marker.is_file()``, true of any plain
   file, and it authorised ``shutil.rmtree``. The aside-directory path accepted a
   plan-only directory on the FILENAME alone.

F5 ``build_spec`` -- a non-list ``tools`` skipped the isinstance branch and then hit
   ``set(spec.get("tools") or [])``, raising an uncaught ``TypeError``.
"""

from __future__ import annotations

import json
import os
import pathlib

import pytest

from kiro_crew import credential_patterns

from .test_producer import BUILD_PY, load_build, make_crew, sign_plan

_posix_only = pytest.mark.skipif(
    os.name != "posix",
    reason="the crew bundle builder is POSIX-only; guarded off on platforms without an "
    "atomic no-follow primitive (Windows). See the POSIX-only entry guard.",
)


def _crew(mod, home: pathlib.Path, name: str = "frontdesk"):
    return mod.resolve_crew(name, home)


def _build(mod, home: pathlib.Path, work: pathlib.Path, select=None):
    crew = mod.resolve_crew("frontdesk", home)
    spec = mod.read_agent_spec(crew)
    cands = mod.enumerate_all(crew, spec)
    work.mkdir(parents=True, exist_ok=True)
    plan_path = sign_plan(mod, crew, spec, work, select=select or {})
    plan = mod.merge_plans([plan_path], "frontdesk")
    mod.verify(plan, "frontdesk", cands)
    return mod.build_bundle(crew, spec, cands, plan, work / "bundle")


# ---------------------------------------------------------------------------
# F1: the marker write must not follow a planted link
# ---------------------------------------------------------------------------
@pytest.mark.skipif(os.name != "posix", reason="needs symlink semantics the fix relies on")
def test_a_planted_marker_symlink_is_refused_and_the_target_survives(
    tmp_path: pathlib.Path,
) -> None:
    """A link at the marker path stops the build, and the victim keeps its bytes.

    The refusal is the part that changed. An earlier version of the fix quietly wrote
    somewhere else and let the build finish, which leaves the operator with a green build
    and an attacker-chosen path in their directory. A link at a path derived from ``--out``
    is a signal, not an obstacle to route around.

    Both halves matter: the surviving bytes are the security property, and the refusal is
    what makes the situation visible to whoever ran the build.
    """
    mod = load_build()
    home = make_crew(tmp_path / "home")
    work = tmp_path / "work"
    work.mkdir()
    victim = tmp_path / "precious.txt"
    victim.write_bytes(b"do not truncate me\n")
    (work / "bundle.staging.owned").symlink_to(victim)

    with pytest.raises(mod.ExportRefused) as caught:
        _build(mod, home, work)
    assert "symlink" in str(caught.value).lower(), str(caught.value)
    assert victim.read_bytes() == b"do not truncate me\n", "the planted link was followed"


@pytest.mark.skipif(os.name != "posix", reason="needs symlink semantics the fix relies on")
def test_MUTATION_a_write_text_marker_truncates_the_link_target(tmp_path: pathlib.Path) -> None:
    """Restore ``write_text`` for the marker and the victim is destroyed.

    This is the defect reproduced. It pins that the fd-based write is what protects the
    target, not something else in the surrounding checks.
    """
    anchor = "    _write_nofollow(path, _STAGING_MARKER_BODY, exclusive=not ours)"
    assert (
        BUILD_PY.read_text(encoding="utf-8").count(anchor) == 1
    ), "the mutation anchor moved or is not unique; re-point it at the marker write"
    mod = load_build(
        mutate=(anchor, '    path.write_text(_STAGING_MARKER_BODY, encoding="utf-8", newline="")')
    )
    home = make_crew(tmp_path / "home")
    work = tmp_path / "work"
    work.mkdir()
    victim = tmp_path / "precious.txt"
    victim.write_bytes(b"do not truncate me\n")
    (work / "bundle.staging.owned").symlink_to(victim)

    _build(mod, home, work)

    assert (
        victim.read_bytes() != b"do not truncate me\n"
    ), "the mutation did not reach the marker write, so this test proves nothing"


@_posix_only
def test_a_successful_build_still_leaves_no_marker(tmp_path: pathlib.Path) -> None:
    """The exclusive write must not break the cleanup the old write had.

    A marker left behind is a licence for the NEXT run to delete whatever is at that path,
    so this property is why the marker exists at all.
    """
    mod = load_build()
    home = make_crew(tmp_path / "home")
    work = tmp_path / "work"
    _build(mod, home, work)
    assert not (work / "bundle.staging.owned").exists()


# ---------------------------------------------------------------------------
# F2: a crew name is a name
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name",
    ["../../etc/passwd", "..", "a/b", "a\\b", "/absolute", "", "sub/../../out"],
)
def test_a_crew_name_that_can_address_a_path_is_refused(name, tmp_path: pathlib.Path) -> None:
    """Every rejected shape, so a partial fix cannot pass.

    ``..`` and ``a/b`` are the two the join actually resolved: ``Path.__truediv__`` treats
    an absolute segment as a new root and ``..`` as a parent step, so the read left the
    source the operator named.
    """
    mod = load_build()
    with pytest.raises(mod.ExportRefused):
        mod.resolve_crew(name, tmp_path)


def test_an_ordinary_crew_name_still_resolves(tmp_path: pathlib.Path) -> None:
    """Non-vacuity: the check must not have become a blanket refusal.

    Names with dots, dashes and unicode are legal filenames and legal crew names; only the
    path-addressing shapes are refused.
    """
    for name in ["frontdesk", "front.desk", "front-desk_2", "cafe-brulee"]:
        crew = mod_resolve(tmp_path, name)
        assert crew.agent_spec_path.name == f"{name}.json"
        assert crew.agent_spec_path.parent.name == "agents"


def mod_resolve(root: pathlib.Path, name: str):
    return load_build().resolve_crew(name, root)


# ---------------------------------------------------------------------------
# F3: the anchor root itself must not be a link
# ---------------------------------------------------------------------------
@_posix_only
def test_a_prompt_inside_a_real_agents_directory_still_inlines(
    tmp_path: pathlib.Path,
) -> None:
    """The end-to-end path the root check sits on must still work."""
    mod = load_build()
    src = make_crew(tmp_path / "home", prompt="file://persona.md")
    (src / "agents" / "persona.md").write_text("the real persona\n", encoding="utf-8")
    crew = mod.resolve_crew("frontdesk", src)
    spec = mod.read_agent_spec(crew)
    result = mod.build_spec(crew, spec, set(), crew.agent_spec_path.parent)
    assert "the real persona" in result.spec["prompt"]


# ---------------------------------------------------------------------------
# F4: ownership must be more than "a file is here"
# ---------------------------------------------------------------------------
def test_a_foreign_file_at_the_marker_path_does_not_authorise_deletion(
    tmp_path: pathlib.Path,
) -> None:
    """An operator's own note must not license a recursive delete of their own directory.

    This is the forged-token case the old ``is_file()`` accepted. The refusal is what keeps
    ``their_work.txt`` on disk.
    """
    mod = load_build()
    home = make_crew(tmp_path / "home")
    work = tmp_path / "work"
    work.mkdir()
    theirs = work / "bundle.staging"
    (theirs / "skills").mkdir(parents=True)
    (theirs / "skills" / "their_work.txt").write_text("hours of it\n", encoding="utf-8")
    (work / "bundle.staging.owned").write_text("a note of mine\n", encoding="utf-8")

    with pytest.raises(mod.ExportRefused) as caught:
        _build(mod, home, work)
    msg = str(caught.value)
    if os.name == "posix":
        assert "did not create it" in msg
    else:
        assert "POSIX-only" in msg
    assert (theirs / "skills" / "their_work.txt").is_file(), "their file was deleted"


def test_a_plan_only_directory_must_carry_a_plan_this_tool_wrote(tmp_path: pathlib.Path) -> None:
    """The name ``curation-plan.json`` is not proof of origin.

    A plan-only directory is the normal state between the two verbs, so it has to be
    accepted -- which is why the check is on the plan's own ``plan_version`` rather than a
    blanket refusal of the shape.
    """
    mod = load_build()
    home = make_crew(tmp_path / "home")
    work = tmp_path / "work"
    out = work / "bundle"
    out.mkdir(parents=True)
    (out / mod.PLAN_FILENAME).write_text("not our plan at all\n", encoding="utf-8")

    with pytest.raises(mod.ExportRefused) as caught:
        _build(mod, home, work)
    msg = str(caught.value)
    if os.name == "posix":
        assert "did not write" in msg
    else:
        assert "POSIX-only" in msg


def test_the_marker_this_build_writes_is_recognised_as_its_own(tmp_path: pathlib.Path) -> None:
    """Non-vacuity for the token: the writer and the reader must agree.

    If they disagreed, every resume would refuse and the crash-cleanup path this marker
    exists for would be dead -- passing tests, dead feature.
    """
    mod = load_build()
    marker = tmp_path / "bundle.staging.owned"
    mod._write_marker_exclusive(marker)
    assert mod._marker_is_ours(marker)
    assert marker.read_text(encoding="utf-8").startswith(mod._STAGING_MARKER_TOKEN)


# ---------------------------------------------------------------------------
# F5: a malformed tools field gets a reason, not a traceback
# ---------------------------------------------------------------------------
@_posix_only
@pytest.mark.parametrize("field", ["tools", "allowedTools"])
@pytest.mark.parametrize("value", [3, "fs_read", {"a": 1}, True])
def test_a_non_list_tool_field_is_refused_not_crashed(field, value, tmp_path: pathlib.Path) -> None:
    """``ExportRefused`` naming the field, rather than ``TypeError`` from a set().

    Both fields and several shapes, because the old guard was ``isinstance(list)`` on one
    of them: a truthy non-iterable skipped that branch and reached the set() below it.
    """
    mod = load_build()
    home = make_crew(tmp_path / "home")
    spec_path = home / "agents" / "frontdesk.json"
    body = json.loads(spec_path.read_text(encoding="utf-8"))
    body[field] = value
    spec_path.write_text(json.dumps(body), encoding="utf-8")

    crew = mod.resolve_crew("frontdesk", home)
    spec = mod.read_agent_spec(crew)
    with pytest.raises(mod.ExportRefused) as caught:
        mod.build_spec(crew, spec, set(), crew.agent_spec_path.parent)
    assert field in str(caught.value)


@_posix_only
def test_a_list_tool_field_still_works(tmp_path: pathlib.Path) -> None:
    """Non-vacuity: the shape check must not refuse the normal spec."""
    mod = load_build()
    home = make_crew(tmp_path / "home", tools=["fs_read"], allowed_tools=["fs_read"])
    crew = mod.resolve_crew("frontdesk", home)
    spec = mod.read_agent_spec(crew)
    result = mod.build_spec(crew, spec, set(), crew.agent_spec_path.parent)
    assert result.spec["tools"] == ["fs_read"]


# ---------------------------------------------------------------------------
# Round-11 GPT F1: a redirected skills ROOT must be refused, not traversed
#
# The per-entry guard already blocks a redirected SKILL.md and the chain guard
# blocks redirected out/staging/previous paths, but the skills root itself was an
# uncovered variant: a symlinked ``<source>/skills`` makes ``rglob`` enumerate a
# tree outside ``--source`` while every id still reads in-bounds, so files sourced
# elsewhere ship in the bundle.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(os.name != "posix", reason="needs symlink semantics the fix relies on")
def test_a_symlinked_skills_root_is_refused(tmp_path: pathlib.Path) -> None:
    """A skills root that redirects outside --source is refused before enumeration.

    The refusal is what keeps ``outside/secret_skill/SKILL.md`` -- a file the operator
    never placed under the crew source -- out of the candidate list and the bundle.
    """
    mod = load_build()
    # A tree OUTSIDE the crew source, carrying a skill that must never be enumerable.
    outside = tmp_path / "outside"
    (outside / "secret_skill").mkdir(parents=True)
    (outside / "secret_skill" / "SKILL.md").write_text("# not from this crew\n", encoding="utf-8")

    home = tmp_path / "home"
    home.mkdir()
    (home / "agents").mkdir()
    skills_root = home / "skills"
    skills_root.symlink_to(outside, target_is_directory=True)

    with pytest.raises(mod.ExportRefused) as caught:
        mod.skill_candidates(skills_root)
    assert "link or junction" in str(caught.value)


@pytest.mark.skipif(os.name != "posix", reason="needs symlink semantics the fix relies on")
def test_MUTATION_a_symlinked_skills_root_would_leak_without_the_guard(
    tmp_path: pathlib.Path,
) -> None:
    """Revert the root-redirect guard and the out-of-source skill becomes enumerable.

    Reddens the fix: with the guard stripped, ``skill_candidates`` follows the link,
    ``rglob`` finds ``secret_skill/SKILL.md`` in the redirected tree, and it appears as a
    selectable candidate whose bytes live outside ``--source``.
    """
    mod = load_build(mutate=("if _is_redirecting_entry(skills_root):", "if False:"))
    outside = tmp_path / "outside"
    (outside / "secret_skill").mkdir(parents=True)
    (outside / "secret_skill" / "SKILL.md").write_text("# not from this crew\n", encoding="utf-8")

    home = tmp_path / "home"
    home.mkdir()
    skills_root = home / "skills"
    skills_root.symlink_to(outside, target_is_directory=True)

    cands = mod.skill_candidates(skills_root)
    assert any(c.id == "secret_skill" for c in cands), (
        "guard stripped: the out-of-source skill should leak into the candidate list, "
        "proving the guard is what blocks it"
    )


# ---------------------------------------------------------------------------
# Round-22 GPT: a non-string ELEMENT of tools/allowedTools must be refused, not
# str()-coerced (tools) or silently dropped (allowedTools). Coercion fabricates a
# capability grant in a SIGNED bundle; a dropped grant is a silent capability
# change. Both invent/alter information -- invalid input gets ExportRefused.
# ---------------------------------------------------------------------------
@_posix_only
@pytest.mark.parametrize("field", ["tools", "allowedTools"])
@pytest.mark.parametrize("bad", [{"name": "fs_read"}, 7, ["nested"], True])
def test_a_non_string_tool_entry_is_refused_not_coerced(field, bad, tmp_path: pathlib.Path) -> None:
    mod = load_build()
    home = make_crew(tmp_path / "home")
    spec_path = home / "agents" / "frontdesk.json"
    body = json.loads(spec_path.read_text(encoding="utf-8"))
    body[field] = ["fs_read", bad]
    spec_path.write_text(json.dumps(body), encoding="utf-8")

    crew = mod.resolve_crew("frontdesk", home)
    spec = mod.read_agent_spec(crew)
    with pytest.raises(mod.ExportRefused) as caught:
        mod.build_spec(crew, spec, set(), crew.agent_spec_path.parent)
    msg = str(caught.value)
    assert field in msg
    assert type(bad).__name__ in msg, "the refusal names the entry's actual type"


@_posix_only
def test_MUTATION_str_coercing_a_tool_entry_would_fabricate_a_grant(tmp_path: pathlib.Path) -> None:
    """Revert to str()-coercion and a dict tool entry becomes a fabricated tool id, not refused."""
    mod = load_build(
        mutate=(
            '        for e in tools:\n            if not isinstance(e, str):\n                raise ExportRefused(\n                    f"\'tools\' contains a {type(e).__name__} entry ({e!r}), not a string. A "\n                    f"tool grant is computed from it and would be fabricated by coercion; the "\n                    f"bundle is signed, so an invented capability cannot be allowed. Fix the "\n                    f"spec."\n                )\n        kept = [e for e in tools if not _is_orphan(e)]\n        orphans = [e for e in tools if _is_orphan(e)]',
            "        kept = [str(e) for e in tools if not _is_orphan(str(e))]\n        orphans = [str(e) for e in tools if _is_orphan(str(e))]",
        )
    )
    home = make_crew(tmp_path / "home")
    spec_path = home / "agents" / "frontdesk.json"
    body = json.loads(spec_path.read_text(encoding="utf-8"))
    body["tools"] = [{"name": "fs_read"}]
    spec_path.write_text(json.dumps(body), encoding="utf-8")

    crew = mod.resolve_crew("frontdesk", home)
    spec = mod.read_agent_spec(crew)
    # With the guard mutated off, the dict is str()-coerced into a fabricated tool id and the
    # build does NOT refuse it -- proving the type check is what prevents the invented grant.
    result = mod.build_spec(crew, spec, set(), crew.agent_spec_path.parent)
    assert result.spec["tools"] == ["{'name': 'fs_read'}"], (
        "mutated: a non-string tool entry is coerced into a fabricated tool id in the signed "
        "spec, which the real type-refusal prevents"
    )


#: The shared vendor/token spellings, imported from the single home both the scrubber
#: and this standalone subset read. The test below asks the CANONICAL scrubber for the
#: shortest token it accepts per prefix, then requires the standalone side to accept that
#: same token -- so a standalone bound HIGHER than canonical (a GitLab body of 16..19, an
#: npm body of 24..35 the scrubber redacts and a tighter subset ships) fails the test. A
#: fixed-string sample would not catch that: it passes at any bound at or below its length.
_B62 = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def _prefix_of(fragment: str) -> str:
    """The literal lead of a fragment (``glpat-``, ``sk-proj-``, ...), up to its class."""
    return fragment[: fragment.index("[")]


def _body(length: int) -> str:
    """``length`` base62 characters -- valid for every class these fragments use."""
    return (_B62 * ((length // len(_B62)) + 1))[:length]


def _canonical_min_body_length(prefix: str, canonical, ceiling: int = 80) -> int | None:
    """Shortest base62 body length canonical accepts after ``prefix``, or None if never.

    Probes upward, so it measures canonical's OWN floor rather than trusting a restated
    one. The standalone side is then required to catch a token at exactly this length.
    """
    for n in range(1, ceiling + 1):
        token = prefix + _body(n)
        if any(rx.search(token) for rx in canonical):
            return n
    return None


@pytest.mark.parametrize("label,fragment", credential_patterns.VENDOR_TOKEN_PATTERNS)
def test_the_standalone_scan_catches_each_vendor_token_at_canonical_minimum(
    label: str, fragment: str
) -> None:
    """Per-format, at CANONICAL's minimum, so a standalone bound above canonical is caught.

    The standalone fallback is the REAL scan path in the deployment venv where the scrubber
    is not importable, so a token the scrubber redacts but the subset misses ships unscanned.
    The length is measured FROM canonical (probed, not restated), and the standalone side
    must catch a token at that length: a subset bound tighter than canonical reddens here,
    naming the exact format. This is the smaller-order form of the missed-format finding.
    """
    from kiro_crew.security import get_credential_patterns

    mod = load_build()
    canonical = get_credential_patterns()
    prefix = _prefix_of(fragment)
    min_len = _canonical_min_body_length(prefix, canonical)
    if min_len is None:
        pytest.skip(f"canonical scrubber does not carry a {label!r}-prefixed pattern")
    token = prefix + _body(min_len)
    assert any(rx.search(token) for _, rx in mod._HARD_PATTERNS), (
        f"the standalone scan misses a {label} token at canonical's own minimum length "
        f"({len(token)} chars); its bound has drifted ABOVE canonical and would ship a "
        "token the scrubber redacts"
    )


def test_an_ordinary_dotted_identifier_is_not_a_false_vendor_token() -> None:
    """Non-vacuity: the vendor patterns must not paint every hyphenated word a secret."""
    mod = load_build()
    for benign in ("just-a-normal-identifier", "sk-short", "npm-run-build"):
        assert not any(
            rx.search(benign) for _, rx in mod._HARD_PATTERNS
        ), f"{benign!r} is not a credential but the standalone scan flagged it"


# ---------------------------------------------------------------------------
# The ``already_resolved=True`` pinned open at the _inline_prompt anchor site
# (build.py:3013) refuses a component swapped for a symlink between the caller's
# resolve and this open.
#
# ``already_resolved=True`` skips only the re-resolution -- it does NOT skip the
# per-component ``O_NOFOLLOW`` walk, which opens EVERY component of the passed
# value descriptor-relative with ``O_RDONLY|O_DIRECTORY|O_NOFOLLOW``. A component
# that becomes a symlink after the value was computed fails its OWN open, and no
# path string is re-resolved once the walk starts. The two tests below prove the
# refusal and, by mutation, that ``O_NOFOLLOW`` is what enforces it.
# ---------------------------------------------------------------------------
@_posix_only
def test_already_resolved_pinned_open_refuses_a_post_resolve_component_swap(
    tmp_path: pathlib.Path,
) -> None:
    """A parent swapped for a symlink AFTER the resolve is refused, not followed.

    Reproduces the finding's exact scenario: a caller resolves ``<base>/mid/leaf`` while
    every component is a real directory, then ``mid`` is replaced with a symlink to an
    attacker directory before the anchor open runs. ``_open_dir_nofollow_pinned`` is called
    with ``already_resolved=True`` -- the flag the finding names -- and must refuse.
    """
    mod = load_build()
    base = tmp_path / "base"
    (base / "mid" / "leaf").mkdir(parents=True)
    victim = tmp_path / "victim"
    (victim / "leaf").mkdir(parents=True)

    # The value a caller resolved BEFORE the swap, all real directories at that instant.
    resolved = (base / "mid" / "leaf").resolve()

    # The swap the finding describes: an intermediate component becomes a link out of tree.
    (base / "mid" / "leaf").rmdir()
    (base / "mid").rmdir()
    (base / "mid").symlink_to(victim, target_is_directory=True)

    # Confirm the swap DID redirect the name, so a naive open-by-string would land in victim.
    assert (base / "mid" / "leaf").resolve() == (victim / "leaf").resolve()

    with pytest.raises(OSError):
        fd = mod._open_dir_nofollow_pinned(resolved, already_resolved=True)
        os.close(fd)  # unreachable if the refusal holds; closes the leak if it does not


@_posix_only
def test_a_clean_resolved_anchor_still_opens(tmp_path: pathlib.Path) -> None:
    """Non-vacuity: an untouched resolved anchor opens, so the refusal above is the swap."""
    mod = load_build()
    anchor = tmp_path / "base" / "mid" / "leaf"
    anchor.mkdir(parents=True)
    fd = mod._open_dir_nofollow_pinned(anchor.resolve(), already_resolved=True)
    try:
        assert os.fstat(fd).st_ino == os.stat(anchor).st_ino
    finally:
        os.close(fd)


@_posix_only
def test_MUTATION_dropping_O_NOFOLLOW_would_follow_the_swapped_parent(
    tmp_path: pathlib.Path,
) -> None:
    """Strip ``O_NOFOLLOW`` from the pinned walk and the swapped parent is followed.

    Reddens the guard: with the flag gone, the open of ``mid`` follows the link into
    ``victim`` and the walk reaches ``victim/leaf`` and returns a descriptor -- exactly the
    hole the per-component ``O_NOFOLLOW`` closes. The mutation anchor pins the two-line block
    inside ``_open_dir_nofollow_pinned`` (the ``resolved =`` line is unique to that function),
    so it cannot land on the identically-worded ``dir_flags`` line elsewhere in the module.
    """
    mod = load_build(
        mutate=(
            "    resolved = dir_path if already_resolved else dir_path.resolve()\n"
            '    dir_flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)',
            "    resolved = dir_path if already_resolved else dir_path.resolve()\n"
            "    dir_flags = os.O_RDONLY | os.O_DIRECTORY",
        )
    )
    base = tmp_path / "base"
    (base / "mid" / "leaf").mkdir(parents=True)
    victim = tmp_path / "victim"
    (victim / "leaf").mkdir(parents=True)
    resolved = (base / "mid" / "leaf").resolve()
    (base / "mid" / "leaf").rmdir()
    (base / "mid").rmdir()
    (base / "mid").symlink_to(victim, target_is_directory=True)

    fd = mod._open_dir_nofollow_pinned(resolved, already_resolved=True)
    try:
        # The descriptor is victim/leaf, reached by following the swapped link -- the leak
        # the real O_NOFOLLOW walk refuses.
        assert os.fstat(fd).st_ino == os.stat(victim / "leaf").st_ino, (
            "mutated: without O_NOFOLLOW the walk should follow the swapped parent into the "
            "attacker directory, proving the flag is what blocks the swap"
        )
    finally:
        os.close(fd)


@_posix_only
def test_a_plan_only_directory_for_another_crew_is_not_owned(tmp_path: pathlib.Path) -> None:
    """A version field is not an ownership claim; the crew the plan names has to match.

    The plan-only directory is the state between the two verbs, and it is deleted
    recursively if it is treated as this build's own staging tree. A curation-plan.json
    that carries the right ``plan_version`` but names a different crew is a foreign file, so
    it must not license that delete, and the foreign file survives the refusal.
    """
    mod = load_build()
    home = make_crew(tmp_path / "home")
    work = tmp_path / "work"
    out = work / "bundle"
    out.mkdir(parents=True)
    (out / mod.PLAN_FILENAME).write_text(
        json.dumps({"plan_version": mod.PLAN_VERSION, "crew": "someone-elses-crew"}),
        encoding="utf-8",
    )
    before = (out / mod.PLAN_FILENAME).read_text(encoding="utf-8")

    with pytest.raises(mod.ExportRefused) as caught:
        _build(mod, home, work)
    assert "did not write" in str(caught.value)
    assert (out / mod.PLAN_FILENAME).is_file(), "the foreign plan-only directory was deleted"
    assert (out / mod.PLAN_FILENAME).read_text(encoding="utf-8") == before


@_posix_only
def test_bundle_digest_refuses_a_staged_leaf_swapped_for_a_symlink(
    tmp_path: pathlib.Path,
) -> None:
    """A staged leaf swapped for a symlink is refused at hashing, not hashed through.

    ``bundle_digest`` signs the manifest and is re-derived to prove ownership before a
    recursive delete, so a leaf that becomes a symlink between the file-shape check and the
    read must not fold the target's bytes into the digest. The read is held through one
    no-follow descriptor, so the redirect fails the open and the digest refuses rather than
    pinning bytes from wherever the link points. A tree of only regular files still hashes.
    """
    mod = load_build()
    root = tmp_path / "bundle"
    (root / "skills").mkdir(parents=True)
    (root / "agent.json").write_text('{"name": "frontdesk"}\n', encoding="utf-8")
    leaf = root / "skills" / "SKILL.md"
    leaf.write_text("# real\n", encoding="utf-8")

    good = mod.bundle_digest(root)
    assert good.startswith("sha256:"), "an all-regular-file tree must still hash"

    outside = tmp_path / "outside.txt"
    outside.write_text("ATTACKER BYTES\n", encoding="utf-8")
    leaf.unlink()
    leaf.symlink_to(outside)

    with pytest.raises(mod.ExportRefused) as caught:
        mod.bundle_digest(root)
    msg = str(caught.value)
    assert "no-follow descriptor" in msg or "following a redirect" in msg
