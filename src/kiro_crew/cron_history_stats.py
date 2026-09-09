"""Fold a cron job's run records into the few numbers a cost audit needs.

The ``cron-history`` directory is bind-masked from every sandboxed agent shell
(``sandbox._CREW_HIDDEN_LEAVES``), so a skill script cannot read run history
itself. It has to come from the gateway, and what crosses that boundary should
be COUNTS rather than the run text they were derived from: a count cannot carry
a credential, and folding here keeps the payload small enough that a registry
with many jobs still fits well under the tool-response ceiling.

The counts answer one question -- has this job ever actually done any work? A job
whose every recorded run said the same thing, or said it had nothing to do, is
paying for a full agent context to reach a conclusion a program could reach.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

#: Result text that means the run found nothing to do. Matched against a run's
#: own summary, so it describes what the job SAID, not what its prompt asked for.
NOOP_RE = re.compile(
    r"("
    r"nothing to do|nothing new|nothing to report|nothing changed|"
    r"no new |no change|no changes|unchanged|no update|no action|"
    r"none found|no matches|no results|no failures|no errors|no issues|"
    r"all clear|all good|all healthy|clean run|looks healthy|is healthy|"
    r"up to date|already (done|handled|posted|processed|triaged)|"
    r"skipped|no-op|idle|0 found|0 new|zero new"
    r")",
    re.IGNORECASE,
)


def normalize_summary(text: str) -> str:
    """Collapse a run summary so two runs that said the same thing compare equal.

    Digits become ``#`` because a timestamp, a count or a percentage changing is
    exactly the case that reads as different while meaning the same thing: "tmp
    1%, home 19%" and "tmp 4%, home 22%" are one result, not two.
    """
    lowered = re.sub(r"\d+", "#", text.strip().lower())
    return re.sub(r"\s+", " ", lowered)


def fold_runs(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Reduce run records to counts.

    ``records`` are history rows as the store serializes them, so each carries at
    least ``status`` and ``summary``. A row whose status is present and not
    ``success`` counts as a failure and contributes to no other tally: a run that
    crashed says nothing about whether the job had work to do.

    ``same_every_run`` needs more than one run to mean anything, so a single
    recorded run reports False rather than trivially True.
    """
    runs = 0
    failures = 0
    noop_runs = 0
    seen: set[str] = set()
    for rec in records:
        status = str(rec.get("status") or "")
        if status and status != "success":
            failures += 1
            continue
        runs += 1
        summary = str(rec.get("summary") or "")
        seen.add(normalize_summary(summary))
        if NOOP_RE.search(summary):
            noop_runs += 1
    return {
        "runs": runs,
        "failures": failures,
        "distinct_summaries": len(seen),
        "noop_runs": noop_runs,
        "same_every_run": runs > 1 and len(seen) == 1,
    }
