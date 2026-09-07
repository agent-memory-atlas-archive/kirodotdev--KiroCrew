"""Measure the episodic admission gate as the classifier its threshold claims to be.

``_EPISODIC_RELEVANCE_THRESHOLD`` decides whether a retrieved fragment is allowed
into the prompt at all. That makes it a binary classifier over (query, fragment)
pairs, and the only honest way to describe one is with both error rates and the two
score distributions it has to separate. This module produces those, over the real
embedder and the real ``VectorMemoryStore``.

**Why the existing rulers do not answer this.** :mod:`.retrieval` measures recall
at k — did the gold evidence rank inside the window — which is a question about
ORDERING. Admission is a question about a CUT: a threshold can be far too low and
leave recall at k untouched, because the rows it wrongly admits arrive below the
ones it correctly admits. Recall at k is blind to exactly the failure a threshold
has, so a separate instrument is required.

**What is reported, and why each part is load-bearing.**

* Precision, recall and F1 at the shipped gate — the headline, and on its own
  worthless. An F1 computed against one distractor per query is a statement about
  the distractor sample as much as about the threshold.
* The full cosine DISTRIBUTION of each class (minimum, median, p90, p99, maximum).
  This is the finding a single F1 hides. If the irrelevant class's upper tail
  reaches above the gate while the relevant class's lower tail reaches below it,
  the two distributions OVERLAP and no threshold value separates them — the gate is
  then choosing a trade-off, not applying a discovered boundary, and a reader is
  entitled to know which of those they are looking at.
* A sweep of every threshold on a fixed grid, so the best achievable F1 is
  measured rather than asserted, and so the shape near the shipped value is
  visible. A threshold sitting on a plateau and one sitting on a cliff have very
  different consequences for a corpus slightly unlike this one.
* Both length branches separately. The gate relaxes to
  ``_EPISODIC_LONG_TEXT_THRESHOLD`` above ``_EPISODIC_LONG_TEXT_CHARS`` on the
  stated rationale that long texts dilute cosine. The corpus states every fact at
  both lengths, so that rationale is measurable rather than assumed.

**Two class balances, both reported.** The BALANCED view gives each query exactly
one distractor, which is the protocol shape a "50 relevant + 50 irrelevant"
benchmark describes and is what makes an older claim comparable. The FULL view
scores every query against every fragment, which is what ``search_episodic``
actually does — it scores the query against every embedded row in the store — and
is the only view with enough irrelevant samples for an upper-tail percentile to
mean anything. Precision differs sharply between the two, and that difference is
information, not an inconsistency.

**Not a runtime path and not part of the default suite.** Loading the embedding
model costs ~700MB of resident memory and several seconds, so
``test_episodic_admission_bench.py`` runs only when
``KIROCREW_BENCH_ADMISSION`` is set and skips with a reason when no model is
resident. Nothing here fabricates a vector: without a real embedder it refuses,
because a hashed bag-of-words stand-in would turn a semantic threshold measurement
into a term-overlap measurement while still printing a plausible F1.

Run it directly::

    KIROCREW_EMBED_MODEL_PATH=/path/to/qwen3-embedding-0.6b.gguf \\
        python -m kiro_crew.eval.bench.admission

or as the opt-in test::

    KIROCREW_BENCH_ADMISSION=1 KIROCREW_EMBED_MODEL_PATH=/path/to/model.gguf \\
        pytest test/test_episodic_admission_bench.py -q -n0 -s
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from kiro_crew import memory_v2
from kiro_crew.vector_memory import (
    _EPISODIC_LONG_TEXT_CHARS,
    _EPISODIC_LONG_TEXT_THRESHOLD,
    _EPISODIC_RELEVANCE_THRESHOLD,
    VectorMemoryStore,
)

from .admission_corpus import ADMISSION_TOPICS, AdmissionTopic
from .errors import BenchRefusal
from .ingest import EmbedFn, search_backend

logger = logging.getLogger(__name__)

#: Fragment lengths the corpus carries, in the order they are reported. These are
#: the two branches of the gate, named rather than derived from a boolean, so a
#: report says which branch a number belongs to.
SHORT = "short"
LONG = "long"
KINDS = (SHORT, LONG)

#: The sweep grid. Fixed and inclusive of both ends so two runs are comparable
#: point by point, and fine enough (0.01) that the plateau around any optimum is
#: visible rather than being a single sampled point.
SWEEP_MIN = 0.20
SWEEP_MAX = 0.90
SWEEP_STEP = 0.01

#: How the balanced view picks each query's single distractor: topic *i* is paired
#: with topic ``(i + stride) % n``. A fixed stride coprime with the topic count
#: (50) makes the pairing a single cycle, so every fragment serves as a distractor
#: exactly once and no topic draws itself. Deterministic by construction — there is
#: no seed to record and no shuffle to reproduce, which is stronger than a seeded
#: one because it cannot silently change when a topic is inserted in the middle.
DISTRACTOR_STRIDE = 17

#: Percentiles reported per class. p99 is included because the irrelevant class's
#: upper tail relative to the gate is the whole question; a median tells you
#: nothing about whether anything crosses.
PERCENTILES = (0.50, 0.90, 0.99)


class AdmissionRefusal(BenchRefusal):
    """Raised rather than reporting a number computed on a compromised sample."""


def validate_corpus(topics: Sequence[AdmissionTopic] = ADMISSION_TOPICS) -> None:
    """Refuse a corpus whose fragments ``write_episodic`` would silently drop.

    Every constraint here is one the store enforces by returning ``False`` from
    the write. A dropped fragment removes a row from the haystack without removing
    it from the corpus the denominator is computed over, so the reported precision
    would describe a store that never held what the report says it did. Checked up
    front so the failure names the offending topic instead of surfacing as a
    missing pair much later.
    """
    from kiro_crew.vector_memory_constants import _contains_injection

    if len(topics) < 2:
        raise AdmissionRefusal(
            f"the corpus carries {len(topics)} topic(s); with fewer than two there is "
            "no irrelevant pair to score against."
        )
    seen_ids: set[str] = set()
    seen_prefixes: dict[str, str] = {}
    for topic in topics:
        if topic.topic_id in seen_ids:
            raise AdmissionRefusal(f"duplicate topic_id in the corpus: {topic.topic_id!r}")
        seen_ids.add(topic.topic_id)
        for kind in KINDS:
            text = getattr(topic, kind)
            label = f"{topic.topic_id}.{kind}"
            # The store's own bounds. Outside them write_episodic returns False.
            if not 10 <= len(text) <= 2000:
                raise AdmissionRefusal(
                    f"{label} is {len(text)} chars; write_episodic accepts 10-2000."
                )
            is_long = len(text) > _EPISODIC_LONG_TEXT_CHARS
            if is_long != (kind == LONG):
                raise AdmissionRefusal(
                    f"{label} is {len(text)} chars, which puts it on the wrong side of the "
                    f"{_EPISODIC_LONG_TEXT_CHARS}-char long-text cutoff for a {kind!r} "
                    "fragment — the two length branches would not be measured separately."
                )
            if _contains_injection(text):
                raise AdmissionRefusal(
                    f"{label} matches an injection pattern, so write_episodic would drop it."
                )
            # Text-hash dedup: the store rejects a second row whose lowercased
            # first 80 characters match an existing one.
            prefix = text[:80].lower()
            if prefix in seen_prefixes:
                raise AdmissionRefusal(
                    f"{label} shares its first 80 characters with {seen_prefixes[prefix]}; "
                    "the store's text-hash dedup would reject the second one."
                )
            seen_prefixes[prefix] = label


@dataclass(frozen=True)
class Pair:
    """One (query, fragment) pair, its cosine, and what the gate did with it."""

    query_topic: str
    doc_topic: str
    kind: str
    cosine: float
    #: The gate value that applied to THIS pair, chosen by the fragment's length.
    gate: float
    relevant: bool
    #: Whether ``search_episodic(relevance_filter=True)`` kept this row. Read from
    #: the store rather than recomputed, so the measurement cannot drift from the
    #: product's own comparison (which reads a cosine rounded to 4 places).
    admitted: bool


@dataclass(frozen=True)
class Distribution:
    """Order statistics of one class's cosines.

    Percentiles are nearest-rank over the observed values, not interpolated, so
    every number printed is a cosine some pair actually scored.
    """

    n: int
    minimum: float
    median: float
    p90: float
    p99: float
    maximum: float
    mean: float

    def describe(self) -> dict[str, object]:
        return {
            "n": self.n,
            "min": round(self.minimum, 4),
            "p50": round(self.median, 4),
            "p90": round(self.p90, 4),
            "p99": round(self.p99, 4),
            "max": round(self.maximum, 4),
            "mean": round(self.mean, 4),
        }


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile: the smallest observed value at or above rank q.

    No interpolation on purpose. An interpolated p99 of an irrelevant-cosine tail
    is a value nothing scored, and the question this report answers — does the
    tail cross the gate — must be answered with observations.
    """
    if not sorted_values:
        raise AdmissionRefusal("cannot take a percentile of an empty class")
    rank = math.ceil(q * len(sorted_values))
    idx = min(len(sorted_values) - 1, max(0, rank - 1))
    return sorted_values[idx]


def distribution(values: Sequence[float]) -> Distribution:
    """Order statistics over *values*."""
    if not values:
        raise AdmissionRefusal("cannot describe an empty class")
    ordered = sorted(values)
    return Distribution(
        n=len(ordered),
        minimum=ordered[0],
        median=_percentile(ordered, 0.50),
        p90=_percentile(ordered, 0.90),
        p99=_percentile(ordered, 0.99),
        maximum=ordered[-1],
        mean=sum(ordered) / len(ordered),
    )


@dataclass(frozen=True)
class Confusion:
    """Counts and the three rates derived from them.

    ``f1`` returns 0.0 when precision and recall are both 0 rather than raising:
    a threshold above every relevant cosine admits nothing, which is a legitimate
    point on the sweep and has to plot.
    """

    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def precision(self) -> float:
        admitted = self.tp + self.fp
        return self.tp / admitted if admitted else 0.0

    @property
    def recall(self) -> float:
        actual = self.tp + self.fn
        return self.tp / actual if actual else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def describe(self) -> dict[str, object]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


@dataclass(frozen=True)
class SweepPoint:
    """F1 at one uniform threshold, ignoring the length relaxation.

    Uniform because the sweep answers "what is the best achievable single value",
    and a two-parameter sweep answers a different question that the per-kind
    sweeps below cover one branch at a time.
    """

    threshold: float
    confusion: Confusion

    def describe(self) -> dict[str, object]:
        return {"threshold": round(self.threshold, 4), **self.confusion.describe()}


def confusion_at_shipped_gate(pairs: Sequence[Pair]) -> Confusion:
    """Score the gate as the store applied it, using the store's own verdict."""
    tp = sum(1 for p in pairs if p.relevant and p.admitted)
    fp = sum(1 for p in pairs if not p.relevant and p.admitted)
    fn = sum(1 for p in pairs if p.relevant and not p.admitted)
    tn = sum(1 for p in pairs if not p.relevant and not p.admitted)
    return Confusion(tp=tp, fp=fp, fn=fn, tn=tn)


def confusion_at(pairs: Sequence[Pair], threshold: float) -> Confusion:
    """Score a hypothetical uniform threshold against the same cosines."""
    tp = sum(1 for p in pairs if p.relevant and p.cosine >= threshold)
    fp = sum(1 for p in pairs if not p.relevant and p.cosine >= threshold)
    fn = sum(1 for p in pairs if p.relevant and p.cosine < threshold)
    tn = sum(1 for p in pairs if not p.relevant and p.cosine < threshold)
    return Confusion(tp=tp, fp=fp, fn=fn, tn=tn)


def sweep(pairs: Sequence[Pair]) -> tuple[SweepPoint, ...]:
    """F1 across the fixed grid. Integer step arithmetic keeps the grid exact."""
    steps = round((SWEEP_MAX - SWEEP_MIN) / SWEEP_STEP)
    return tuple(
        SweepPoint(threshold=t, confusion=confusion_at(pairs, t))
        for t in (round(SWEEP_MIN + i * SWEEP_STEP, 4) for i in range(steps + 1))
    )


def best_point(points: Sequence[SweepPoint]) -> SweepPoint:
    """The highest-F1 grid point, breaking ties toward the LOWER threshold.

    Toward the lower value because among equal-F1 cuts the looser one admits more
    relevant context, and a missed memory is the failure a user notices.
    """
    return max(points, key=lambda p: (p.confusion.f1, -p.threshold))


def f1_band(points: Sequence[SweepPoint], floor: float) -> tuple[float, float] | None:
    """The contiguous threshold range around the optimum where F1 stays >= *floor*.

    This is what says whether a reported F1 could have SELECTED a threshold. A
    band one grid step wide means the number pins the value; a band spanning a
    tenth of the cosine range means every threshold in it produces the same
    headline, so the headline is evidence about the corpus and not about the cut.
    Returns None when no sampled threshold reaches the requested floor.
    """
    if not points:
        raise AdmissionRefusal("cannot find a band in an empty sweep")
    best = best_point(points)
    if best.confusion.f1 < floor:
        return None
    ordered = sorted(points, key=lambda p: p.threshold)
    pivot = next(i for i, p in enumerate(ordered) if p.threshold == best.threshold)
    low = high = pivot
    while low > 0 and ordered[low - 1].confusion.f1 >= floor:
        low -= 1
    while high + 1 < len(ordered) and ordered[high + 1].confusion.f1 >= floor:
        high += 1
    return ordered[low].threshold, ordered[high].threshold


def balanced_subset(pairs: Sequence[Pair], topics: Sequence[AdmissionTopic]) -> tuple[Pair, ...]:
    """Each query's own fragment plus exactly one distractor, per length branch.

    Reproduces the 1:1 class balance a "N relevant + N irrelevant" protocol
    describes, so a claim made under that shape can be compared against this one
    without the full view's 49:1 imbalance dominating precision.
    """
    order = [t.topic_id for t in topics]
    wanted: set[tuple[str, str, str]] = set()
    for i, tid in enumerate(order):
        distractor = order[(i + DISTRACTOR_STRIDE) % len(order)]
        if distractor == tid:
            raise AdmissionRefusal(
                f"the distractor stride {DISTRACTOR_STRIDE} maps {tid!r} onto itself at "
                f"{len(order)} topics; it must be coprime with the topic count."
            )
        for kind in KINDS:
            wanted.add((tid, tid, kind))
            wanted.add((tid, distractor, kind))
    chosen = tuple(p for p in pairs if (p.query_topic, p.doc_topic, p.kind) in wanted)
    expected = len(order) * len(KINDS) * 2
    if len(chosen) != expected:
        raise AdmissionRefusal(
            f"the balanced view selected {len(chosen)} pairs, expected {expected}. "
            "The pair matrix is incomplete, so the balance is not 1:1."
        )
    return chosen


@dataclass(frozen=True)
class AdmissionReport:
    """Everything a reader needs to decide whether the gate is tuned."""

    model_id: str
    backend: str
    n_topics: int
    shipped_short_gate: float
    shipped_long_gate: float
    long_text_chars: int
    #: Every (query, fragment) pair — the population ``search_episodic`` scores.
    pairs: tuple[Pair, ...]
    #: The 1:1 subset, one distractor per query. See :func:`balanced_subset`.
    balanced: tuple[Pair, ...]
    #: Same measured cosines evaluated by V2's runtime admission policy. This
    #: compares the classifier only, not age weighting, MMR or context budgets.
    member_v2_pairs: tuple[Pair, ...] = ()

    def kind_pairs(self, kind: str) -> tuple[Pair, ...]:
        return tuple(p for p in self.pairs if p.kind == kind)

    def class_distribution(self, kind: str, relevant: bool) -> Distribution:
        return distribution([p.cosine for p in self.kind_pairs(kind) if p.relevant is relevant])

    def overlaps(self, kind: str) -> bool:
        """Whether the two classes' observed ranges intersect at all.

        An intersection means no single cut separates them, which is the
        difference between a threshold that was discovered and one that was
        chosen. Computed on ranges rather than on a tail percentile so it cannot
        be softened by picking a friendlier quantile.
        """
        rel = self.class_distribution(kind, True)
        irr = self.class_distribution(kind, False)
        return irr.maximum >= rel.minimum

    def false_positives(self, kind: str, limit: int = 5) -> tuple[Pair, ...]:
        """The highest-scoring wrongly-admitted pairs, for label review.

        Printed because a reader must be able to check the labels themselves: if
        the top false positives are pairs a person would call related, the
        overlap is a labelling artefact and the finding is weaker. If they are
        plainly unrelated, it is not.
        """
        wrong = [p for p in self.kind_pairs(kind) if not p.relevant and p.admitted]
        return tuple(sorted(wrong, key=lambda p: p.cosine, reverse=True)[:limit])

    def describe(self) -> dict[str, object]:
        out: dict[str, object] = {
            "model_id": self.model_id,
            "backend": self.backend,
            "n_topics": self.n_topics,
            "gate": {
                "short": self.shipped_short_gate,
                "long": self.shipped_long_gate,
                "long_text_chars": self.long_text_chars,
            },
            "n_pairs_full": len(self.pairs),
            "n_pairs_balanced": len(self.balanced),
        }
        if self.member_v2_pairs:
            out["member_v2_policy"] = {
                "revision": memory_v2.ALGORITHM_VERSION,
                "provisional_model_dependent": True,
                "short_gate": memory_v2.SHORT_COSINE_FLOOR,
                "long_gate": memory_v2.LONG_COSINE_FLOOR,
                "scope": "admission_only_not_ranking_or_context_budget",
                **{
                    kind: confusion_at_shipped_gate(
                        tuple(p for p in self.member_v2_pairs if p.kind == kind)
                    ).describe()
                    for kind in KINDS
                },
            }
        for kind in KINDS:
            full = self.kind_pairs(kind)
            bal = tuple(p for p in self.balanced if p.kind == kind)
            points = sweep(full)
            bal_points = sweep(bal)
            bal_best = best_point(bal_points)
            band = f1_band(bal_points, 0.98)
            out[kind] = {
                "shipped_full": confusion_at_shipped_gate(full).describe(),
                "shipped_balanced": confusion_at_shipped_gate(bal).describe(),
                "relevant": self.class_distribution(kind, True).describe(),
                "irrelevant": self.class_distribution(kind, False).describe(),
                "distributions_overlap": self.overlaps(kind),
                "best_uniform": best_point(points).describe(),
                "best_uniform_balanced": bal_best.describe(),
                "balanced_f1_band_98": list(band) if band is not None else None,
                "sweep": [p.describe() for p in points],
                "sweep_balanced": [p.describe() for p in bal_points],
                "top_false_positives": [
                    {
                        "query_topic": p.query_topic,
                        "doc_topic": p.doc_topic,
                        "cosine": round(p.cosine, 4),
                    }
                    for p in self.false_positives(kind)
                ],
            }
        return out


def measure(
    embed_fn: EmbedFn,
    *,
    db_path: Path,
    topics: Sequence[AdmissionTopic] = ADMISSION_TOPICS,
) -> AdmissionReport:
    """Load the corpus into a real store and read the gate's verdict per pair.

    The cosine values are the store's, not this module's: every pair is scored by
    ``search_episodic`` and admission is decided by a second call with
    ``relevance_filter=True`` — the exact call ``get_episodic_context`` makes. A
    reimplemented cosine or a reimplemented comparison would measure this file
    rather than the product, and the comparison in particular is subtle (the gate
    reads a cosine already rounded to four places).
    """
    validate_corpus(topics)
    probe = embed_fn("admission benchmark embedding width probe")
    if not probe:
        raise AdmissionRefusal(
            "the embedding model is not resident, so every fragment would be stored "
            "with a NULL embedding and search_episodic would fall back to FTS5 "
            "keyword matching. That measures term overlap, not a cosine threshold."
        )

    store = VectorMemoryStore(db_path=db_path, embedding_dim=len(probe))
    store.init()
    try:
        store.embed_fn = embed_fn  # type: ignore[attr-defined]  # the wiring production uses
        text_owner: dict[str, tuple[str, str]] = {}
        for topic in topics:
            for kind in KINDS:
                text = getattr(topic, kind)
                if not store.write_episodic(text=text, source="benchmark"):
                    raise AdmissionRefusal(
                        f"the store refused to write {topic.topic_id}.{kind}, so the "
                        "haystack is smaller than the corpus and every rate below "
                        "would describe a store that never held it."
                    )
                text_owner[text] = (topic.topic_id, kind)

        n_docs = len(topics) * len(KINDS)
        pairs: list[Pair] = []
        member_pairs: list[Pair] = []
        for topic in topics:
            query_vec = embed_fn(topic.query)
            if not query_vec:
                raise AdmissionRefusal(
                    f"embedding the query for {topic.topic_id} returned nothing; the "
                    "model became unavailable part way through the run."
                )
            # mmr=False so the returned set is the ranked candidates themselves.
            # MMR would rerank on pairwise diversity and, at limit == the whole
            # store, still return every row — but it is off because the quantity
            # being measured is the gate, and leaving a second selection stage in
            # the path invites reading its effects as the gate's.
            scored = store.search_episodic(
                query_embedding=query_vec,
                query_text=topic.query,
                limit=n_docs,
                mmr=False,
                relevance_filter=False,
            )
            kept = store.search_episodic(
                query_embedding=query_vec,
                query_text=topic.query,
                limit=n_docs,
                mmr=False,
                relevance_filter=True,
            )
            if len(scored) != n_docs:
                raise AdmissionRefusal(
                    f"the query for {topic.topic_id} scored {len(scored)} of {n_docs} "
                    "fragments; the pair matrix would be incomplete."
                )
            admitted_ids = {row["id"] for row in kept}
            for row in scored:
                owner = text_owner.get(row["text"])
                if owner is None:
                    raise AdmissionRefusal(
                        "the store returned a fragment the corpus does not contain; "
                        f"the database at {db_path} was not empty."
                    )
                doc_topic, kind = owner
                pairs.append(
                    Pair(
                        query_topic=topic.topic_id,
                        doc_topic=doc_topic,
                        kind=kind,
                        cosine=float(row["cosine_sim"]),
                        gate=(
                            _EPISODIC_LONG_TEXT_THRESHOLD
                            if kind == LONG
                            else _EPISODIC_RELEVANCE_THRESHOLD
                        ),
                        relevant=doc_topic == topic.topic_id,
                        admitted=row["id"] in admitted_ids,
                    )
                )
                evidence = memory_v2.relevance_evidence(
                    memory_v2.terms(topic.query), row["text"], float(row["cosine_sim"])
                )
                member_pairs.append(
                    Pair(
                        query_topic=topic.topic_id,
                        doc_topic=doc_topic,
                        kind=kind,
                        cosine=float(row["cosine_sim"]),
                        gate=evidence["cosine_floor"],
                        relevant=doc_topic == topic.topic_id,
                        admitted=evidence["admitted"],
                    )
                )
    finally:
        store.close()

    from kiro_crew.embeddings import get_shared_embedder

    return AdmissionReport(
        model_id=get_shared_embedder().model_id,
        backend=search_backend(),
        n_topics=len(topics),
        shipped_short_gate=_EPISODIC_RELEVANCE_THRESHOLD,
        shipped_long_gate=_EPISODIC_LONG_TEXT_THRESHOLD,
        long_text_chars=_EPISODIC_LONG_TEXT_CHARS,
        pairs=tuple(pairs),
        balanced=balanced_subset(pairs, topics),
        member_v2_pairs=tuple(member_pairs),
    )


def format_report(report: AdmissionReport) -> str:
    """A plain-text rendering, ordered so the distributions come before the F1."""
    out: list[str] = [
        "Episodic admission gate — measured",
        f"  embedder      {report.model_id}",
        f"  search path   {report.backend}",
        f"  corpus        {report.n_topics} topics, "
        f"{len(report.pairs)} pairs full / {len(report.balanced)} balanced",
        f"  shipped gate  short<={report.long_text_chars}ch "
        f"{report.shipped_short_gate}   long>{report.long_text_chars}ch "
        f"{report.shipped_long_gate}",
    ]
    for kind in KINDS:
        full = report.kind_pairs(kind)
        bal = tuple(p for p in report.balanced if p.kind == kind)
        rel = report.class_distribution(kind, True)
        irr = report.class_distribution(kind, False)
        points = sweep(full)
        best = best_point(points)
        gate = report.shipped_long_gate if kind == LONG else report.shipped_short_gate
        out += [
            "",
            f"── {kind} fragments (gate {gate}) ─────────────────────────",
            "  cosine distribution      n     min     p50     p90     p99     max",
            f"    relevant          {rel.n:6d}  {rel.minimum:6.3f}  {rel.median:6.3f}  "
            f"{rel.p90:6.3f}  {rel.p99:6.3f}  {rel.maximum:6.3f}",
            f"    irrelevant        {irr.n:6d}  {irr.minimum:6.3f}  {irr.median:6.3f}  "
            f"{irr.p90:6.3f}  {irr.p99:6.3f}  {irr.maximum:6.3f}",
            f"  distributions overlap: {'YES' if report.overlaps(kind) else 'no'}"
            f"  (irrelevant max {irr.maximum:.3f} vs relevant min {rel.minimum:.3f})",
        ]
        for name, subset in (("full 1:%d" % (report.n_topics - 1), full), ("balanced 1:1", bal)):
            c = confusion_at_shipped_gate(subset)
            out.append(
                f"  at the shipped gate, {name:<12} "
                f"P={c.precision:.3f} R={c.recall:.3f} F1={c.f1:.3f} "
                f"(tp={c.tp} fp={c.fp} fn={c.fn} tn={c.tn})"
            )
        bc = best.confusion
        out.append(
            f"  best uniform threshold on the grid: {best.threshold:.2f} "
            f"→ F1={bc.f1:.3f} (P={bc.precision:.3f} R={bc.recall:.3f})"
        )
        bal_points = sweep(bal)
        band = f1_band(bal_points, 0.98)
        if band is None:
            out.append("  under the 1:1 balance, F1 never reaches 0.98 on the sweep grid")
        else:
            band_low, band_high = band
            out.append(
                f"  under the 1:1 balance, F1 stays >= 0.98 for every threshold in "
                f"[{band_low:.2f}, {band_high:.2f}]"
            )
        out.append("  F1 by threshold:")
        for point in points:
            if round(point.threshold * 100) % 5:
                continue
            c = point.confusion
            out.append(
                f"    {point.threshold:.2f}  F1={c.f1:.3f}  P={c.precision:.3f}  "
                f"R={c.recall:.3f}"
            )
        out.append("  highest-scoring wrongly-admitted pairs:")
        for fp in report.false_positives(kind):
            out.append(f"    {fp.cosine:.3f}  {fp.query_topic!r} ← {fp.doc_topic!r}")
    if report.member_v2_pairs:
        out.append(
            f"\nMember V2 admission ({memory_v2.ALGORITHM_VERSION}; model-dependent provisional policy):"
        )
        for kind in KINDS:
            c = confusion_at_shipped_gate(
                tuple(p for p in report.member_v2_pairs if p.kind == kind)
            )
            out.append(f"  {kind}: P={c.precision:.3f} R={c.recall:.3f} F1={c.f1:.3f}")
        out.append("  Admission only; this does not evaluate ranking or context budgets.")
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark and print the report. ``--json`` writes the raw numbers."""
    parser = argparse.ArgumentParser(
        prog="python -m kiro_crew.eval.bench.admission",
        description=(
            "Measure the episodic admission gate as a classifier: precision, recall, "
            "F1, both cosine distributions, and a threshold sweep. Needs a resident "
            "embedding model."
        ),
    )
    parser.add_argument("--json", type=Path, help="also write the full report as JSON here")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING)
    from .ingest import prepare_embedder

    try:
        embed_fn = prepare_embedder()
        with tempfile.TemporaryDirectory(prefix="kc-admission-bench-") as tmp:
            report = measure(embed_fn, db_path=Path(tmp) / "admission.db")
    except BenchRefusal as exc:
        print(f"refused: {exc}")
        return 2
    print(format_report(report))
    if args.json:
        args.json.write_text(json.dumps(report.describe(), indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
