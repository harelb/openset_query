"""Semantic data types for epistemic grounding (pure, stdlib-only).

Design rules, in order of importance:

1. A ``GroundingQuery`` is the QUESTION. It never carries detector fields
   (scores, frame counts, thresholds) -- those belong to evidence and to
   admissibility models. ``required_confidence`` defaults to 0.0 so that in
   legacy-parity mode "admissible evidence exists" is exactly sufficient,
   reproducing today's mission semantics.

2. An ``EvidenceRecord`` is a faithful statement of WHAT WAS OBSERVED (or
   what a completed search failed to observe), never of what is true.
   Temporal decay is deliberately NOT baked into the record: records are
   archive facts, and different ``TemporalValidityModel``s must be
   replayable over the same serialized records.

3. Negative evidence is a record of a COMPLETED SEARCH, not the absence of
   records. Its ``coverage`` field (0..1, exploration-grid fraction or
   archive-scan completeness -- the producer says which in ``provenance``)
   is what separates CONTRADICTED from UNKNOWN. ``coverage=None`` means
   "unknown", which can never satisfy a sufficiency test.

4. Every field an existing producer cannot supply is Optional. Nothing in
   this module invents values.


Vendored from dcist_sim_isaac.grounding (dcist_sim @ 7e8bf5a) for the
planability-assessment work; keep semantics in lockstep with the original.
Design invariants (do not break): decay applied only at compute_status;
negative evidence = completed search with coverage; coverage=None never
contradicts.
"""
from __future__ import annotations

import enum
import json
from dataclasses import dataclass, field


class EvidenceSource(enum.Enum):
    """Where a piece of evidence came from.

    Values match the mission's ``discovered.source`` vocabulary where one
    exists (``dsg`` / ``requery`` / ``archive_requery``), extended with the
    sources the mission has but never named.
    """

    COMMITTED_DSG = "committed_dsg"      # live/committed scene graph query
    PRIOR_ARCHIVE = "prior_archive"      # tier 2a: another mission's keyframes
    CURRENT_ARCHIVE = "current_archive"  # tier 2b: this mission's keyframes
    FRONTIER_SEARCH = "frontier_search"  # exploration-grid coverage sweep
    INJECTED_NODE = "injected_node"      # a node this mission wrote itself
    HUMAN = "human"                      # reserved; no producer yet


# The mission's `discovered.source` strings -> EvidenceSource.
DISCOVERY_SOURCE_MAP = {
    "dsg": EvidenceSource.COMMITTED_DSG,
    "requery": EvidenceSource.CURRENT_ARCHIVE,
    "archive_requery": EvidenceSource.PRIOR_ARCHIVE,
}


class Polarity(enum.Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


class EpistemicStatus(enum.Enum):
    """The three-way verdict. UNKNOWN is not CONTRADICTED: a failure to
    detect, a dead detector, or thin coverage all leave the proposition
    OPEN, and nothing in this package may silently convert them to false."""

    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNKNOWN = "unknown"


class ReasonCode(enum.Enum):
    """Machine-readable justification attached to every GroundingResult.

    Kept deliberately small; add a member only when a producer can actually
    emit it (the enum is an interface, not an aspiration).
    """

    ADMISSIBLE_POSITIVE_EVIDENCE = "admissible_positive_evidence"
    UNSUPPORTED_AFTER_SUFFICIENT_SEARCH = "unsupported_after_sufficient_search"
    INSUFFICIENT_COVERAGE = "insufficient_coverage"
    PERCEPTION_UNAVAILABLE = "perception_unavailable"
    STALE_HISTORICAL_EVIDENCE = "stale_historical_evidence"
    BELOW_REQUIRED_CONFIDENCE = "below_required_confidence"
    NO_GROUNDING_OPERATOR = "no_grounding_operator"
    NO_EVIDENCE = "no_evidence"
    NO_CALIBRATION = "no_calibration"


@dataclass(frozen=True)
class GroundingQuery:
    """A proposition the planner wants evidence about.

    ``predicate`` is a lowercase family name ("exists", "object-in-region",
    "reachable", "current-observed", ...); ``args`` are canonical class
    names / region labels, already normalized by the caller
    (``nl_openset.normalize_class`` is the house normalizer).
    """

    predicate: str
    args: tuple
    required_confidence: float = 0.0
    freshness_s: float | None = None

    def __post_init__(self):
        object.__setattr__(self, "args", tuple(self.args))

    def key(self) -> str:
        """Stable display/log key, e.g. ``exists(suitcase)``."""
        return "%s(%s)" % (self.predicate, ",".join(str(a) for a in self.args))

    def to_json(self) -> dict:
        d = {"predicate": self.predicate, "args": list(self.args)}
        if self.required_confidence:
            d["required_confidence"] = self.required_confidence
        if self.freshness_s is not None:
            d["freshness_s"] = self.freshness_s
        return d

    @classmethod
    def from_json(cls, d: dict) -> "GroundingQuery":
        return cls(
            predicate=str(d["predicate"]),
            args=tuple(d.get("args", ())),
            required_confidence=float(d.get("required_confidence", 0.0)),
            freshness_s=d.get("freshness_s"),
        )


@dataclass(frozen=True)
class EvidenceRecord:
    """One unit of evidence about one proposition.

    ``confidence`` is the detector score for perceptual positives; ``None``
    for structural facts (a DSG node either exists or it does not) -- a
    ``None`` confidence is treated as 1.0 by belief math, because the
    producing channel offered no graded signal, not because we are certain.

    ``coverage`` is only meaningful on NEGATIVE records (rule 3 in the
    module docstring). ``provenance`` carries whatever the producing event
    actually had: score / n_frames / n_pixels / frame_ts_range / min_score /
    best_rejected_score / keyframe_dir / symbol / requery_enabled / ...
    """

    evidence_id: str
    source: EvidenceSource
    polarity: Polarity
    predicate: str
    args: tuple
    t_recorded_ns: int
    confidence: float | None = None
    t_collected_ns: int | None = None
    session_id: str | None = None
    coverage: float | None = None
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "args", tuple(self.args))

    @property
    def effective_confidence(self) -> float:
        return 1.0 if self.confidence is None else float(self.confidence)

    def to_json(self) -> dict:
        return {
            "evidence_id": self.evidence_id,
            "source": self.source.value,
            "polarity": self.polarity.value,
            "predicate": self.predicate,
            "args": list(self.args),
            "t_recorded_ns": self.t_recorded_ns,
            "confidence": self.confidence,
            "t_collected_ns": self.t_collected_ns,
            "session_id": self.session_id,
            "coverage": self.coverage,
            "provenance": self.provenance,
        }

    @classmethod
    def from_json(cls, d: dict) -> "EvidenceRecord":
        return cls(
            evidence_id=str(d["evidence_id"]),
            source=EvidenceSource(d["source"]),
            polarity=Polarity(d["polarity"]),
            predicate=str(d["predicate"]),
            args=tuple(d.get("args", ())),
            t_recorded_ns=int(d["t_recorded_ns"]),
            confidence=d.get("confidence"),
            t_collected_ns=d.get("t_collected_ns"),
            session_id=d.get("session_id"),
            coverage=d.get("coverage"),
            provenance=dict(d.get("provenance", {})),
        )


@dataclass(frozen=True)
class IdentityHypothesis:
    """Minimal identity-uncertainty stub (deliberately not correspondence).

    One hypothesis per fused detection cluster (`merge_detections` clusters
    ARE identity hypotheses), per DSG object node, or per injected node.
    Cross-source association is out of scope for the first prototype; two
    hypotheses of the same class within a merge radius but from different
    sources are a QUESTION (for a disambiguation policy), never silently
    merged.
    """

    hypothesis_id: str
    class_name: str
    xy: tuple
    evidence_ids: tuple = ()

    def to_json(self) -> dict:
        return {
            "hypothesis_id": self.hypothesis_id,
            "class_name": self.class_name,
            "xy": list(self.xy),
            "evidence_ids": list(self.evidence_ids),
        }


@dataclass(frozen=True)
class GroundingResult:
    """The resolution of one GroundingQuery against a set of evidence.

    ``records`` carries the full EvidenceRecords the status was computed
    from -- in memory only (to_json serializes their IDs via supporting/
    contradicting; the records themselves live in grounding.jsonl).
    """

    query: GroundingQuery
    status: EpistemicStatus
    reason: ReasonCode
    belief: float | None = None
    supporting: tuple = ()
    contradicting: tuple = ()
    hypotheses: tuple = ()
    suggested_next: str | None = None
    records: tuple = ()

    def to_json(self) -> dict:
        return {
            "query": self.query.to_json(),
            "status": self.status.value,
            "reason": self.reason.value,
            "belief": self.belief,
            "supporting": list(self.supporting),
            "contradicting": list(self.contradicting),
            "hypotheses": [h.to_json() for h in self.hypotheses],
            "suggested_next": self.suggested_next,
        }


def dumps_record(obj) -> str:
    """One-line JSON for jsonl sinks (shared so key order stays stable)."""
    return json.dumps(obj.to_json() if hasattr(obj, "to_json") else obj,
                      sort_keys=True)
