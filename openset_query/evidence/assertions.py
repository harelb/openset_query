"""Persisted epistemic state: Assertions and EvidenceObligations (stdlib-only).

An ``Assertion`` is a *materialized* :class:`GroundingResult` — the answer to
one :class:`GroundingQuery` at a point in time, with identity and enough
metadata to be stored in Neo4j, aged, and recomputed. Assertions are always
recomputable from their underlying evidence (Observations / SearchEpisodes);
the graph copy is a cache of ``compute_status``, never the only copy.

An ``EvidenceObligation`` is what an optimistic plan emits for each fact it
assumed without adequate certification: the query to resolve, the verification
mechanism chosen to resolve it (a versioned skill-contract name, e.g.
``reprocess_memory`` / ``acquire_view``), and its lifecycle state.
"""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field, replace

from .types import EpistemicStatus, GroundingQuery, GroundingResult, ReasonCode


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass(frozen=True)
class Assertion:
    """One persisted epistemic verdict about one proposition.

    ``scope`` records what the verdict is *about* beyond the predicate args —
    e.g. a spatial validity region or the coverage basis of a negative — as an
    open dict (producers write only what they actually have; see the
    EvidenceRecord.provenance discipline). ``policy_version`` pins the status
    semantics + admissibility configuration that computed this verdict so a
    stored Assertion can be invalidated when policy changes.
    """

    assertion_id: str
    query: GroundingQuery
    status: EpistemicStatus
    reason: ReasonCode
    computed_at_ns: int
    belief: float | None = None
    supporting: tuple = ()  # evidence_ids
    contradicting: tuple = ()  # evidence_ids
    freshness_s: float | None = None
    scope: dict = field(default_factory=dict)
    policy_version: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "supporting", tuple(self.supporting))
        object.__setattr__(self, "contradicting", tuple(self.contradicting))

    @classmethod
    def from_result(
        cls,
        result: GroundingResult,
        computed_at_ns: int,
        assertion_id: str | None = None,
        policy_version: str | None = None,
        scope: dict | None = None,
    ) -> "Assertion":
        return cls(
            assertion_id=assertion_id or _new_id("assert"),
            query=result.query,
            status=result.status,
            reason=result.reason,
            computed_at_ns=int(computed_at_ns),
            belief=result.belief,
            supporting=tuple(result.supporting),
            contradicting=tuple(result.contradicting),
            freshness_s=result.query.freshness_s,
            scope=dict(scope or {}),
            policy_version=policy_version,
        )

    def is_fresh(self, now_ns: int) -> bool:
        """True while the verdict is within its freshness horizon (no horizon
        -> always fresh; staleness of the *evidence* is the temporal model's
        job at recompute time, this only ages the cached verdict)."""
        if self.freshness_s is None:
            return True
        return (int(now_ns) - self.computed_at_ns) <= self.freshness_s * 1e9

    def certifies(self, now_ns: int) -> bool:
        """True iff a strict planner may use this fact: SUPPORTED, at or above
        the query's required confidence, and still fresh."""
        if self.status is not EpistemicStatus.SUPPORTED:
            return False
        if not self.is_fresh(now_ns):
            return False
        if self.belief is None:
            return True  # ungraded channel (structural fact)
        return self.belief >= self.query.required_confidence

    def to_json(self) -> dict:
        return {
            "assertion_id": self.assertion_id,
            "query": self.query.to_json(),
            "status": self.status.value,
            "reason": self.reason.value,
            "computed_at_ns": self.computed_at_ns,
            "belief": self.belief,
            "supporting": list(self.supporting),
            "contradicting": list(self.contradicting),
            "freshness_s": self.freshness_s,
            "scope": self.scope,
            "policy_version": self.policy_version,
        }

    @classmethod
    def from_json(cls, d: dict) -> "Assertion":
        return cls(
            assertion_id=str(d["assertion_id"]),
            query=GroundingQuery.from_json(d["query"]),
            status=EpistemicStatus(d["status"]),
            reason=ReasonCode(d["reason"]),
            computed_at_ns=int(d["computed_at_ns"]),
            belief=d.get("belief"),
            supporting=tuple(d.get("supporting", ())),
            contradicting=tuple(d.get("contradicting", ())),
            freshness_s=d.get("freshness_s"),
            scope=dict(d.get("scope", {})),
            policy_version=d.get("policy_version"),
        )


class ObligationStatus(enum.Enum):
    OPEN = "open"
    RESOLVED = "resolved"
    FAILED = "failed"


@dataclass(frozen=True)
class EvidenceObligation:
    """A commitment to verify one unresolved fact an optimistic plan relies on.

    ``mechanism`` names the VerificationMechanism (from the active capability
    profile) selected to discharge the obligation; ``estimated_cost`` follows
    the GroundingOperator convention ``{"time_s": .., "motion_m": ..}``.
    Lifecycle transitions return new frozen instances (records stay
    append-only in logs).
    """

    obligation_id: str
    query: GroundingQuery
    mechanism: str
    created_at_ns: int
    status: ObligationStatus = ObligationStatus.OPEN
    assertion_id: str | None = None  # the Assertion this must update
    resolved_at_ns: int | None = None
    estimated_cost: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        query: GroundingQuery,
        mechanism: str,
        created_at_ns: int,
        assertion_id: str | None = None,
        estimated_cost: dict | None = None,
        provenance: dict | None = None,
    ) -> "EvidenceObligation":
        return cls(
            obligation_id=_new_id("oblig"),
            query=query,
            mechanism=mechanism,
            created_at_ns=int(created_at_ns),
            assertion_id=assertion_id,
            estimated_cost=dict(estimated_cost or {}),
            provenance=dict(provenance or {}),
        )

    def resolved(self, resolved_at_ns: int) -> "EvidenceObligation":
        return replace(
            self, status=ObligationStatus.RESOLVED, resolved_at_ns=int(resolved_at_ns)
        )

    def failed(self, resolved_at_ns: int) -> "EvidenceObligation":
        return replace(
            self, status=ObligationStatus.FAILED, resolved_at_ns=int(resolved_at_ns)
        )

    def to_json(self) -> dict:
        return {
            "obligation_id": self.obligation_id,
            "query": self.query.to_json(),
            "mechanism": self.mechanism,
            "created_at_ns": self.created_at_ns,
            "status": self.status.value,
            "assertion_id": self.assertion_id,
            "resolved_at_ns": self.resolved_at_ns,
            "estimated_cost": self.estimated_cost,
            "provenance": self.provenance,
        }

    @classmethod
    def from_json(cls, d: dict) -> "EvidenceObligation":
        return cls(
            obligation_id=str(d["obligation_id"]),
            query=GroundingQuery.from_json(d["query"]),
            mechanism=str(d["mechanism"]),
            created_at_ns=int(d["created_at_ns"]),
            status=ObligationStatus(d.get("status", "open")),
            assertion_id=d.get("assertion_id"),
            resolved_at_ns=d.get("resolved_at_ns"),
            estimated_cost=dict(d.get("estimated_cost", {})),
            provenance=dict(d.get("provenance", {})),
        )
