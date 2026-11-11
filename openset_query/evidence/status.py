"""The semantic core: evidence -> epistemic status (one pure function).

This function IS the UNKNOWN != CONTRADICTED contract. Decision order:

1. No records at all                          -> UNKNOWN / NO_EVIDENCE
2. Admissible positive evidence, temporally
   valid, effective confidence >= required    -> SUPPORTED
3. Positives exist but all decayed stale      -> UNKNOWN / STALE_HISTORICAL_EVIDENCE
   (never CONTRADICTED -- fading support is not disproof)
4. Positives exist but under-confident        -> UNKNOWN / BELOW_REQUIRED_CONFIDENCE
5. No positive; a NEGATIVE record of a
   completed search with coverage >= bar,
   from a channel that was CAPABLE of seeing
   the class and ALIVE at the time            -> CONTRADICTED / UNSUPPORTED_AFTER_SUFFICIENT_SEARCH
6. No positive; search channel dead or blind  -> UNKNOWN / PERCEPTION_UNAVAILABLE
7. No positive; coverage under the bar        -> UNKNOWN / INSUFFICIENT_COVERAGE

Records handed to this function are ALREADY admissibility-filtered (the
producing operator applied its EvidenceAdmissibilityModel; rejected raw
detections live on as provenance like ``best_rejected_score``, not as
records). Keeping admissibility upstream keeps this function replayable over
serialized evidence with different temporal models.

DELIBERATE DIVERGENCE FROM LEGACY, documented rather than papered over:
today `should_exit_not_found` licenses exit 4 when the worker died
(`requery_enabled=False`) and exploration ran out -- operationally a refusal,
epistemically an UNKNOWN. The mission's refusal DECISION stays reproducible
via `policy.refusal_licensed`; this function reports the honest LABEL. The
gap between the two is a research finding, and the grounding log records
both.


Vendored from dcist_sim_isaac.grounding (dcist_sim @ 7e8bf5a) for the
planability-assessment work; keep semantics in lockstep with the original.
Design invariants (do not break): decay applied only at compute_status;
negative evidence = completed search with coverage; coverage=None never
contradicts.
"""

from __future__ import annotations

from .temporal import NoDecay
from .types import (
    EpistemicStatus,
    EvidenceRecord,
    GroundingQuery,
    GroundingResult,
    Polarity,
    ReasonCode,
)

# Mirrors the mission's --coverage-limit default (explore_mission.py). A
# drift-guard test asserts equality; importing explore_mission here would
# drag argparse/main-module machinery into the semantic core for one float.
DEFAULT_COVERAGE_SUFFICIENCY = 0.95


def _search_channel_alive(record: EvidenceRecord) -> bool:
    """Was the perception channel behind this negative record alive?

    The mission records `requery_enabled` on its `negative_gate` event; a
    negative record lacking the key is treated as alive (structural sources
    like a DSG lookup have no worker to die).
    """
    return bool(record.provenance.get("requery_enabled", True))


def compute_status(
    query: GroundingQuery,
    records,
    *,
    temporal=None,
    now_ns=None,
    coverage_sufficiency=DEFAULT_COVERAGE_SUFFICIENCY,
    channel_capable=True,
) -> GroundingResult:
    """Resolve one query against its evidence. Pure; no I/O, no clock.

    ``now_ns`` is the caller's clock (pass mission start / report time);
    required only when ``temporal`` actually decays. ``channel_capable=False``
    states that no available channel could structurally have observed the
    class (novel class vs a closed-labelspace DSG) -- negative evidence from
    a blind channel can never contradict.
    """
    temporal = temporal or NoDecay()
    records = tuple(records)
    if not records:
        return GroundingResult(
            query=query, status=EpistemicStatus.UNKNOWN, reason=ReasonCode.NO_EVIDENCE
        )

    positives = [r for r in records if r.polarity is Polarity.POSITIVE]
    negatives = [r for r in records if r.polarity is Polarity.NEGATIVE]

    if positives:
        fresh, stale = [], []
        for r in positives:
            (stale if temporal.is_stale(r, now_ns) else fresh).append(r)
        if fresh:
            scored = [
                (temporal.weight(r, now_ns) * r.effective_confidence, r) for r in fresh
            ]
            best_eff, _best = max(scored, key=lambda p: p[0])
            if best_eff >= query.required_confidence:
                return GroundingResult(
                    query=query,
                    status=EpistemicStatus.SUPPORTED,
                    reason=ReasonCode.ADMISSIBLE_POSITIVE_EVIDENCE,
                    belief=best_eff,
                    supporting=tuple(r.evidence_id for _e, r in scored),
                    records=records,
                )
            return GroundingResult(
                query=query,
                status=EpistemicStatus.UNKNOWN,
                reason=ReasonCode.BELOW_REQUIRED_CONFIDENCE,
                belief=best_eff,
                supporting=tuple(r.evidence_id for _e, r in scored),
                records=records,
            )
        # Every positive decayed below the temporal floor.
        decayed = max(
            temporal.weight(r, now_ns) * r.effective_confidence for r in stale
        )
        return GroundingResult(
            query=query,
            status=EpistemicStatus.UNKNOWN,
            reason=ReasonCode.STALE_HISTORICAL_EVIDENCE,
            belief=decayed,
            supporting=tuple(r.evidence_id for r in stale),
            records=records,
        )

    # No positive evidence. Only a completed, sufficiently-covering search
    # by a capable, alive channel may contradict.
    if negatives:
        dead = [r for r in negatives if not _search_channel_alive(r)]
        alive = [r for r in negatives if _search_channel_alive(r)]
        if not channel_capable or (dead and not alive):
            return GroundingResult(
                query=query,
                status=EpistemicStatus.UNKNOWN,
                reason=ReasonCode.PERCEPTION_UNAVAILABLE,
                contradicting=tuple(r.evidence_id for r in negatives),
                records=records,
            )
        sufficient = [
            r
            for r in alive
            if r.coverage is not None and r.coverage >= coverage_sufficiency
        ]
        if sufficient:
            return GroundingResult(
                query=query,
                status=EpistemicStatus.CONTRADICTED,
                reason=ReasonCode.UNSUPPORTED_AFTER_SUFFICIENT_SEARCH,
                belief=0.0,
                contradicting=tuple(r.evidence_id for r in sufficient),
                records=records,
            )
        return GroundingResult(
            query=query,
            status=EpistemicStatus.UNKNOWN,
            reason=ReasonCode.INSUFFICIENT_COVERAGE,
            contradicting=tuple(r.evidence_id for r in alive),
            records=records,
        )

    # Records existed but were neither positive nor negative -- unreachable
    # with the two-valued Polarity enum, kept for fail-closed completeness.
    return GroundingResult(
        query=query,
        status=EpistemicStatus.UNKNOWN,
        reason=ReasonCode.NO_EVIDENCE,
        records=records,
    )
