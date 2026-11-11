"""Acquisition policies: which unresolved proposition to verify next.

Ported from ``dcist_sim_isaac.grounding.policy`` (dcist_sim @ 7e8bf5a) for the
planability-assessment work — the ``choose_next_query`` half only (the
FixedTier mission-ladder advisor is mission-specific and stays behind).
Transparent heuristics, deliberately not value-of-information estimates; the
interface is shaped so a VOI objective can slot in later.

Duck-typing: ``results`` is any iterable of objects with ``.status``
(:class:`EpistemicStatus`), ``.belief`` (float | None), and ``.query`` — both
the vendored :class:`~openset_query.evidence.GroundingResult` and the
persisted :class:`~openset_query.evidence.Assertion` qualify. ``plans``
is any iterable of objects with ``.queries``
(the planner's ``SkeletonRequirement``
qualifies, as does the original PlanSketch).
"""

from __future__ import annotations

from .types import EpistemicStatus


def _unknowns(results):
    return [r for r in results if r.status is EpistemicStatus.UNKNOWN]


def _uncertainty(result) -> float:
    """Distance of belief from certainty; belief None reads as maximally
    uncertain (0.5). Transparent, not calibrated — see module docstring."""
    b = 0.5 if result.belief is None else float(result.belief)
    return 1.0 - abs(b - 0.5) * 2.0


class MostUncertainPredicatePolicy:
    """Resolve the UNKNOWN predicate with belief closest to 0.5 first."""

    name = "most-uncertain"

    def choose_next_query(
        self, results, available_operators=(), cost_fn=None, plans=None
    ):
        unknowns = _unknowns(results)
        if not unknowns:
            return None
        return max(unknowns, key=_uncertainty).query


class CheapestGroundingPolicy:
    """Resolve whichever UNKNOWN predicate has the cheapest available
    operator (ties broken by result order)."""

    name = "cheapest"

    def choose_next_query(
        self, results, available_operators=(), cost_fn=None, plans=None
    ):
        unknowns = _unknowns(results)
        if not unknowns:
            return None
        if cost_fn is None:
            return unknowns[0].query
        best, best_cost = None, None
        for res in unknowns:
            ops = [op for op in available_operators if op.can_evaluate(res.query)]
            if not ops:
                continue
            cost = min(cost_fn(op, res.query) for op in ops)
            if best_cost is None or cost < best_cost:
                best, best_cost = res.query, cost
        return best


class PlanDisambiguationPolicy:
    """Prioritize predicates whose resolution most changes which candidate
    plans remain feasible.

    Heuristic (transparent approximation of value-of-information): for each
    UNKNOWN predicate, count the candidate plans that require it AND are not
    already dead (no CONTRADICTED requirement). A predicate required by every
    live plan scores the number of live plans (resolving it negatively could
    kill everything — maximally decision-relevant); a predicate required by no
    live plan scores 0 and is never investigated. Ties break toward higher
    uncertainty.
    """

    name = "plan-disambiguation"

    def choose_next_query(
        self, results, available_operators=(), cost_fn=None, plans=None
    ):
        unknowns = _unknowns(results)
        if not unknowns:
            return None
        if not plans:
            # Without plan structure, degrade to most-uncertain.
            return max(unknowns, key=_uncertainty).query

        by_key = {r.query.key(): r for r in results}
        live_plans = []
        for plan in plans:
            dead = any(
                by_key.get(q.key()) is not None
                and by_key[q.key()].status is EpistemicStatus.CONTRADICTED
                for q in plan.queries
            )
            if not dead:
                live_plans.append(plan)

        def relevance(res):
            key = res.query.key()
            return sum(
                1 for plan in live_plans if any(q.key() == key for q in plan.queries)
            )

        scored = [(relevance(r), _uncertainty(r), r) for r in unknowns]
        scored = [s for s in scored if s[0] > 0]
        if not scored:
            return None  # nothing UNKNOWN is decision-relevant
        scored.sort(key=lambda s: (-s[0], -s[1]))
        return scored[0][2].query


POLICIES = {
    p.name: p
    for p in (
        MostUncertainPredicatePolicy(),
        CheapestGroundingPolicy(),
        PlanDisambiguationPolicy(),
    )
}
