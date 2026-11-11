"""Temporal validity of evidence (pure, stdlib-only).

WHERE DECAY LIVES. Deliberately at STATUS COMPUTATION, as a weight -- not in
admissibility (``accept_detections`` has no timestamp input and must stay
byte-identical for legacy parity) and not on the record (records are archive
facts; different decay models must be replayable over the same serialized
evidence, which makes decay models directly comparable).

Decay never CONTRADICTS. An expired positive downgrades SUPPORTED ->
UNKNOWN(stale); historical support fading is not disproof.

These hand-coded values are an experimental scaffold, not a persistence
model. A predicate-family model (existence decays slowly, spatial relations
faster, transient state current-only) is the intended follow-up, but with
exactly one timestamped-change dataset banked (none), fitting one now would
be over-engineering -- the interface is the deliverable.


Vendored from dcist_sim_isaac.grounding (dcist_sim @ 7e8bf5a) for the
planability-assessment work; keep semantics in lockstep with the original.
Design invariants (do not break): decay applied only at compute_status;
negative evidence = completed search with coverage; coverage=None never
contradicts.
"""

from __future__ import annotations


class TemporalValidityModel:
    """Interface: weight in [0, 1] plus a staleness verdict."""

    def weight(self, record, now_ns) -> float:
        raise NotImplementedError

    def is_stale(self, record, now_ns) -> bool:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"model": type(self).__name__}


def record_age_s(record, now_ns) -> float | None:
    """Age of the evidence in seconds, or None when it cannot be computed.

    Prefers ``t_collected_ns`` (when the observation happened -- for archive
    hits, the keyframe timestamps) over ``t_recorded_ns`` (when we wrote the
    record). A negative age (clock skew, replay) clamps to 0.
    """
    if now_ns is None:
        return None
    t = record.t_collected_ns
    if t is None:
        t = record.t_recorded_ns
    if t is None:
        return None
    return max(0.0, (int(now_ns) - int(t)) / 1e9)


class NoDecay(TemporalValidityModel):
    """Legacy semantics: a prior-mission archive hit counts as current truth.

    This is the model the existing tier ladder implicitly runs, which is
    exactly why it must exist by name -- 'treat-history-as-current' is an
    experimental arm, not an unexamined default.
    """

    def weight(self, record, now_ns) -> float:
        return 1.0

    def is_stale(self, record, now_ns) -> bool:
        return False


class FixedDecay(TemporalValidityModel):
    """Exponential half-life decay with a hard staleness floor.

    weight = 0.5 ** (age_s / half_life_s); stale once weight < floor.
    Unknown age fails CLOSED for historical sources: a record whose age
    cannot be computed is treated as stale rather than fresh (missing
    provenance must not launder old evidence into current truth).
    """

    def __init__(self, half_life_s: float, floor: float = 0.5):
        if half_life_s <= 0:
            raise ValueError("half_life_s must be positive")
        if not 0.0 < floor <= 1.0:
            raise ValueError("floor must be in (0, 1]")
        self.half_life_s = float(half_life_s)
        self.floor = float(floor)

    def weight(self, record, now_ns) -> float:
        age = record_age_s(record, now_ns)
        if age is None:
            return 0.0
        return 0.5 ** (age / self.half_life_s)

    def is_stale(self, record, now_ns) -> bool:
        return self.weight(record, now_ns) < self.floor

    def describe(self) -> dict:
        return {
            "model": "FixedDecay",
            "half_life_s": self.half_life_s,
            "floor": self.floor,
        }
