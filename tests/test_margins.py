"""Pure tests for the relative-margin evidence terms (evidence/margins.py).

Pins the definitional property the spec's required test names: the margin is
target-vs-CONTROL per crop, so an object visible in MANY true frames can never
be rejected by a top-1-vs-top-2-frame comparison — more true frames only add
evidence.
"""
import math

import pytest

from openset_query.evidence.margins import (
    ObservationEvidence,
    best_margin,
    distinct_view_count,
    view_spread_m,
)


def _ev(obs_id="obs1", ts=1, sam3=0.9, target=0.30, controls=None, depth=500,
        xyz=(1.0, 2.0, 0.0), prompt="a mug"):
    return ObservationEvidence(
        obs_id=obs_id, timestamp_ns=ts, sam3_score=sam3, target_score=target,
        control_scores={"a wall": 0.10, "a floor": 0.12} if controls is None
        else controls,
        depth_valid_px=depth, view_xyz=xyz, target_prompt=prompt)


# ---------------------------------------------------------------------------
# Margin semantics
# ---------------------------------------------------------------------------


def test_margin_is_target_minus_best_control():
    e = _ev(target=0.30, controls={"a wall": 0.10, "a floor": 0.22})
    assert e.control_max == pytest.approx(0.22)
    assert e.margin == pytest.approx(0.08)


def test_margin_none_when_unscored():
    assert _ev(target=None).margin is None
    assert _ev(controls={}).margin is None
    assert _ev(controls={}).control_max is None


def test_margin_never_compares_frames_of_the_same_object():
    """Spec-required pin: N views of a true object each keep their OWN
    target-vs-control margin — adding more true frames cannot drag any view's
    margin down (no top-1-vs-top-2-frame arithmetic exists to reject with)."""
    one_view = [_ev(obs_id="a", ts=1, target=0.30)]
    many_views = [_ev(obs_id=str(i), ts=i, target=0.30) for i in range(1, 8)]
    assert best_margin(one_view) == pytest.approx(best_margin(many_views))
    assert all(e.margin == pytest.approx(one_view[0].margin)
               for e in many_views)


# ---------------------------------------------------------------------------
# Multi-view + geometry
# ---------------------------------------------------------------------------


def test_distinct_view_count_collapses_cotimestamped_detections():
    assert distinct_view_count([_ev(ts=5), _ev(ts=5), _ev(ts=9)]) == 2


def test_distinct_view_count_falls_back_to_record_count():
    assert distinct_view_count([_ev(ts=None), _ev(ts=None)]) == 2


def test_view_spread_measures_distance_from_mean():
    evs = [_ev(ts=1, xyz=(0.0, 0.0, 0.0)), _ev(ts=2, xyz=(2.0, 0.0, 0.0))]
    assert view_spread_m(evs) == pytest.approx(1.0)


def test_view_spread_none_for_fewer_than_two_located_views():
    assert view_spread_m([_ev(xyz=(1.0, 1.0, 0.0))]) is None
    assert view_spread_m([_ev(xyz=None), _ev(xyz=None)]) is None


# ---------------------------------------------------------------------------
# Serialization: raw terms round-trip
# ---------------------------------------------------------------------------


def test_json_round_trip_preserves_raw_terms_and_derives_margin():
    e = _ev(target=0.31, controls={"a wall": 0.07})
    d = e.to_json()
    assert d["control_scores"] == {"a wall": 0.07}     # raw terms present
    assert d["margin"] == pytest.approx(0.24)          # derived, for readers
    back = ObservationEvidence.from_json(d)
    assert back == e
    assert back.margin == pytest.approx(e.margin)
