"""Assertion + EvidenceObligation: identity, JSON round-trip, certification
semantics, obligation lifecycle."""

import pytest

from openset_query.evidence import (
    Assertion,
    EpistemicStatus,
    EvidenceObligation,
    GroundingQuery,
    GroundingResult,
    ObligationStatus,
    ReasonCode,
)

NS = 1_000_000_000  # 1 s in ns


def _q(**kw):
    defaults = dict(predicate="exists", args=("suitcase",))
    defaults.update(kw)
    return GroundingQuery(**defaults)


def _result(
    status=EpistemicStatus.SUPPORTED,
    reason=ReasonCode.ADMISSIBLE_POSITIVE_EVIDENCE,
    belief=0.9,
    q=None,
):
    return GroundingResult(
        query=q or _q(),
        status=status,
        reason=reason,
        belief=belief,
        supporting=("ev1", "ev2"),
    )


def test_from_result_and_json_roundtrip():
    a = Assertion.from_result(
        _result(),
        computed_at_ns=5 * NS,
        policy_version="v1",
        scope={"region": "kitchen"},
    )
    assert a.assertion_id.startswith("assert_")
    assert a.supporting == ("ev1", "ev2")
    assert a.policy_version == "v1"

    b = Assertion.from_json(a.to_json())
    assert b == a


def test_certifies_supported_fresh_and_confident():
    q = _q(required_confidence=0.8, freshness_s=10.0)
    a = Assertion.from_result(_result(belief=0.9, q=q), computed_at_ns=0)
    assert a.freshness_s == 10.0
    assert a.certifies(now_ns=5 * NS)  # fresh, confident
    assert not a.certifies(now_ns=11 * NS)  # stale verdict
    low = Assertion.from_result(_result(belief=0.5, q=q), computed_at_ns=0)
    assert not low.certifies(now_ns=5 * NS)  # under-confident


def test_certifies_never_for_non_supported():
    for status, reason in [
        (EpistemicStatus.CONTRADICTED, ReasonCode.UNSUPPORTED_AFTER_SUFFICIENT_SEARCH),
        (EpistemicStatus.UNKNOWN, ReasonCode.NO_EVIDENCE),
    ]:
        a = Assertion.from_result(
            _result(status=status, reason=reason), computed_at_ns=0
        )
        assert not a.certifies(now_ns=0)


def test_certifies_ungraded_channel_belief_none():
    # None belief = structural fact (no graded signal), certifiable when fresh.
    a = Assertion.from_result(_result(belief=None), computed_at_ns=0)
    assert a.certifies(now_ns=0)


def test_no_freshness_horizon_is_always_fresh():
    a = Assertion.from_result(_result(), computed_at_ns=0)
    assert a.is_fresh(now_ns=10**18)


def test_obligation_lifecycle_and_roundtrip():
    o = EvidenceObligation.create(
        _q(),
        mechanism="reprocess_memory",
        created_at_ns=1 * NS,
        assertion_id="assert_abc",
        estimated_cost={"time_s": 3.0},
    )
    assert o.status is ObligationStatus.OPEN
    assert o.obligation_id.startswith("oblig_")

    r = o.resolved(resolved_at_ns=2 * NS)
    assert r.status is ObligationStatus.RESOLVED
    assert r.resolved_at_ns == 2 * NS
    assert o.status is ObligationStatus.OPEN  # frozen original untouched

    f = o.failed(resolved_at_ns=3 * NS)
    assert f.status is ObligationStatus.FAILED

    back = EvidenceObligation.from_json(r.to_json())
    assert back == r


def test_assertion_is_frozen():
    a = Assertion.from_result(_result(), computed_at_ns=0)
    with pytest.raises(Exception):
        a.status = EpistemicStatus.UNKNOWN
