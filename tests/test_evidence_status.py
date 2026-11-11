"""compute_status truth table: the UNKNOWN != CONTRADICTED contract.

Each test is one row of the semantics the research abstraction promises.
If one of these rows changes, the meaning of every grounding.jsonl changes
with it -- treat edits here as breaking changes.
"""

import pytest

from openset_query.evidence.status import (
    DEFAULT_COVERAGE_SUFFICIENCY,
    compute_status,
)
from openset_query.evidence.temporal import FixedDecay, NoDecay
from openset_query.evidence.types import (
    EpistemicStatus,
    EvidenceRecord,
    EvidenceSource,
    GroundingQuery,
    Polarity,
    ReasonCode,
)

NOW_NS = 2_000_000_000_000_000_000
HOUR_NS = 3_600 * 10**9

QUERY = GroundingQuery(predicate="exists", args=("suitcase",))


def positive(eid="pos1", conf=0.84, t_collected=NOW_NS - HOUR_NS, **prov):
    return EvidenceRecord(
        evidence_id=eid,
        source=EvidenceSource.PRIOR_ARCHIVE,
        polarity=Polarity.POSITIVE,
        predicate="exists",
        args=("suitcase",),
        t_recorded_ns=NOW_NS,
        confidence=conf,
        t_collected_ns=t_collected,
        provenance=prov or {"score": conf, "n_frames": 8},
    )


def negative(eid="neg1", coverage=0.97, requery_enabled=True):
    return EvidenceRecord(
        evidence_id=eid,
        source=EvidenceSource.CURRENT_ARCHIVE,
        polarity=Polarity.NEGATIVE,
        predicate="exists",
        args=("suitcase",),
        t_recorded_ns=NOW_NS,
        coverage=coverage,
        provenance={"requery_enabled": requery_enabled, "scan_all": True},
    )


class TestNoEvidence:
    def test_no_evidence_is_unknown(self):
        res = compute_status(QUERY, [])
        assert res.status is EpistemicStatus.UNKNOWN
        assert res.reason is ReasonCode.NO_EVIDENCE
        assert res.belief is None


class TestPositiveEvidence:
    def test_admissible_positive_supports(self):
        res = compute_status(QUERY, [positive()], now_ns=NOW_NS)
        assert res.status is EpistemicStatus.SUPPORTED
        assert res.reason is ReasonCode.ADMISSIBLE_POSITIVE_EVIDENCE
        assert res.belief == pytest.approx(0.84)
        assert res.supporting == ("pos1",)

    def test_structural_positive_confidence_none_supports(self):
        res = compute_status(QUERY, [positive(conf=None)], now_ns=NOW_NS)
        assert res.status is EpistemicStatus.SUPPORTED
        assert res.belief == pytest.approx(1.0)

    def test_under_required_confidence_is_unknown_not_contradicted(self):
        strict = GroundingQuery(
            predicate="exists", args=("suitcase",), required_confidence=0.95
        )
        res = compute_status(strict, [positive(conf=0.84)], now_ns=NOW_NS)
        assert res.status is EpistemicStatus.UNKNOWN
        assert res.reason is ReasonCode.BELOW_REQUIRED_CONFIDENCE
        assert res.belief == pytest.approx(0.84)

    def test_positive_beats_negative_regardless_of_order(self):
        for records in ([positive(), negative()], [negative(), positive()]):
            res = compute_status(QUERY, records, now_ns=NOW_NS)
            assert res.status is EpistemicStatus.SUPPORTED


class TestTemporalValidity:
    def test_stale_positive_downgrades_supported_to_unknown(self):
        # 10 half-lives old: weight ~0.001, far below the 0.5 floor.
        old = positive(t_collected=NOW_NS - 10 * HOUR_NS)
        fresh_model = NoDecay()
        decay_model = FixedDecay(half_life_s=3600.0)
        assert (
            compute_status(QUERY, [old], temporal=fresh_model, now_ns=NOW_NS).status
            is EpistemicStatus.SUPPORTED
        )
        res = compute_status(QUERY, [old], temporal=decay_model, now_ns=NOW_NS)
        assert res.status is EpistemicStatus.UNKNOWN
        assert res.reason is ReasonCode.STALE_HISTORICAL_EVIDENCE
        assert res.status is not EpistemicStatus.CONTRADICTED
        assert 0.0 < res.belief < 0.01  # decayed, not erased

    def test_fresh_record_survives_decay_model(self):
        recent = positive(t_collected=NOW_NS - 60 * 10**9)  # 1 min old
        res = compute_status(
            QUERY, [recent], temporal=FixedDecay(half_life_s=3600.0), now_ns=NOW_NS
        )
        assert res.status is EpistemicStatus.SUPPORTED

    def test_unknown_age_fails_closed_under_decay(self):
        # No t_collected: age falls back to t_recorded, which is old here.
        rec = EvidenceRecord(
            evidence_id="undated",
            source=EvidenceSource.PRIOR_ARCHIVE,
            polarity=Polarity.POSITIVE,
            predicate="exists",
            args=("suitcase",),
            t_recorded_ns=NOW_NS - 10 * HOUR_NS,
            confidence=0.9,
        )
        res = compute_status(
            QUERY, [rec], temporal=FixedDecay(half_life_s=3600.0), now_ns=NOW_NS
        )
        assert res.status is EpistemicStatus.UNKNOWN
        assert res.reason is ReasonCode.STALE_HISTORICAL_EVIDENCE

    def test_mixed_fresh_and_stale_uses_fresh(self):
        old = positive(eid="old", t_collected=NOW_NS - 100 * HOUR_NS)
        new = positive(eid="new", conf=0.7, t_collected=NOW_NS - 60 * 10**9)
        res = compute_status(
            QUERY, [old, new], temporal=FixedDecay(half_life_s=3600.0), now_ns=NOW_NS
        )
        assert res.status is EpistemicStatus.SUPPORTED
        assert "new" in res.supporting
        assert "old" not in res.supporting


class TestNegativeEvidence:
    def test_sufficient_search_contradicts(self):
        res = compute_status(QUERY, [negative(coverage=0.97)])
        assert res.status is EpistemicStatus.CONTRADICTED
        assert res.reason is ReasonCode.UNSUPPORTED_AFTER_SUFFICIENT_SEARCH
        assert res.belief == 0.0
        assert res.contradicting == ("neg1",)

    def test_thin_coverage_is_unknown(self):
        res = compute_status(QUERY, [negative(coverage=0.40)])
        assert res.status is EpistemicStatus.UNKNOWN
        assert res.reason is ReasonCode.INSUFFICIENT_COVERAGE

    def test_unknown_coverage_can_never_contradict(self):
        res = compute_status(QUERY, [negative(coverage=None)])
        assert res.status is EpistemicStatus.UNKNOWN
        assert res.reason is ReasonCode.INSUFFICIENT_COVERAGE

    def test_dead_channel_is_unknown_not_contradicted(self):
        # The worker died: legacy refuses, the honest label is UNKNOWN.
        res = compute_status(QUERY, [negative(coverage=0.97, requery_enabled=False)])
        assert res.status is EpistemicStatus.UNKNOWN
        assert res.reason is ReasonCode.PERCEPTION_UNAVAILABLE

    def test_blind_channel_is_unknown_not_contradicted(self):
        # Novel class vs closed labelspace: structurally incapable channel.
        res = compute_status(QUERY, [negative(coverage=0.97)], channel_capable=False)
        assert res.status is EpistemicStatus.UNKNOWN
        assert res.reason is ReasonCode.PERCEPTION_UNAVAILABLE

    def test_coverage_bar_is_inclusive_at_sufficiency(self):
        at_bar = negative(coverage=DEFAULT_COVERAGE_SUFFICIENCY)
        assert compute_status(QUERY, [at_bar]).status is (EpistemicStatus.CONTRADICTED)

    def test_alive_channel_outweighs_dead_one(self):
        dead = negative(eid="dead", coverage=0.97, requery_enabled=False)
        alive = negative(eid="alive", coverage=0.97, requery_enabled=True)
        res = compute_status(QUERY, [dead, alive])
        assert res.status is EpistemicStatus.CONTRADICTED
        assert res.contradicting == ("alive",)


class TestCoverageDefaultDriftGuard:
    def test_pinned_value(self):
        """The vendored core must keep the value the dcist_sim original
        mirrors from explore_mission's --coverage-limit default (0.95).
        The original's drift guard imports the mission parser; here we pin
        the value instead — dcist_sim is not a dependency of this repo."""
        assert DEFAULT_COVERAGE_SUFFICIENCY == 0.95

    def test_matches_vendor_source_when_available(self):
        """If the dcist_sim original is importable (dev checkouts), the two
        copies must agree — the vendor contract is semantic lockstep."""
        try:
            from dcist_sim_isaac.grounding import status as origin
        except ImportError:
            pytest.skip("dcist_sim_isaac not on path; value pin covers this")
        assert DEFAULT_COVERAGE_SUFFICIENCY == origin.DEFAULT_COVERAGE_SUFFICIENCY
