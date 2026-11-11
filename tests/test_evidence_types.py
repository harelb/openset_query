"""Grounding semantic types: stability, round-trips, immutability."""

import dataclasses
import json

import pytest

from openset_query.evidence.types import (
    DISCOVERY_SOURCE_MAP,
    EpistemicStatus,
    EvidenceRecord,
    EvidenceSource,
    GroundingQuery,
    GroundingResult,
    IdentityHypothesis,
    Polarity,
    ReasonCode,
    dumps_record,
)


def _record(**overrides):
    base = dict(
        evidence_id="ev1",
        source=EvidenceSource.PRIOR_ARCHIVE,
        polarity=Polarity.POSITIVE,
        predicate="exists",
        args=("suitcase",),
        t_recorded_ns=1_700_000_000_000_000_000,
        confidence=0.84,
        t_collected_ns=1_699_999_000_000_000_000,
        session_id="/path/to/archive",
        provenance={"score": 0.84, "n_frames": 8, "n_pixels": 13027},
    )
    base.update(overrides)
    return EvidenceRecord(**base)


class TestEnumStability:
    """Serialized values are an interface (grounding.jsonl outlives code)."""

    def test_status_values(self):
        assert {s.value for s in EpistemicStatus} == {
            "supported",
            "contradicted",
            "unknown",
        }

    def test_reason_codes_pinned(self):
        assert {r.value for r in ReasonCode} == {
            "admissible_positive_evidence",
            "unsupported_after_sufficient_search",
            "insufficient_coverage",
            "perception_unavailable",
            "stale_historical_evidence",
            "below_required_confidence",
            "no_grounding_operator",
            "no_evidence",
            "no_calibration",
        }

    def test_discovery_source_map_matches_mission_vocabulary(self):
        # explore_mission's `discovered.source` values, verbatim.
        assert set(DISCOVERY_SOURCE_MAP) == {"dsg", "requery", "archive_requery"}


class TestRoundTrips:
    def test_query_round_trip(self):
        q = GroundingQuery(
            predicate="exists",
            args=["suitcase"],
            required_confidence=0.3,
            freshness_s=60.0,
        )
        q2 = GroundingQuery.from_json(json.loads(json.dumps(q.to_json())))
        assert q2 == q
        assert q2.args == ("suitcase",)  # list input normalized to tuple

    def test_query_defaults_omitted_from_json(self):
        d = GroundingQuery(predicate="exists", args=("x",)).to_json()
        assert "required_confidence" not in d
        assert "freshness_s" not in d

    def test_record_round_trip(self):
        rec = _record()
        rec2 = EvidenceRecord.from_json(json.loads(dumps_record(rec)))
        assert rec2 == rec

    def test_record_optional_fields_default_none(self):
        rec = EvidenceRecord(
            evidence_id="e",
            source=EvidenceSource.COMMITTED_DSG,
            polarity=Polarity.NEGATIVE,
            predicate="exists",
            args=("x",),
            t_recorded_ns=1,
        )
        assert rec.confidence is None
        assert rec.coverage is None
        assert rec.session_id is None
        assert rec.provenance == {}
        assert EvidenceRecord.from_json(rec.to_json()) == rec

    def test_effective_confidence_none_reads_one(self):
        assert _record(confidence=None).effective_confidence == 1.0
        assert _record(confidence=0.3).effective_confidence == 0.3

    def test_result_to_json(self):
        q = GroundingQuery(predicate="exists", args=("x",))
        res = GroundingResult(
            query=q,
            status=EpistemicStatus.SUPPORTED,
            reason=ReasonCode.ADMISSIBLE_POSITIVE_EVIDENCE,
            belief=0.9,
            supporting=("ev1",),
            hypotheses=(IdentityHypothesis("h1", "x", (1.0, 2.0), ("ev1",)),),
        )
        d = json.loads(dumps_record(res))
        assert d["status"] == "supported"
        assert d["reason"] == "admissible_positive_evidence"
        assert d["hypotheses"][0]["xy"] == [1.0, 2.0]


class TestImmutability:
    def test_query_frozen(self):
        q = GroundingQuery(predicate="exists", args=("x",))
        with pytest.raises(dataclasses.FrozenInstanceError):
            q.predicate = "other"

    def test_record_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            _record().confidence = 0.0

    def test_query_key(self):
        q = GroundingQuery(predicate="object-in-region", args=("suitcase", "lobby"))
        assert q.key() == "object-in-region(suitcase,lobby)"
