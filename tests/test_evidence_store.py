"""Unit tests for evidence.store (FakeDb — no Neo4j required).

The full ingest -> wipe -> re-ingest + JSONL re-import acceptance runs against
a real database in test_evidence_store_db.py (env-gated on NEO4J_URI).
"""
import json

import pytest

from openset_query.evidence import (
    Assertion,
    EpistemicStatus,
    EvidenceSource,
    EvidenceStore,
    GroundingQuery,
    Polarity,
    ReasonCode,
    SearchEpisode,
    assertion_id_for,
    compute_status,
)
from openset_query.evidence import store as store_mod


class FakeDb:
    """Duck-typed Neo4jWrapper: records every execute; canned responses by
    (substring, response) rules, first match wins."""

    def __init__(self):
        self.executed = []
        self.rules = []

    def respond(self, substring, records):
        self.rules.append((substring, records))

    def execute(self, cypher, parameters=None, **kw):
        self.executed.append((cypher, kw or parameters or {}))
        for substring, records in self.rules:
            if substring in cypher:
                return (records, None, None)
        return ([], None, None)

    def calls_containing(self, substring):
        return [(c, p) for c, p in self.executed if substring in c]


# ---------------------------------------------------------------------------
# SearchEpisode
# ---------------------------------------------------------------------------


def test_episode_to_record_is_negative_with_coverage_passthrough():
    ep = SearchEpisode.create(
        "exists", ("water fountain",), 100, 300, coverage=0.97,
        mechanism="reprocess_memory", session_id="s1",
        provenance={"requery_enabled": False},
    )
    rec = ep.to_record()
    assert rec.polarity is Polarity.NEGATIVE
    assert rec.coverage == 0.97
    assert rec.evidence_id == ep.episode_id
    assert rec.t_recorded_ns == 300
    assert rec.provenance["mechanism"] == "reprocess_memory"
    assert rec.provenance["t_start_ns"] == 100
    # channel-alive flag must pass through untouched (status.py reads it)
    assert rec.provenance["requery_enabled"] is False
    assert rec.source is EvidenceSource.CURRENT_ARCHIVE


def test_episode_json_roundtrip():
    ep = SearchEpisode.create(
        "exists", ("broom",), 1, 2, coverage=None,
        mechanism="frontier_search", source="frontier_search",
        provenance={"grid_cells": 40},
    )
    again = SearchEpisode.from_json(json.loads(json.dumps(ep.to_json())))
    assert again == ep


def test_coverage_none_episode_never_contradicts():
    # Store-level restatement of the core invariant: an unknown-coverage
    # search must leave the proposition UNKNOWN, not CONTRADICTED.
    q = GroundingQuery("exists", ("broom",))
    ep = SearchEpisode.create("exists", ("broom",), 0, 10, coverage=None)
    result = compute_status(q, [ep.to_record()])
    assert result.status is EpistemicStatus.UNKNOWN
    assert result.reason is ReasonCode.INSUFFICIENT_COVERAGE


# ---------------------------------------------------------------------------
# Deterministic assertion identity
# ---------------------------------------------------------------------------


def test_assertion_id_deterministic_and_query_sensitive():
    q1 = GroundingQuery("exists", ("backpack",), required_confidence=0.5)
    q2 = GroundingQuery("exists", ("backpack",), required_confidence=0.5)
    q3 = GroundingQuery("exists", ("backpack",), required_confidence=0.6)
    assert assertion_id_for(q1) == assertion_id_for(q2)
    assert assertion_id_for(q1) != assertion_id_for(q3)
    assert assertion_id_for(q1).startswith("assert_")


# ---------------------------------------------------------------------------
# EvidenceStore against FakeDb
# ---------------------------------------------------------------------------


def test_add_episode_writes_node_and_time_fanout():
    db = FakeDb()
    db.respond("MERGE (e)-[:COVERED]->(f)", [{"n": 2}])
    store = EvidenceStore(db, session_id="sess")
    ep = SearchEpisode.create("exists", ("mop",), 100, 200, coverage=1.0)
    n = store.add_episode(ep, covered_by_time=True)
    assert n == 6  # 2 per frame label x 3 labels (canned)

    node_calls = db.calls_containing("MERGE (e:SearchEpisode")
    assert len(node_calls) == 1
    _, params = node_calls[0]
    assert params["args_json"] == '["mop"]'
    assert params["session_id"] == "sess"  # store default fills missing

    fanout_calls = db.calls_containing("MERGE (e)-[:COVERED]->(f)")
    assert len(fanout_calls) == 3
    assert all("timestamp_ns >= $t0" in c for c, _ in fanout_calls)
    for label in store_mod.FRAME_LABELS:
        assert any(f"(f:{label})" in c for c, _ in fanout_calls)


def test_add_episode_explicit_frames_uses_symbol_match():
    db = FakeDb()
    store = EvidenceStore(db)
    ep = SearchEpisode.create("exists", ("mop",), 0, 1)
    store.add_episode(ep, covered_frames=["F(1)", "a0"])
    fanout_calls = db.calls_containing("MERGE (e)-[:COVERED]->(f)")
    assert len(fanout_calls) == 3
    assert all("f.nodeSymbol IN $syms" in c for c, _ in fanout_calls)
    assert all(p["syms"] == ["F(1)", "a0"] for _, p in fanout_calls)


def test_compute_assertion_supported_and_edges_refreshed():
    db = FakeDb()
    db.respond("HAS_OBSERVATION", [
        {"ns": "O(1)_100", "ts": 100, "score": 0.9, "detector": "sam3",
         "mechanism": "sam3_query", "obj": "O(1)"},
        {"ns": "O(1)_200", "ts": 200, "score": 0.8, "detector": "sam3",
         "mechanism": "sam3_query", "obj": "O(1)"},
    ])
    store = EvidenceStore(db, policy_version="p1")
    q = GroundingQuery("exists", ("backpack",))
    a = store.compute_assertion(q, now_ns=1_000)

    assert a.status is EpistemicStatus.SUPPORTED
    assert a.belief == 0.9
    assert set(a.supporting) == {"O(1)_100", "O(1)_200"}
    assert a.assertion_id == assertion_id_for(q)
    assert a.policy_version == "p1"

    # Persisted node + stale-edge refresh + evidence edges.
    assert len(db.calls_containing("MERGE (a:Assertion")) == 1
    assert len(db.calls_containing("DELETE r")) == 1
    supports = db.calls_containing("MERGE (ev)-[:SUPPORTS]->(a)")
    assert len(supports) == 2  # Observation-keyed + SearchEpisode-keyed match
    assert all(set(p["ids"]) == set(a.supporting) for _, p in supports)
    assert not db.calls_containing("MERGE (ev)-[:CONTRADICTS]->(a)")


def test_compute_assertion_contradicted_by_episode():
    db = FakeDb()
    db.respond("MATCH (e:SearchEpisode {predicate", [
        {"episode_id": "episode_abc", "t0": 0, "t1": 10, "coverage": 1.0,
         "mechanism": "reprocess_memory", "session_id": None,
         "source": "current_archive", "provenance_json": "{}"},
    ])
    store = EvidenceStore(db)
    q = GroundingQuery("exists", ("water fountain",))
    a = store.compute_assertion(q, now_ns=1_000)
    assert a.status is EpistemicStatus.CONTRADICTED
    assert a.contradicting == ("episode_abc",)
    contras = db.calls_containing("MERGE (ev)-[:CONTRADICTS]->(a)")
    assert len(contras) == 2
    assert all(p["ids"] == ["episode_abc"] for _, p in contras)


def test_compute_assertion_no_evidence_unknown_not_persisted_when_disabled():
    db = FakeDb()
    store = EvidenceStore(db)
    a = store.compute_assertion(GroundingQuery("exists", ("unicorn",)),
                                now_ns=5, persist=False)
    assert a.status is EpistemicStatus.UNKNOWN
    assert a.reason is ReasonCode.NO_EVIDENCE
    assert not db.calls_containing("MERGE (a:Assertion")


def test_certifies_composes_with_store_output():
    db = FakeDb()
    db.respond("HAS_OBSERVATION", [
        {"ns": "O(9)_5", "ts": 5, "score": 0.4, "detector": "sam3",
         "mechanism": "sam3_query", "obj": "O(9)"},
    ])
    store = EvidenceStore(db)
    weak = store.compute_assertion(
        GroundingQuery("exists", ("broom",), required_confidence=0.7),
        now_ns=10, persist=False)
    assert weak.status is EpistemicStatus.UNKNOWN
    assert weak.reason is ReasonCode.BELOW_REQUIRED_CONFIDENCE
    assert not weak.certifies(now_ns=10)


def test_import_jsonl_roundtrips_lines(tmp_path):
    db = FakeDb()
    store = EvidenceStore(db)
    ep = SearchEpisode.create("exists", ("mop",), 0, 1, coverage=1.0)
    q = GroundingQuery("exists", ("mop",))
    assertion = Assertion.from_result(
        compute_status(q, [ep.to_record()]),
        computed_at_ns=7, assertion_id=assertion_id_for(q))
    path = tmp_path / "sidecar.jsonl"
    with open(path, "w") as f:
        f.write(json.dumps({"kind": "search_episode",
                            "episode": ep.to_json(),
                            "covered": ["F(1)"]}) + "\n")
        f.write(json.dumps({"kind": "assertion",
                            "assertion": assertion.to_json()}) + "\n")

    counts = store.import_jsonl(path)
    assert counts == {"search_episode": 1, "assertion": 1}
    assert len(db.calls_containing("MERGE (e:SearchEpisode")) == 1
    assert len(db.calls_containing("MERGE (a:Assertion")) == 1
    # covered frames re-attached by symbol
    fanout = db.calls_containing("MERGE (e)-[:COVERED]->(f)")
    assert all(p["syms"] == ["F(1)"] for _, p in fanout)


# ---------------------------------------------------------------------------
# Drift guard: local label constants stay in lockstep with heracles.constants
# ---------------------------------------------------------------------------


def test_labels_match_heracles_constants():
    heracles_constants = pytest.importorskip("heracles.constants")
    assert store_mod.OBJECTS == heracles_constants.OBJECTS
    assert store_mod.OBSERVATIONS == heracles_constants.OBSERVATIONS
    assert store_mod.SEARCH_EPISODES == heracles_constants.SEARCH_EPISODES
    assert store_mod.ASSERTIONS == heracles_constants.ASSERTIONS
    assert store_mod.HAS_OBSERVATION == heracles_constants.HAS_OBSERVATION
    assert store_mod.COVERED == heracles_constants.COVERED
    assert store_mod.SUPPORTS == heracles_constants.SUPPORTS
    assert store_mod.CONTRADICTS == heracles_constants.CONTRADICTS
    assert set(store_mod.FRAME_LABELS) == {
        heracles_constants.AGENTS,
        heracles_constants.SUBKEYFRAMES,
        heracles_constants.TRAJECTORY_FRAMES,
    }
