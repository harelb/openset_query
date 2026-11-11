"""EvidenceStore integration tests against a real Neo4j.

Gated on NEO4J_URI / NEO4J_AUTH (same convention as the heracles DB tests) —
point them at a THROWAWAY database; every test wipes it. Example:

    docker run -d --rm --name planability_test_neo4j -p 7799:7687 \
        -e NEO4J_AUTH=neo4j/testpass neo4j:5.18
    NEO4J_URI=bolt://localhost:7799 NEO4J_AUTH=neo4j/testpass pytest ...

Acceptance: ingest -> wipe (initialize_db) -> re-ingest + JSONL re-import
reproduces identical assertion statuses.
"""

import os

import pytest

_NEO4J_ENV_URI = os.environ.get("NEO4J_URI", "")
if not _NEO4J_ENV_URI:
    pytest.skip(
        "Skipping: NEO4J_URI env var is not set (throwaway DB required).",
        allow_module_level=True,
    )

if not hasattr(
    pytest.importorskip("heracles.graph_interface"), "insert_trajectory_frames_to_db"
):
    pytest.skip(
        "Skipping: this heracles has no TrajectoryFrame support (optional feature).",
        allow_module_level=True,
    )

from heracles.graph_interface import (  # noqa: E402
    initialize_db,
    insert_edges,
    insert_observations_to_db,
    insert_trajectory_frames_to_db,
)
from heracles.query_interface import Neo4jWrapper  # noqa: E402

from openset_query.evidence import (  # noqa: E402
    EpistemicStatus,
    EvidenceStore,
    GroundingQuery,
    ReasonCode,
    SearchEpisode,
)


@pytest.fixture(scope="module")
def db():
    user, password = os.environ.get("NEO4J_AUTH", "neo4j/neo4j").split("/", 1)
    wrapper = Neo4jWrapper(_NEO4J_ENV_URI, (user, password), atomic_queries=True)
    wrapper.connect()
    yield wrapper
    wrapper.close()


def _count(db, cypher, **kw):
    records, _, _ = db.execute(cypher, **kw)
    return records[0]["n"]


def _frame(sym, ts):
    return {
        "nodeSymbol": sym,
        "timestamp_ns": ts,
        "path": f"/data/{sym}.jpg",
        "pos_x": 0.0,
        "pos_y": 0.0,
        "pos_z": 0.0,
        "pose_qw": 1.0,
        "pose_qx": 0.0,
        "pose_qy": 0.0,
        "pose_qz": 0.0,
        "place_id": None,
        "embedding": None,
        "embedding_dim": None,
        "embedding_model": None,
    }


def _ingest(db):
    """One wipe + minimal scene: 3 frames, one backpack with 2 scored views."""
    initialize_db(db)
    insert_trajectory_frames_to_db(
        db, [_frame("F(1)", 100), _frame("F(2)", 200), _frame("F(3)", 300)]
    )
    db.execute(
        "MERGE (o:Object {nodeSymbol: 'O(1)'}) "
        "SET o.class = 'backpack', o.name = 'backpack'"
    )
    insert_observations_to_db(
        db,
        [
            {
                "nodeSymbol": "O(1)_100",
                "timestamp_ns": 100,
                "mask_file": "m1.png",
                "bbox_2d_min_x": 0,
                "bbox_2d_min_y": 0,
                "bbox_2d_max_x": 5,
                "bbox_2d_max_y": 5,
                "score": 0.9,
                "detector": "sam3",
                "mechanism": "sam3_query",
            },
            {
                "nodeSymbol": "O(1)_200",
                "timestamp_ns": 200,
                "mask_file": "m2.png",
                "bbox_2d_min_x": 0,
                "bbox_2d_min_y": 0,
                "bbox_2d_max_x": 5,
                "bbox_2d_max_y": 5,
                "score": 0.8,
                "detector": "sam3",
                "mechanism": "sam3_query",
            },
        ],
    )
    insert_edges(
        db,
        "HAS_OBSERVATION",
        "Object",
        "Observation",
        [{"from": "O(1)", "to": "O(1)_100"}, {"from": "O(1)", "to": "O(1)_200"}],
    )


Q_PRESENT = GroundingQuery("exists", ("backpack",), required_confidence=0.5)
Q_ABSENT = GroundingQuery("exists", ("water fountain",))
Q_THIN = GroundingQuery("exists", ("broom",))


def _populate_epistemic_layer(store):
    """Episodes + assertions covering all three status outcomes."""
    ep_full = SearchEpisode.create(
        "exists",
        ("water fountain",),
        100,
        300,
        coverage=1.0,
        mechanism="reprocess_memory",
    )
    n_cov = store.add_episode(ep_full, covered_by_time=True)
    assert n_cov == 3  # the three TrajectoryFrames
    ep_thin = SearchEpisode.create(
        "exists", ("broom",), 100, 200, coverage=0.4, mechanism="reprocess_memory"
    )
    store.add_episode(ep_thin, covered_frames=["F(1)", "F(2)"])

    return {
        "present": store.compute_assertion(Q_PRESENT, now_ns=1_000),
        "absent": store.compute_assertion(Q_ABSENT, now_ns=1_000),
        "thin": store.compute_assertion(Q_THIN, now_ns=1_000),
    }


def test_wipe_survival_roundtrip(db, tmp_path):
    _ingest(db)
    store = EvidenceStore(db, session_id="s1", policy_version="p1")
    store.ensure_schema()
    first = _populate_epistemic_layer(store)

    # Sanity on the three-way spread + graph shape before the wipe.
    assert first["present"].status is EpistemicStatus.SUPPORTED
    assert first["present"].belief == 0.9
    assert first["absent"].status is EpistemicStatus.CONTRADICTED
    assert first["thin"].status is EpistemicStatus.UNKNOWN
    assert first["thin"].reason is ReasonCode.INSUFFICIENT_COVERAGE
    assert (
        _count(
            db, "MATCH (:Observation)-[r:SUPPORTS]->(:Assertion) RETURN count(r) AS n"
        )
        == 2
    )
    assert (
        _count(
            db,
            "MATCH (:SearchEpisode)-[r:CONTRADICTS]->(:Assertion) RETURN count(r) AS n",
        )
        == 2
    )  # absent (sufficient) + thin
    assert (
        _count(
            db,
            "MATCH (:SearchEpisode)-[r:COVERED]->(:TrajectoryFrame) "
            "RETURN count(r) AS n",
        )
        == 5
    )

    sidecar = tmp_path / "epistemic.jsonl"
    n_lines = store.export_jsonl(sidecar)
    assert n_lines == 5  # 2 episodes + 3 assertions

    # --- wipe + re-ingest ---------------------------------------------------
    _ingest(db)
    assert _count(db, "MATCH (n:SearchEpisode) RETURN count(n) AS n") == 0
    assert _count(db, "MATCH (n:Assertion) RETURN count(n) AS n") == 0

    store2 = EvidenceStore(db, session_id="s2", policy_version="p1")
    store2.ensure_schema()
    counts = store2.import_jsonl(sidecar)
    assert counts == {"search_episode": 2, "assertion": 3}

    # Imported assertions reproduce the pre-wipe verdicts verbatim.
    loaded = {a.assertion_id: a for a in store2.load_assertions()}
    for a in first.values():
        b = loaded[a.assertion_id]
        assert b.status is a.status
        assert b.reason is a.reason
        assert b.belief == a.belief
        assert set(b.supporting) == set(a.supporting)
        assert set(b.contradicting) == set(a.contradicting)

    # COVERED and evidence edges re-attached to the re-ingested frames/obs.
    assert (
        _count(
            db,
            "MATCH (:SearchEpisode)-[r:COVERED]->(:TrajectoryFrame) "
            "RETURN count(r) AS n",
        )
        == 5
    )
    assert (
        _count(
            db, "MATCH (:Observation)-[r:SUPPORTS]->(:Assertion) RETURN count(r) AS n"
        )
        == 2
    )
    assert (
        _count(
            db,
            "MATCH (:SearchEpisode)-[r:CONTRADICTS]->(:Assertion) RETURN count(r) AS n",
        )
        == 2
    )

    # RECOMPUTING from the re-ingested evidence reproduces identical statuses
    # and identities (assertions are recomputable materializations).
    for q, key in ((Q_PRESENT, "present"), (Q_ABSENT, "absent"), (Q_THIN, "thin")):
        again = store2.compute_assertion(q, now_ns=1_000)
        assert again.assertion_id == first[key].assertion_id
        assert again.status is first[key].status
        assert again.reason is first[key].reason
        assert again.belief == first[key].belief
        assert set(again.supporting) == set(first[key].supporting)

    # Export from the rebuilt DB is byte-identical to the original sidecar
    # (deterministic ordering end to end).
    sidecar2 = tmp_path / "epistemic2.jsonl"
    store2.export_jsonl(sidecar2)
    assert sidecar2.read_text() == sidecar.read_text()


def test_recompute_updates_assertion_in_place(db):
    _ingest(db)
    store = EvidenceStore(db, session_id="s1")
    store.ensure_schema()
    a1 = store.compute_assertion(Q_THIN, now_ns=10)
    assert a1.status is EpistemicStatus.UNKNOWN
    assert a1.reason is ReasonCode.NO_EVIDENCE

    # New evidence arrives: a completed sufficient search flips the verdict —
    # same assertion node, refreshed edges, no stale SUPPORTS/CONTRADICTS.
    ep = SearchEpisode.create(
        "exists", ("broom",), 100, 300, coverage=1.0, mechanism="reprocess_memory"
    )
    store.add_episode(ep, covered_by_time=True)
    a2 = store.compute_assertion(Q_THIN, now_ns=20)
    assert a2.assertion_id == a1.assertion_id
    assert a2.status is EpistemicStatus.CONTRADICTED

    assert _count(db, "MATCH (n:Assertion) RETURN count(n) AS n") == 1
    records, _, _ = db.execute(
        "MATCH (n:Assertion) RETURN n.status AS s, n.computed_at_ns AS t"
    )
    assert records[0]["s"] == "contradicted"
    assert records[0]["t"] == 20
    assert (
        _count(db, "MATCH ()-[r:CONTRADICTS]->(:Assertion) RETURN count(r) AS n") == 1
    )


def test_observation_records_read_scores_from_graph(db):
    _ingest(db)
    store = EvidenceStore(db)
    recs = store.observation_records(GroundingQuery("exists", ("backpack",)))
    assert [r.evidence_id for r in recs] == ["O(1)_100", "O(1)_200"]
    assert [r.confidence for r in recs] == [0.9, 0.8]
    assert all(r.provenance["object_symbol"] == "O(1)" for r in recs)
    assert store.observation_records(GroundingQuery("exists", ("zebra",))) == []


def test_episode_provenance_roundtrips_through_graph(db):
    _ingest(db)
    store = EvidenceStore(db)
    store.ensure_schema()
    ep = SearchEpisode.create(
        "exists",
        ("mop",),
        100,
        200,
        coverage=0.9,
        mechanism="frontier_search",
        source="frontier_search",
        provenance={"requery_enabled": False, "grid_cells": 40},
    )
    store.add_episode(ep)
    (back,) = store.episodes_for(GroundingQuery("exists", ("mop",)))
    assert back == ep
    # Dead-channel negative -> UNKNOWN / PERCEPTION_UNAVAILABLE (invariant
    # survives the DB round trip).
    a = store.compute_assertion(
        GroundingQuery("exists", ("mop",)), now_ns=999, persist=False
    )
    assert a.status is EpistemicStatus.UNKNOWN
    assert a.reason is ReasonCode.PERCEPTION_UNAVAILABLE
