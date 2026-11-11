"""Visibility materializer integration tests against a real Neo4j.

Gated on NEO4J_URI / NEO4J_AUTH pointing at a THROWAWAY database (see
test_evidence_store_db.py for the docker one-liner).
"""

import os

import pytest

_NEO4J_ENV_URI = os.environ.get("NEO4J_URI", "")
if not _NEO4J_ENV_URI:
    pytest.skip(
        "Skipping: NEO4J_URI env var is not set (throwaway DB required).",
        allow_module_level=True,
    )

from heracles.query_interface import Neo4jWrapper  # noqa: E402

from openset_query.evidence import EvidenceStore, SearchEpisode  # noqa: E402
from openset_query.evidence import visibility  # noqa: E402


@pytest.fixture(scope="module")
def db():
    user, password = os.environ.get("NEO4J_AUTH", "neo4j/neo4j").split("/", 1)
    wrapper = Neo4jWrapper(_NEO4J_ENV_URI, (user, password), atomic_queries=True)
    wrapper.connect()
    yield wrapper
    wrapper.close()


@pytest.fixture(autouse=True)
def scene(db):
    """Wipe + a hand-computable scene: identity calib (sensor == body),
    fx=fy=100, 640x480.

    F(1)/a0/s0 at the origin looking +z (identity quat), F(2) looking -z
    (180 deg about y). Objects: A at (0,0,2) seen by the +z frames at the
    image center; B at (0,0,-2) seen only by F(2); C at (0,0,50) beyond the
    10 m ceiling; D at (10,0,2) projecting to u=820, off image.
    """
    db.execute("MATCH (n) DETACH DELETE n")
    identity16 = [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    db.execute(
        "CREATE (c:CameraCalib {calib_id: 'calibtest', fx: 100.0, fy: 100.0, "
        "cx: 320.0, cy: 240.0, width: 640, height: 480, depth_scale: 0.001, "
        "body_T_sensor: $bts})",
        bts=identity16,
    )
    db.execute(
        "CREATE (:TrajectoryFrame {nodeSymbol: 'F(1)', timestamp_ns: 100, "
        "  pos_x: 0.0, pos_y: 0.0, pos_z: 0.0, "
        "  pose_qw: 1.0, pose_qx: 0.0, pose_qy: 0.0, pose_qz: 0.0}), "
        "(:TrajectoryFrame {nodeSymbol: 'F(2)', timestamp_ns: 200, "
        "  pos_x: 0.0, pos_y: 0.0, pos_z: 0.0, "
        "  pose_qw: 0.0, pose_qx: 0.0, pose_qy: 1.0, pose_qz: 0.0}), "
        "(:Agent {nodeSymbol: 'a0', timestamp_ns: 100, "
        "  pos_x: 0.0, pos_y: 0.0, pos_z: 0.0, "
        "  rot_w: 1.0, rot_x: 0.0, rot_y: 0.0, rot_z: 0.0}), "
        "(:SubKeyframe {nodeSymbol: 's0', timestamp_ns: 150, "
        "  center: point({x: 0.0, y: 0.0, z: 0.0}), "
        "  rot_w: 1.0, rot_x: 0.0, rot_y: 0.0, rot_z: 0.0})"
    )
    db.execute(
        "MATCH (c:CameraCalib {calib_id: 'calibtest'}) "
        "MATCH (f) WHERE f.nodeSymbol IN ['F(1)', 'F(2)', 'a0', 's0'] "
        "MERGE (f)-[:HAS_CALIB]->(c)"
    )
    db.execute(
        "CREATE (:Object {nodeSymbol: 'A', class: 'backpack', "
        "  pos_x: 0.0, pos_y: 0.0, pos_z: 2.0}), "
        "(:Object {nodeSymbol: 'B', class: 'mop', "
        "  pos_x: 0.0, pos_y: 0.0, pos_z: -2.0}), "
        "(:Object {nodeSymbol: 'C', class: 'chair', "
        "  pos_x: 0.0, pos_y: 0.0, pos_z: 50.0}), "
        "(:Object {nodeSymbol: 'D', class: 'broom', "
        "  pos_x: 10.0, pos_y: 0.0, pos_z: 2.0})"
    )
    yield


def _pairs(db):
    records, _, _ = db.execute(
        "MATCH (f)-[v:VIEWS]->(o:Object) "
        "RETURN f.nodeSymbol AS f, o.nodeSymbol AS o, v.u AS u, v.v AS v, "
        "v.depth_m AS d ORDER BY f, o"
    )
    return {(r["f"], r["o"]): (r["u"], r["v"], r["d"]) for r in records}


def test_materialize_views_writes_expected_edges(db):
    n = visibility.materialize_views(db, max_depth_m=10.0)
    pairs = _pairs(db)
    assert set(pairs) == {("F(1)", "A"), ("a0", "A"), ("s0", "A"), ("F(2)", "B")}
    assert n == 4
    for key in (("F(1)", "A"), ("a0", "A"), ("s0", "A"), ("F(2)", "B")):
        u, v, d = pairs[key]
        assert u == pytest.approx(320.0)
        assert v == pytest.approx(240.0)
        assert d == pytest.approx(2.0)

    # Idempotent: rerun MERGEs onto the same relationships.
    visibility.materialize_views(db, max_depth_m=10.0)
    records, _, _ = db.execute("MATCH ()-[r:VIEWS]->() RETURN count(r) AS n")
    assert records[0]["n"] == 4

    # Without the depth ceiling, the 50 m object becomes a view for +z frames.
    visibility.materialize_views(db)
    assert set(_pairs(db)) == {
        ("F(1)", "A"),
        ("a0", "A"),
        ("s0", "A"),
        ("F(2)", "B"),
        ("F(1)", "C"),
        ("a0", "C"),
        ("s0", "C"),
    }


def test_refresh_drops_stale_edges(db):
    visibility.materialize_views(db)  # includes the depth-unbounded C views
    n = visibility.materialize_views(db, max_depth_m=10.0, refresh=True)
    assert n == 4
    assert set(_pairs(db)) == {("F(1)", "A"), ("a0", "A"), ("s0", "A"), ("F(2)", "B")}


def test_episode_visual_coverage(db):
    visibility.materialize_views(db, max_depth_m=10.0)
    store = EvidenceStore(db, session_id="s1")
    store.ensure_schema()
    ep_one = SearchEpisode.create("exists", ("mop",), 100, 100, coverage=1.0)
    store.add_episode(ep_one, covered_frames=["F(1)"])
    ep_both = SearchEpisode.create("exists", ("mop",), 100, 200, coverage=1.0)
    store.add_episode(ep_both, covered_frames=["F(1)", "F(2)"])

    # F(1) views only A -> 1 of 4 positioned objects; F(1)+F(2) view A and B.
    assert visibility.episode_visual_coverage(db, ep_one.episode_id) == pytest.approx(
        0.25
    )
    assert visibility.episode_visual_coverage(db, ep_both.episode_id) == pytest.approx(
        0.5
    )

    records, _, _ = db.execute(
        "MATCH (e:SearchEpisode {episode_id: $eid}) "
        "RETURN e.visual_coverage AS vc, e.coverage AS c",
        eid=ep_one.episode_id,
    )
    assert records[0]["vc"] == pytest.approx(0.25)
    assert records[0]["c"] == 1.0  # producer-reported coverage untouched

    # Scoped denominator + no-objects guard.
    assert visibility.episode_visual_coverage(
        db, ep_one.episode_id, object_symbols=["A", "B"], write=False
    ) == pytest.approx(0.5)
    assert (
        visibility.episode_visual_coverage(
            db, ep_one.episode_id, object_symbols=["nonexistent"]
        )
        is None
    )
