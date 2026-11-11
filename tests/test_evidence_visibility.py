"""Unit tests for evidence.visibility (FakeDb — no Neo4j required).

The acceptance criterion is the parity property: the materialized VIEWS pairs must
match object_discovery._in_frustum decisions — i.e. exactly the view-set that
drives detect_frustum_aware's skip logic — over randomized poses.
"""
import numpy as np
import pytest

from openset_query.evidence import visibility
from openset_query.keyframe_store import Keyframe
from openset_query.object_discovery import _in_frustum
from openset_query.reprojection import CameraCalib


class FakeDb:
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

    def view_rows(self):
        rows = []
        for cypher, params in self.executed:
            if "MERGE (f)-[v:VIEWS]->(o)" in cypher:
                rows.extend(params["rows"])
        return rows


def _random_quat(rng):
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


_CALIB_KW = dict(fx=380.0, fy=380.0, cx=320.0, cy=240.0, width=640, height=480,
                 depth_scale=1e-3)
# Non-trivial body->sensor offset: 90 deg about x + a translation, so the test
# exercises the full world_T_body @ body_T_sensor chain.
_BODY_T_SENSOR = np.array([
    [1, 0, 0, 0.1],
    [0, 0, -1, 0.0],
    [0, 1, 0, 0.2],
    [0, 0, 0, 1.0],
])


def _fixture(n_frames=12, n_objects=25, seed=7):
    rng = np.random.default_rng(seed)
    frame_rows, keyframes = [], {}
    for i in range(n_frames):
        pos = rng.uniform(-5, 5, size=3)
        qx, qy, qz, qw = _random_quat(rng)
        sym = f"F({i})"
        frame_rows.append({
            "ns": sym, "x": pos[0], "y": pos[1], "z": pos[2],
            "qx": qx, "qy": qy, "qz": qz, "qw": qw,
            "fx": _CALIB_KW["fx"], "fy": _CALIB_KW["fy"],
            "cx": _CALIB_KW["cx"], "cy": _CALIB_KW["cy"],
            "width": _CALIB_KW["width"], "height": _CALIB_KW["height"],
            "bts": list(_BODY_T_SENSOR.reshape(-1)),
        })
        world_T_body = visibility._pose_from_quat(
            pos[0], pos[1], pos[2], qx, qy, qz, qw)
        keyframes[sym] = Keyframe(
            timestamp_ns=i, rgb_path="", depth_path="",
            world_T_body=world_T_body,
            calib=CameraCalib(body_T_sensor=_BODY_T_SENSOR, **_CALIB_KW))
    object_rows = [
        {"ns": f"O({j})", "x": x, "y": y, "z": z}
        for j, (x, y, z) in enumerate(rng.uniform(-8, 8, size=(n_objects, 3)))
    ]
    return frame_rows, object_rows, keyframes


def test_views_match_in_frustum_parity():
    frame_rows, object_rows, keyframes = _fixture()
    max_depth_m = 6.0
    db = FakeDb()
    db.respond("MATCH (o:Object) WHERE o.pos_x IS NOT NULL", object_rows)
    db.respond("MATCH (n:TrajectoryFrame)-[:HAS_CALIB]", frame_rows)

    n = visibility.materialize_views(db, max_depth_m=max_depth_m)
    rows = db.view_rows()
    assert n == len(rows)

    expected = set()
    for o in object_rows:
        centroid = np.array([o["x"], o["y"], o["z"]])
        for sym, kf in keyframes.items():
            if _in_frustum(centroid, kf, max_depth_m=max_depth_m):
                expected.add((sym, o["ns"]))
    got = {(r["f"], r["o"]) for r in rows}
    assert got == expected
    assert expected, "degenerate fixture: no view pairs at all"
    assert len(expected) < len(object_rows) * len(keyframes), \
        "degenerate fixture: everything views everything"

    # u/v/depth_m carry the actual projection, inside the padded bounds.
    objects_by_ns = {o["ns"]: o for o in object_rows}
    for r in rows:
        kf = keyframes[r["f"]]
        o = objects_by_ns[r["o"]]
        u, v, z, ok = visibility._project_batch(
            kf.world_T_body,
            {"body_T_sensor": _BODY_T_SENSOR, **_CALIB_KW},
            [[o["x"], o["y"], o["z"]]], max_depth_m=max_depth_m)
        assert ok[0]
        assert r["u"] == pytest.approx(u[0])
        assert r["v"] == pytest.approx(v[0])
        assert r["d"] == pytest.approx(z[0])
        assert 0 <= r["u"] < _CALIB_KW["width"]
        assert 0.1 < r["d"] <= max_depth_m


def test_views_match_frustum_skip_set_per_frame():
    # detect_frustum_aware skips a frame iff ANY found object is in its
    # frustum — the per-frame any() over VIEWS must reproduce that skip-set.
    frame_rows, object_rows, keyframes = _fixture(seed=11)
    max_depth_m = 6.0
    db = FakeDb()
    db.respond("MATCH (o:Object) WHERE o.pos_x IS NOT NULL", object_rows)
    db.respond("MATCH (n:TrajectoryFrame)-[:HAS_CALIB]", frame_rows)
    visibility.materialize_views(db, max_depth_m=max_depth_m)
    frames_with_views = {r["f"] for r in db.view_rows()}

    centroids = [np.array([o["x"], o["y"], o["z"]]) for o in object_rows]
    skip_set = {
        sym for sym, kf in keyframes.items()
        if any(_in_frustum(c, kf, max_depth_m=max_depth_m) for c in centroids)
    }
    assert frames_with_views == skip_set


def test_project_batch_gates():
    calib = {"body_T_sensor": np.eye(4), "fx": 100.0, "fy": 100.0,
             "cx": 320.0, "cy": 240.0, "width": 640, "height": 480}
    pts = [
        [0, 0, 2],      # dead center, in view
        [0, 0, -2],     # behind the camera
        [0, 0, 0.05],   # closer than the 0.1 m front gate
        [0, 0, 50],     # beyond max_depth
        [10, 0, 2],     # u = 820, off image
    ]
    u, v, z, ok = visibility._project_batch(np.eye(4), calib, pts,
                                            max_depth_m=10.0)
    assert list(ok) == [True, False, False, False, False]
    assert u[0] == 320.0 and v[0] == 240.0 and z[0] == 2.0
    # margin widens the image bound gate
    *_, ok_margin = visibility._project_batch(np.eye(4), calib, pts,
                                              margin=200.0, max_depth_m=10.0)
    assert bool(ok_margin[4])
    # no depth ceiling: the far point is a view
    *_, ok_nodepth = visibility._project_batch(np.eye(4), calib, pts)
    assert bool(ok_nodepth[3])


def test_object_filter_and_empty_graph():
    frame_rows, object_rows, _ = _fixture(n_frames=3, n_objects=4)
    db = FakeDb()
    db.respond("MATCH (o:Object) WHERE o.pos_x IS NOT NULL", object_rows)
    db.respond("MATCH (n:TrajectoryFrame)-[:HAS_CALIB]", frame_rows)
    visibility.materialize_views(db, object_symbols=["O(1)", "O(2)"])
    obj_queries = [(c, p) for c, p in db.executed
                   if "RETURN o.nodeSymbol AS ns" in c]
    assert obj_queries
    assert all(p.get("syms") == ["O(1)", "O(2)"] for _, p in obj_queries)

    empty = FakeDb()
    assert visibility.materialize_views(empty) == 0
    assert empty.view_rows() == []


def test_refresh_drops_stale_views_first():
    frame_rows, object_rows, _ = _fixture(n_frames=2, n_objects=2)
    db = FakeDb()
    db.respond("MATCH (o:Object) WHERE o.pos_x IS NOT NULL", object_rows)
    db.respond("MATCH (n:TrajectoryFrame)-[:HAS_CALIB]", frame_rows)
    visibility.materialize_views(db, refresh=True)
    deletes = [c for c, _ in db.executed if "DELETE r" in c and ":VIEWS" in c]
    assert len(deletes) == len(visibility.FRAME_LABELS)


def test_labels_match_heracles_constants():
    heracles_constants = pytest.importorskip("heracles.constants")
    assert visibility.VIEWS == heracles_constants.VIEWS
    assert visibility.COVERED == heracles_constants.COVERED
    assert set(visibility.FRAME_LABELS) == {
        heracles_constants.AGENTS,
        heracles_constants.SUBKEYFRAMES,
        heracles_constants.TRAJECTORY_FRAMES,
    }
