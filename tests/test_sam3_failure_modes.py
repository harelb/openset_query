"""Failure-mode coverage for the SAM3 frontend — the things that actually break
in the field: corrupt/partial keyframes on disk, bad depth/mask data,
pose-quaternion edge cases, and fusion robustness. All numpy-only / mock-based: no GPU, no SAM3, no Neo4j.
"""

import numpy as np
import pytest

from openset_query import keyframe_store as ks
from openset_query import object_discovery as od
from openset_query.config import Sam3FrontendConfig
from openset_query.reprojection import (
    CameraCalib,
    fuse_detections,
    mask_to_points,
    points_to_object,
)
from openset_query.sam3_segmenter import Detection


def _calib(body_T_sensor=None):
    return CameraCalib(
        100.0,
        100.0,
        50.0,
        50.0,
        100,
        100,
        1e-3,
        np.eye(4) if body_T_sensor is None else body_T_sensor,
    )


def _kf(calib, world_T_body, ts=1):
    return ks.Keyframe(
        timestamp_ns=ts,
        rgb_path="",
        depth_path="",
        world_T_body=world_T_body,
        calib=calib,
    )


# ---------------------------------------------------------------------------
# Pose quaternion extraction (ingest) — a wrong branch silently corrupts poses
# ---------------------------------------------------------------------------


def _quat_to_R(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _Rx(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _Ry(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _Rz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


@pytest.mark.parametrize(
    "R",
    [
        np.eye(3),  # trace>0 branch
        _Rx(np.pi / 2),
        _Ry(np.pi / 2),
        _Rz(np.pi / 2),
        _Rx(np.pi),
        _Ry(np.pi),
        _Rz(np.pi),  # each hits a different largest-diagonal branch
        _Rz(0.3) @ _Ry(-0.7) @ _Rx(1.1),  # general composed rotation
    ],
)
def test_quat_from_R_round_trips(R):
    from openset_query.ingest_keyframes import _quat_from_R

    q = _quat_from_R(R)
    assert abs(np.linalg.norm(q) - 1.0) < 1e-9  # unit quaternion
    np.testing.assert_allclose(_quat_to_R(q), R, atol=1e-9)  # reconstructs the rotation


# ---------------------------------------------------------------------------
# Keyframe loading — partial data in the store drops the frame, not the run
# ---------------------------------------------------------------------------


def _store_with(tmp_path, **frame_kw):
    from store_helpers import build_store

    sb = build_store(tmp_path / "store")
    fid = sb.frame(1, **frame_kw)
    return sb.close(), fid


def _load(store_dir, **kw):
    from hydra_python.image_store import ImageStore

    store = ImageStore.open(store_dir)
    try:
        return ks.load_keyframes_from_store(store, **kw)
    finally:
        store.close()


def test_loader_frame_without_depth_is_skipped(tmp_path):
    store, _ = _store_with(tmp_path, depth=False)
    assert _load(store) == []


def test_loader_frame_without_rgb_is_skipped(tmp_path):
    store, _ = _store_with(tmp_path, rgb=False)
    assert _load(store) == []


def test_loader_sensorless_frame_is_skipped(tmp_path):
    """A frame whose sensor row has gone missing cannot be reprojected."""
    import sqlite3

    store, _ = _store_with(tmp_path)
    con = sqlite3.connect(str(store / "store.sqlite"))
    con.execute("PRAGMA foreign_keys = OFF")
    con.execute("DELETE FROM sensors")
    con.commit()
    con.close()
    assert _load(store) == []


def test_loader_missing_anchor_falls_back_to_odometry(tmp_path):
    odom = np.eye(4)
    odom[:3, 3] = [1.0, 2.0, 3.0]
    from store_helpers import build_store, node_id

    sb = build_store(tmp_path / "store")
    sb.frame(1, anchor=node_id("a0"), pose=odom)
    store = sb.close()
    (kf,) = _load(store, pose_lookup={})  # anchor not in the lookup
    np.testing.assert_allclose(kf.world_T_body, odom)


def test_loader_poses_keyframes_from_the_anchor_node(tmp_path):
    from store_helpers import build_store, node_id

    sb = build_store(tmp_path / "store")
    sb.frame(1, anchor=node_id("a0"))
    store = sb.close()
    anchor = np.eye(4)
    anchor[:3, 3] = [5.0, 0.0, 1.0]
    (kf,) = _load(store, pose_lookup={node_id("a0"): anchor})
    np.testing.assert_allclose(kf.world_T_body, anchor)
    assert kf.frame_id is not None and kf.sensor_id is not None
    assert kf.calib.depth_scale == pytest.approx(1e-3)


# ---------------------------------------------------------------------------
# Reprojection — bad depth / mask shapes
# ---------------------------------------------------------------------------


def test_mask_depth_shape_mismatch_returns_empty():
    calib = _calib()
    mask = np.ones((50, 50), dtype=bool)  # different resolution than depth
    depth = np.full((100, 100), 2.0)
    pts = mask_to_points(mask, depth, calib, np.eye(4))
    assert pts.shape == (0, 3)  # skipped, not crashed


def test_all_invalid_depth_returns_empty():
    calib = _calib()
    mask = np.ones((100, 100), dtype=bool)
    depth = np.zeros((100, 100))  # all 0 -> all invalid
    assert mask_to_points(mask, depth, calib, np.eye(4)).shape == (0, 3)


def test_nan_inf_depth_holes_skipped():
    calib = _calib()
    mask = np.zeros((100, 100), dtype=bool)
    mask[50, 50] = mask[50, 51] = mask[50, 52] = True
    depth = np.full((100, 100), np.nan)
    depth[50, 50] = 2.0
    depth[50, 51] = np.inf
    depth[50, 52] = 0.0  # invalid (below min)
    pts = mask_to_points(mask, depth, calib, np.eye(4))
    assert pts.shape == (1, 3)  # only the finite, in-range pixel


def test_coincident_points_make_zero_volume_bbox():
    pts = np.tile([1.0, 2.0, 3.0], (20, 1))
    obj = points_to_object(pts, "x", 0.9, percentile_lo=0, percentile_hi=100)
    assert obj is not None
    np.testing.assert_allclose(obj.bbox_dimensions, [0, 0, 0])


# ---------------------------------------------------------------------------
# Frustum test — drives the early-stop skip; must respect range + orientation
# ---------------------------------------------------------------------------


def test_in_frustum_point_in_front_center():
    kf = _kf(_calib(), np.eye(4))
    assert od._in_frustum([0.0, 0.0, 2.0], kf) is True


def test_in_frustum_behind_camera_false():
    kf = _kf(_calib(), np.eye(4))
    assert od._in_frustum([0.0, 0.0, -2.0], kf) is False


def test_in_frustum_outside_image_false():
    kf = _kf(_calib(), np.eye(4))
    assert od._in_frustum([10.0, 0.0, 2.0], kf) is False  # projects to u~550, off-image


def test_in_frustum_beyond_max_depth_false():
    kf = _kf(_calib(), np.eye(4))
    assert od._in_frustum([0.0, 0.0, 2.0], kf, max_depth_m=1.0) is False
    assert od._in_frustum([0.0, 0.0, 2.0], kf, max_depth_m=5.0) is True


# ---------------------------------------------------------------------------
# Fusion robustness — correct weighting, order independence, no caller mutation
# ---------------------------------------------------------------------------


def _o(centroid, label, score):
    c = np.asarray(centroid, float)
    return points_to_object(c.reshape(1, 3), label, score, 0, 100)


def test_fuse_three_views_score_weighted_centroid():
    # x=0/0.2/0.4 with scores 0.9/0.6/0.3 -> weighted mean 0.24/1.8 = 0.1333...
    objs = [
        _o([0.0, 0, 0], "c", 0.9),
        _o([0.2, 0, 0], "c", 0.6),
        _o([0.4, 0, 0], "c", 0.3),
    ]
    fused = fuse_detections(objs, centroid_dist_m=0.75)
    assert len(fused) == 1
    np.testing.assert_allclose(fused[0].centroid[0], 0.24 / 1.8, atol=1e-9)


def test_fuse_order_independent():
    base = [
        _o([0.0, 0, 0], "c", 0.9),
        _o([0.2, 0, 0], "c", 0.6),
        _o([0.4, 0, 0], "c", 0.3),
    ]
    a = fuse_detections([_o(list(o.centroid), o.label, o.score) for o in base], 0.75)[
        0
    ].centroid
    rev = fuse_detections(
        [_o(list(o.centroid), o.label, o.score) for o in reversed(base)], 0.75
    )[0].centroid
    np.testing.assert_allclose(a, rev, atol=1e-9)


def test_fuse_does_not_mutate_caller_objects():
    a, b = _o([0.0, 0, 0], "c", 0.9), _o([0.2, 0, 0], "c", 0.6)
    a_before = a.centroid.copy()
    fuse_detections([a, b], 0.75)
    np.testing.assert_allclose(a.centroid, a_before)  # input object untouched


# ---------------------------------------------------------------------------
# detect_in_keyframes sets the object's primary color (regression: was magenta)
# ---------------------------------------------------------------------------


class _ColorSegmenter:
    def segment(self, image, prompts):
        mask = np.zeros((100, 100), dtype=bool)
        mask[45:56, 45:56] = True
        return [
            Detection(label="box", mask=mask, box=np.array([45, 45, 56, 56]), score=0.9)
        ]


def test_detect_in_keyframes_sets_primary_color(monkeypatch):
    calib = _calib()
    rgb = np.zeros((100, 100, 3), np.uint8)
    rgb[45:56, 45:56] = (10, 20, 30)  # known object color
    monkeypatch.setattr(ks, "read_rgb", lambda kf: rgb)
    monkeypatch.setattr(ks, "read_depth_meters", lambda kf: np.full((100, 100), 2.0))
    cfg = Sam3FrontendConfig(min_mask_depth_pixels=10)
    discovered = od.detect_in_keyframes(
        [_kf(calib, np.eye(4))], _ColorSegmenter(), ["box"], cfg
    )
    assert len(discovered) == 1
    assert discovered[0].obj.color == (10, 20, 30)  # not the (255,0,255) sentinel
