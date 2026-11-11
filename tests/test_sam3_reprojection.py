"""Unit tests for the SAM3 frontend reprojection geometry (numpy-only)."""

import numpy as np
import pytest

from openset_query.reprojection import (
    CameraCalib,
    depth_mode_gate,
    depth_to_meters,
    fuse_detections,
    mask_to_points,
    points_to_object,
)


def make_calib(body_T_sensor=None):
    if body_T_sensor is None:
        body_T_sensor = np.eye(4)
    return CameraCalib(
        fx=100.0,
        fy=100.0,
        cx=50.0,
        cy=50.0,
        width=100,
        height=100,
        depth_scale=1e-3,  # raw stored in millimeters
        body_T_sensor=body_T_sensor,
    )


def test_single_pixel_at_principal_point_identity_pose():
    calib = make_calib()
    depth_raw = np.zeros((100, 100), dtype=np.uint16)
    depth_raw[50, 50] = 2000  # 2.0 m
    mask = np.zeros((100, 100), dtype=bool)
    mask[50, 50] = True

    depth_m = depth_to_meters(depth_raw, calib)
    pts = mask_to_points(mask, depth_m, calib, np.eye(4))

    assert pts.shape == (1, 3)
    np.testing.assert_allclose(pts[0], [0.0, 0.0, 2.0], atol=1e-9)


def test_offset_pixel_backprojection():
    calib = make_calib()
    depth_m = np.full((100, 100), np.nan)
    depth_m[50, 60] = 2.0  # u=60, v=50
    mask = np.zeros((100, 100), dtype=bool)
    mask[50, 60] = True

    pts = mask_to_points(mask, depth_m, calib, np.eye(4))
    # x = d*(u-cx)/fx = 2*(60-50)/100 = 0.2
    np.testing.assert_allclose(pts[0], [0.2, 0.0, 2.0], atol=1e-9)


def test_world_translation_applied():
    calib = make_calib()
    depth_m = np.full((100, 100), np.nan)
    depth_m[50, 50] = 2.0
    mask = np.zeros((100, 100), dtype=bool)
    mask[50, 50] = True

    world_T_body = np.eye(4)
    world_T_body[:3, 3] = [10.0, 0.0, 0.0]

    pts = mask_to_points(mask, depth_m, calib, world_T_body)
    np.testing.assert_allclose(pts[0], [10.0, 0.0, 2.0], atol=1e-9)


def test_world_rotation_applied():
    calib = make_calib()
    depth_m = np.full((100, 100), np.nan)
    depth_m[50, 60] = 2.0  # camera point [0.2, 0, 2]
    mask = np.zeros((100, 100), dtype=bool)
    mask[50, 60] = True

    # 90deg about z: [x, y, z] -> [-y, x, z]
    world_T_body = np.eye(4)
    world_T_body[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]

    pts = mask_to_points(mask, depth_m, calib, world_T_body)
    np.testing.assert_allclose(pts[0], [0.0, 0.2, 2.0], atol=1e-9)


def test_extrinsics_compose_with_pose():
    # body_T_sensor translates +1 in body-x; world pose translates +10 in world-x.
    body_T_sensor = np.eye(4)
    body_T_sensor[:3, 3] = [1.0, 0.0, 0.0]
    calib = make_calib(body_T_sensor)

    depth_m = np.full((100, 100), np.nan)
    depth_m[50, 50] = 2.0  # camera point [0, 0, 2]
    mask = np.zeros((100, 100), dtype=bool)
    mask[50, 50] = True

    world_T_body = np.eye(4)
    world_T_body[:3, 3] = [10.0, 0.0, 0.0]

    pts = mask_to_points(mask, depth_m, calib, world_T_body)
    # sensor->body adds +1x: [1,0,2]; body->world adds +10x: [11,0,2]
    np.testing.assert_allclose(pts[0], [11.0, 0.0, 2.0], atol=1e-9)


def test_invalid_depth_rejected():
    calib = make_calib()
    depth_m = np.full((100, 100), np.nan)
    depth_m[50, 50] = 0.0  # below min_depth
    depth_m[50, 51] = 100.0  # above max_depth
    mask = np.zeros((100, 100), dtype=bool)
    mask[50, 50] = True
    mask[50, 51] = True

    pts = mask_to_points(mask, depth_m, calib, np.eye(4))
    assert pts.shape == (0, 3)


def test_points_to_object_centroid_and_bbox():
    pts = np.array(
        [[0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [0.5, 0.5, 2.0]], dtype=np.float64
    )
    obj = points_to_object(pts, label="chair", score=0.9, percentile_lo=0, percentile_hi=100)
    assert obj is not None
    np.testing.assert_allclose(obj.centroid, [0.5, 0.5 / 3.0 * 1.0, 2.0], atol=1e-9)
    np.testing.assert_allclose(obj.bbox_min, [0.0, 0.0, 2.0])
    np.testing.assert_allclose(obj.bbox_max, [1.0, 0.5, 2.0])
    np.testing.assert_allclose(obj.bbox_dimensions, [1.0, 0.5, 0.0])


def test_points_to_object_empty_returns_none():
    assert points_to_object(np.empty((0, 3)), "x", 1.0) is None


def _obj(centroid, label, score):
    c = np.asarray(centroid, dtype=np.float64)
    return points_to_object(c.reshape(1, 3), label, score, 0, 100)


def test_fuse_merges_close_same_label():
    a = _obj([0.0, 0.0, 0.0], "chair", 0.8)
    b = _obj([0.3, 0.0, 0.0], "chair", 0.6)  # within 0.75 m
    fused = fuse_detections([a, b], centroid_dist_m=0.75)
    assert len(fused) == 1
    assert fused[0].score == pytest.approx(0.8)


def test_fuse_keeps_far_apart():
    a = _obj([0.0, 0.0, 0.0], "chair", 0.8)
    b = _obj([5.0, 0.0, 0.0], "chair", 0.6)
    assert len(fuse_detections([a, b], centroid_dist_m=0.75)) == 2


def test_fuse_keeps_different_labels():
    a = _obj([0.0, 0.0, 0.0], "chair", 0.8)
    b = _obj([0.1, 0.0, 0.0], "table", 0.6)
    assert len(fuse_detections([a, b], centroid_dist_m=0.75)) == 2


# --- depth-mode gate: reject background bleed behind a segmented object ------


def test_depth_mode_gate_drops_separated_background_cluster():
    # Object surface at ~2.5 m plus a smaller background cluster at ~5.0 m that a
    # mask edge picked up. The background is separated by an empty depth gap.
    depths = np.concatenate([np.full(200, 2.5), np.full(40, 5.0)])
    keep = depth_mode_gate(depths)
    assert keep[:200].all()
    assert not keep[200:].any()


def test_depth_mode_gate_keeps_contiguous_deep_object():
    # A genuinely deep object (e.g. a couch) spans a contiguous depth range with
    # no gap; the whole thing must survive (no fixed-size assumption).
    depths = np.linspace(2.0, 3.2, 300)
    assert depth_mode_gate(depths).all()


def test_depth_mode_gate_keeps_all_when_sparse():
    # Too few points to form a histogram cluster -> don't gate (never worse).
    depths = np.array([2.0, 5.0, 8.0])
    assert depth_mode_gate(depths).all()


def test_mask_to_points_gates_background_bleed():
    calib = make_calib()
    depth_m = np.full((100, 100), np.nan)
    mask = np.zeros((100, 100), dtype=bool)
    # object: 10x10 block near the principal point at 2.5 m
    depth_m[40:50, 40:50] = 2.5
    mask[40:50, 40:50] = True
    # background bleed: 4x5 block at 5.0 m (separated in depth)
    depth_m[40:44, 60:65] = 5.0
    mask[40:44, 60:65] = True

    pts = mask_to_points(mask, depth_m, calib, np.eye(4))
    assert pts.shape[0] == 100  # only the near object block survives
    assert pts[:, 2].max() < 3.0  # no 5.0 m background points
