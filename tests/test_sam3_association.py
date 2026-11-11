"""Same-object association (association.py): 3D clustering of per-view detections."""

import numpy as np
import pytest

from openset_query import association as assoc
from openset_query import object_discovery as od
from openset_query.config import Sam3FrontendConfig
from openset_query.reprojection import Object3D


class _KF:
    def __init__(self, fid, ns=None):
        self.frame_id, self.cache_ns = fid, ns


def det(label, lo, hi, score=0.8, frame=0, points=True, rng=None):
    """A per-view detection with a box [lo, hi] (and points filling it)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    pts = None
    if points:
        rng = rng or np.random.default_rng(frame * 7 + int(score * 100))
        pts = lo + rng.random((400, 3)) * (hi - lo)
    o = Object3D(
        centroid=(lo + hi) / 2,
        bbox_min=lo,
        bbox_max=hi,
        num_points=400,
        label=label,
        score=score,
        points=pts,
    )
    ob = od.Observation(
        timestamp_ns=1000 + frame,
        mask_file="",
        bbox_2d=(0, 0, 5, 5),
        score=score,
        keyframe=_KF(frame),
    )
    return od.DiscoveredObject(obj=o, observations=[ob])


def chair(x, frame, jitter=0.0, score=0.8):
    return det("chair", [x + jitter, 0, 0], [x + 0.5 + jitter, 0.5, 0.9], score, frame)


def test_two_views_of_one_chair_and_a_second_chair_two_metres_away():
    views = [
        chair(0.0, 0, score=0.9),
        chair(0.0, 1, jitter=0.12, score=0.7),
        chair(2.0, 2, score=0.8),
    ]
    out = assoc.cluster_detections(views)
    assert len(out) == 2
    a, b = out  # best cluster first
    assert a.obj.score == 0.9 and len(a.observations) == 2 and a.num_frames == 2
    assert [o.score for o in a.observations] == [0.9, 0.7]  # best view first
    assert b.num_frames == 1 and b.obj.centroid[0] > 1.9
    # merged box comes from the pooled points (robust percentiles), inside the union
    assert a.obj.bbox_min[0] >= -1e-9 and a.obj.bbox_max[0] <= 0.62 + 1e-9


def test_mug_inside_table_is_not_merged_across_labels():
    table = det("table", [0, 0, 0], [1.6, 0.9, 0.75], 0.9, 0)
    mug = det("mug", [0.5, 0.4, 0.75], [0.6, 0.5, 0.85], 0.8, 0)
    mug2 = det("mug", [0.51, 0.41, 0.74], [0.61, 0.5, 0.86], 0.6, 1)
    out = assoc.cluster_detections([table, mug, mug2])
    labels = sorted((d.obj.label, len(d.observations)) for d in out)
    assert labels == [("mug", 2), ("table", 1)]


def test_same_label_partial_view_is_merged_by_containment():
    full = det("sofa", [0, 0, 0], [2.0, 0.9, 0.8], 0.9, 0)
    part = det(
        "sofa", [1.25, 0.1, 0.15], [1.9, 0.8, 0.65], 0.6, 1
    )  # IoU small, contained
    iou, contain, ratio = assoc.overlap_terms(
        full.obj.bbox_min, full.obj.bbox_max, part.obj.bbox_min, part.obj.bbox_max
    )
    assert iou < 0.2 and contain > 0.99
    (one,) = assoc.cluster_detections([full, part])
    assert len(one.observations) == 2


def test_smeared_box_does_not_swallow_small_objects():
    # one bad view spanning a row of three chairs must not chain them together
    smeared = det("chair", [0, 0, 0], [4.5, 0.6, 1.0], 0.5, 9)
    chairs = [chair(x, i) for i, x in enumerate((0.0, 2.0, 4.0))]
    out = assoc.cluster_detections(chairs + [smeared])
    # (its centroid may coincide with the middle chair's, which then absorbs it)
    homes = [
        next(i for i, d in enumerate(out) if c.observations[0] in d.observations)
        for c in chairs
    ]
    assert len(set(homes)) == 3 and len(out) >= 3


def test_adjacent_distinct_chairs_stay_apart():
    out = assoc.cluster_detections([chair(0.0, 0), chair(0.65, 1)])
    assert len(out) == 2


def test_appearance_gate_splits_lookalike_boxes():
    a, b = chair(0.0, 0), chair(0.05, 1)
    e1 = np.array([1.0, 0.0])
    e2 = np.array([0.0, 1.0])
    assert len(assoc.cluster_detections([a, b], embeddings=[e1, e1])) == 1
    assert len(assoc.cluster_detections([a, b], embeddings=[e1, e2])) == 2
    # a missing embedding never blocks a geometric match
    assert len(assoc.cluster_detections([a, b], embeddings=[e1, None])) == 1


def test_no_points_falls_back_to_score_weighted_union():
    a = det("box", [0, 0, 0], [1, 1, 1], 0.9, 0, points=False)
    b = det("box", [0.1, 0, 0], [1.1, 1, 1], 0.3, 1, points=False)
    (one,) = assoc.cluster_detections([a, b])
    assert np.allclose(one.obj.bbox_min, [0, 0, 0]) and np.allclose(
        one.obj.bbox_max, [1.1, 1, 1]
    )
    assert one.obj.centroid[0] == pytest.approx((0.5 * 0.9 + 0.6 * 0.3) / 1.2)
    assert one.obj.score == 0.9 and one.obj.num_points == 800


def test_inputs_are_not_mutated_and_empty_input():
    views = [chair(0.0, 0), chair(0.05, 1)]
    before = [v.obj.bbox_max.copy() for v in views]
    assoc.cluster_detections(views)
    assert all(np.array_equal(b, v.obj.bbox_max) for b, v in zip(before, views))
    assert assoc.cluster_detections([]) == []


def test_many_detections_scale_linearly_and_group_per_object():
    rng = np.random.default_rng(0)
    views = []
    for k in range(200):  # 200 objects on a 4 m grid, 5 noisy views each
        x, y = (k % 20) * 4.0, (k // 20) * 4.0
        for f in range(5):
            j = rng.normal(0, 0.05, 3)
            views.append(
                det(
                    "cone",
                    [x + j[0], y + j[1], 0],
                    [x + 0.4 + j[0], y + 0.4 + j[1], 0.7],
                    0.5 + 0.01 * f,
                    frame=k * 5 + f,
                    rng=rng,
                )
            )
    import time

    t = time.perf_counter()
    out = assoc.cluster_detections(views)
    assert len(out) == 200 and all(d.num_frames == 5 for d in out)
    assert time.perf_counter() - t < 5.0


def test_huge_boxes_do_not_explode_the_grid():
    big = det("wall", [0, 0, 0], [500, 500, 50], 0.9, 0, points=False)
    small = det("wall", [1, 1, 1], [2, 2, 2], 0.5, 1, points=False)
    out = assoc.cluster_detections([big, small])
    assert 1 <= len(out) <= 2


def test_associate_dispatch_and_config_params():
    cfg = Sam3FrontendConfig()
    views = [chair(0.0, 0), chair(0.1, 1), chair(3.0, 2)]
    assert len(od.associate(views, cfg)) == 2
    cfg.association = "centroid"
    assert len(od.associate(views, cfg)) == 2
    p = assoc.params_from_config(
        Sam3FrontendConfig(fuse_centroid_dist_m=0.5, assoc_iou_min=0.4)
    )
    assert p.max_dist_m == 0.5 and p.iou_min == 0.4
    p = assoc.params_from_config(Sam3FrontendConfig(assoc_max_dist_m=1.5))
    assert p.max_dist_m == 1.5


def test_frame_keys_count_distinct_frames():
    a, b = chair(0.0, 4), chair(0.02, 4)  # two detections of one frame
    b.observations[0].keyframe = _KF(4)
    (one,) = assoc.cluster_detections([a, b])
    assert len(one.observations) == 2 and one.num_frames == 1
