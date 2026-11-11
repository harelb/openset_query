"""Integration tests for the SAM3 frontend keyframe + discovery pipeline.

Exercises the image-store keyframe contract and the
detect -> reproject -> fuse path, using a fake segmenter (no SAM3 / no GPU / no
Neo4j).
"""

import os

import numpy as np
import pytest

from openset_query import keyframe_store as ks
from openset_query import object_discovery as od
from openset_query.config import Sam3FrontendConfig
from openset_query.reprojection import CameraCalib
from openset_query.sam3_segmenter import Detection


def test_store_keyframe_contract(tmp_path):
    """A store keyframe -> Keyframe: calib from the sensors row, pose composed from
    the anchor node, ids carried for the observation write-back."""
    from hydra_python.image_store import ImageStore
    from store_helpers import build_store, node_id

    sb = build_store(tmp_path / "store")
    fid = sb.frame(123456789, anchor=node_id("a1"))
    store_dir = sb.close()

    world_T_body = np.eye(4)
    world_T_body[:3, 3] = [5.0, 6.0, 7.0]
    store = ImageStore.open(store_dir)
    try:
        (kf,) = ks.load_keyframes_from_store(
            store, pose_lookup={node_id("a1"): world_T_body}
        )
    finally:
        store.close()
    assert kf.timestamp_ns == 123456789
    assert kf.frame_id == fid
    assert isinstance(kf.calib, CameraCalib)
    assert kf.calib.fx == 50.0 and kf.calib.depth_scale == pytest.approx(1e-3)
    np.testing.assert_allclose(kf.world_T_body[:3, 3], [5.0, 6.0, 7.0])
    # Identity extrinsics -> camera position == body position.
    np.testing.assert_allclose(kf.camera_position, [5.0, 6.0, 7.0])
    # Depth is uint16 millimeters x depth_scale.
    np.testing.assert_allclose(ks.read_depth_meters(kf), 2.0)
    assert ks.read_rgb(kf).shape == (48, 64, 3)


class FakeSegmenter:
    """Returns one 'chair' detection: a filled square mask around (cx, cy)."""

    def __init__(self, calib):
        self.calib = calib

    def segment(self, image, prompts):
        if "chair" not in prompts:
            return []
        mask = np.zeros((self.calib.height, self.calib.width), dtype=bool)
        mask[45:56, 45:56] = True  # 11x11 block centered on principal point
        return [
            Detection(
                label="chair", mask=mask, box=np.array([45, 45, 56, 56]), score=0.9
            )
        ]


def _calib():
    return CameraCalib(100, 100, 50, 50, 100, 100, 1e-3, np.eye(4))


def _keyframe(calib, world_T_body, ts):
    return ks.Keyframe(
        timestamp_ns=ts,
        rgb_path="",
        depth_path="",
        world_T_body=world_T_body,
        calib=calib,
    )


def test_detect_reproject_fuse_two_views(monkeypatch):
    calib = _calib()
    # Two keyframes from different body poses both looking at a point ~2m ahead.
    kf1 = _keyframe(calib, np.eye(4), ts=1)
    pose2 = np.eye(4)
    pose2[:3, 3] = [0.1, 0.0, 0.0]  # small lateral shift -> centroids within fuse dist
    kf2 = _keyframe(calib, pose2, ts=2)

    # Constant 2 m depth across the image for both views.
    depth = np.full((100, 100), 2.0)
    monkeypatch.setattr(ks, "read_rgb", lambda kf: np.zeros((100, 100, 3), np.uint8))
    monkeypatch.setattr(ks, "read_depth_meters", lambda kf: depth)

    config = Sam3FrontendConfig(min_mask_depth_pixels=10, fuse_centroid_dist_m=0.75)
    seg = FakeSegmenter(calib)
    discovered = od.detect_in_keyframes([kf1, kf2], seg, ["chair"], config)

    # Two views of the same chair fuse into one object with both observations.
    assert len(discovered) == 1
    do = discovered[0]
    assert do.obj.label == "chair"
    assert len(do.observations) == 2
    # Centroid near principal-point ray at 2 m, between the two body poses (~x=0.05).
    np.testing.assert_allclose(do.obj.centroid[2], 2.0, atol=1e-6)
    assert -0.1 <= do.obj.centroid[0] <= 0.2


def test_below_min_pixels_rejected(monkeypatch):
    calib = _calib()
    kf = _keyframe(calib, np.eye(4), ts=1)
    depth = np.full((100, 100), 2.0)
    monkeypatch.setattr(ks, "read_rgb", lambda kf: np.zeros((100, 100, 3), np.uint8))
    monkeypatch.setattr(ks, "read_depth_meters", lambda kf: depth)

    config = Sam3FrontendConfig(min_mask_depth_pixels=1000)  # 11x11=121 < 1000
    discovered = od.detect_in_keyframes([kf], FakeSegmenter(calib), ["chair"], config)
    assert discovered == []


def _discovered(centroid, label="chair", score=0.9):
    from openset_query.reprojection import Object3D

    obj = Object3D(
        centroid=np.asarray(centroid, float),
        bbox_min=np.asarray(centroid, float) - 0.5,
        bbox_max=np.asarray(centroid, float) + 0.5,
        num_points=100,
        label=label,
        score=score,
    )
    return od.DiscoveredObject(
        obj=obj,
        observations=[
            od.Observation(timestamp_ns=1, mask_file="", bbox_2d=(0, 0, 10, 10))
        ],
    )


def _dobj_from_points(points, label, score, ts):
    from openset_query.reprojection import points_to_object

    # percentile 0/100 so the per-view box is the raw extent; fusion re-reduces.
    obj = points_to_object(np.asarray(points, float), label, score, 0, 100)
    return od.DiscoveredObject(
        obj=obj,
        observations=[
            od.Observation(timestamp_ns=ts, mask_file="", bbox_2d=(0, 0, 10, 10))
        ],
    )


def test_fuse_pools_points_and_trims_outlier_view():
    # A clean dense view (0.4 m cube at origin) fused with a "bled" view sharing
    # the same centroid region but carrying a minority far tail at x=2 m. Union of
    # per-view boxes would span ~2.2 m; pooling the points and reducing once trims
    # the minority tail, keeping the box object-sized.
    g = np.linspace(-0.2, 0.2, 10)
    clean = np.array([[x, y, z] for x in g for y in g for z in g])  # 1000 pts
    tail = np.column_stack([np.full(30, 2.0), np.zeros(30), np.zeros(30)])
    bled = np.vstack([clean[:200], tail])  # centroid ~0.26 -> within 0.75 of origin

    a = _dobj_from_points(clean, "chair", 0.9, ts=1)
    b = _dobj_from_points(bled, "chair", 0.8, ts=2)
    fused = od._fuse([a, b], 0.75)

    assert len(fused) == 1
    assert len(fused[0].observations) == 2
    dim = fused[0].obj.bbox_dimensions
    assert dim[0] < 0.6  # union would be ~2.2 m; pool-then-reduce stays object-sized


def test_save_mask_unique_per_detection_in_same_frame(tmp_path):
    import types

    kf = types.SimpleNamespace(timestamp_ns=42)
    m1 = np.zeros((10, 10), bool)
    m1[0:3, 0:3] = True
    m2 = np.zeros((10, 10), bool)
    m2[5:8, 5:8] = True
    p1 = od._save_mask(m1, kf, "chair", str(tmp_path))
    p2 = od._save_mask(m2, kf, "chair", str(tmp_path))
    assert p1 != p2  # two "chair" masks in one frame must not overwrite each other
    assert os.path.exists(p1) and os.path.exists(p2)


def _store_and_kf(tmp_path, ts=42):
    from hydra_python.image_store import ImageStore
    from store_helpers import build_store

    sb = build_store(tmp_path / "store", size=(20, 20))
    fid = sb.frame(ts)
    store_dir = sb.close()
    store = ImageStore.open(store_dir)
    try:
        (kf,) = ks.load_keyframes_from_store(store)
    finally:
        store.close()
    assert kf.frame_id == fid
    return store_dir, kf


def _mask_png(path, box):
    from PIL import Image as PILImage

    x0, y0, x1, y1 = box
    m = np.zeros((20, 20), np.uint8)
    m[y0 : y1 + 1, x0 : x1 + 1] = 255
    PILImage.fromarray(m).save(path)
    return str(path)


@pytest.mark.adt4
def test_record_observations_keeps_mask_and_bbox_paired_same_ts(tmp_path):
    """Two detections in the SAME keyframe (same ts) fused into one object must
    each keep their own mask paired with their own bbox, on the keyframe's existing
    frame row (no new frame, no new image)."""
    from hydra_python.image_store import ImageStore

    from store_helpers import node_id

    store_dir, kf = _store_and_kf(tmp_path)
    masks = tmp_path / "masks"
    masks.mkdir()
    box_a, box_b = (1, 2, 4, 6), (10, 11, 15, 18)
    obs = [
        od.Observation(
            42, _mask_png(masks / "a_mask.png", box_a), box_a, kf.rgb_path, keyframe=kf
        ),
        od.Observation(
            42, _mask_png(masks / "b_mask.png", box_b), box_b, kf.rgb_path, keyframe=kf
        ),
    ]

    w = od.open_sam3_writer(str(store_dir))
    od.record_observations(w, "O1", obs, new_object=True)
    w.close()

    store = ImageStore.open(store_dir)
    try:
        got = store.observations(node_id("O1"))
        assert len(got) == 2  # both co-timestamped observations survive
        assert {o.frame_id for o in got} == {kf.frame_id}  # reused frame row
        assert len(store.frames()) == 1
        for o in got:
            assert o.kept
            ys, xs = np.nonzero(store.mask(o))
            assert (
                int(xs.min()),
                int(ys.min()),
                int(xs.max()),
                int(ys.max()),
            ) == tuple(o.bbox2d)
            assert str(o.rgb_path) == kf.rgb_path
    finally:
        store.close()
    # the write-back the Neo4j Observation node is built from
    assert [o.frame_id for o in obs] == [kf.frame_id] * 2
    assert all(
        o.obs_id is not None and o.mask_path and o.depth_path and o.kept for o in obs
    )
    d = od._obs_dict("O1", obs[0])
    assert (
        d["frame_id"] == kf.frame_id
        and d["bbox_2d"] == list(box_a)
        and d["kept"] is True
    )
    assert d["rgb_path"] == kf.rgb_path and d["mask_path"].endswith(".png")


@pytest.mark.adt4
def test_record_observations_is_idempotent_and_new_object_drops_stale_rows(tmp_path):
    from store_helpers import node_id

    store_dir, kf = _store_and_kf(tmp_path)
    box = (1, 2, 4, 6)

    def _obs():
        return [od.Observation(42, "", box, kf.rgb_path, score=0.7, keyframe=kf)]

    w = od.open_sam3_writer(str(store_dir))
    od.record_observations(w, "O1", _obs())
    od.record_observations(w, "O1", _obs())  # same (frame, bbox): no-op
    n = lambda: len(w.reader().observations(node_id("O1")))  # noqa: E731
    assert n() == 1
    od.record_observations(w, "O1", _obs(), new_object=True)  # a reused node index
    assert n() == 1  # stale row replaced, not stacked
    w.close()


@pytest.mark.adt4
def test_record_observations_creates_a_frame_for_a_db_sourced_keyframe(tmp_path):
    """A keyframe loaded from Neo4j whose frame is not in this store still gets an
    observation: a frame row is added from the keyframe's own files and calib."""

    from hydra_python.image_store import ImageStore

    from store_helpers import build_store, node_id

    src = build_store(tmp_path / "src", size=(20, 20))
    src.frame(99)
    src_dir = src.close()
    from hydra_python.image_store import ImageStore as _IS

    s_src = _IS.open(src_dir)
    (kf,) = ks.load_keyframes_from_store(s_src)
    s_src.close()
    kf.frame_id = None
    kf.sensor_id = None

    dst = build_store(tmp_path / "dst", size=(20, 20)).close()
    w = od.open_sam3_writer(str(dst))
    od.record_observations(
        w, "O3", [od.Observation(99, "", (0, 0, 5, 5), kf.rgb_path, keyframe=kf)]
    )
    w.close()

    store = ImageStore.open(dst)
    try:
        (o,) = store.observations(node_id("O3"))
        assert (
            o.frame.timestamp_ns == 99 and o.rgb_path.exists() and o.depth_path.exists()
        )
        assert store.calib(o.frame.sensor_id) is not None
    finally:
        store.close()


@pytest.mark.adt4
def test_upsert_creates_object_in_memory():
    import spark_dsg

    G = spark_dsg.DynamicSceneGraph()
    config = Sam3FrontendConfig()
    touched = od.upsert_objects(
        [_discovered([1.0, 2.0, 3.0])], G, db=None, config=config
    )
    assert len(touched) == 1
    assert touched[0].startswith("O")

    objs = list(G.get_layer(spark_dsg.DsgLayers.OBJECTS).nodes)
    assert len(objs) == 1
    assert objs[0].attributes.name == "chair"
    assert objs[0].attributes.bounding_box.is_valid()


@pytest.mark.adt4
def test_upsert_dedups_nearby_same_label():
    import spark_dsg

    G = spark_dsg.DynamicSceneGraph()
    config = Sam3FrontendConfig(dedup_centroid_dist_m=0.75)
    od.upsert_objects([_discovered([1.0, 2.0, 3.0])], G, db=None, config=config)
    # A near-duplicate (within 0.75 m, same label) should not create a 2nd node.
    od.upsert_objects([_discovered([1.1, 2.0, 3.0])], G, db=None, config=config)
    assert len(list(G.get_layer(spark_dsg.DsgLayers.OBJECTS).nodes)) == 1


@pytest.mark.adt4
def test_upsert_distinct_objects_both_created():
    import spark_dsg

    G = spark_dsg.DynamicSceneGraph()
    config = Sam3FrontendConfig(dedup_centroid_dist_m=0.75)
    od.upsert_objects([_discovered([1.0, 2.0, 3.0])], G, db=None, config=config)
    od.upsert_objects([_discovered([9.0, 9.0, 9.0])], G, db=None, config=config)
    assert len(list(G.get_layer(spark_dsg.DsgLayers.OBJECTS).nodes)) == 2


# ---------------------------------------------------------------------------
# Step 4: openset-framework path (frame_index_select + upsert_via_graph_writer)
# ---------------------------------------------------------------------------


class _FakeFrameIndex:
    """Returns canned query_by_text rows (mimics TrajectoryFrameIndex)."""

    def __init__(self, rows):
        self._rows = rows

    def query_by_text(self, text, k=10, radius_m=None, center_xyz=None):
        return self._rows[:k]


def test_frame_index_select_bridges_path_and_calib(tmp_path):
    """query_by_text rows -> Keyframes: store frame resolved by rgb path, calib shared,
    place_id carried, scores preserved/descending."""
    from hydra_python.image_store import ImageStore
    from store_helpers import build_store

    sb = build_store(tmp_path / "store")
    f1, f2 = sb.frame(111), sb.frame(222)
    store_dir = sb.close()
    store = ImageStore.open(store_dir)
    try:
        p1, p2 = str(store.frame(f1).rgb_path), str(store.frame(f2).rgb_path)
        rows = [
            {"path": p1, "score": 0.9, "place_id": "p1"},
            {"path": p2, "score": 0.8, "place_id": "p2"},
            {"path": None, "score": 0.7, "place_id": "p3"},  # dropped (no path)
            {"path": "/elsewhere/x_rgb.jpg", "score": 0.6},  # dropped (not in store)
        ]
        ranked = od.frame_index_select(_FakeFrameIndex(rows), "car", k=5, store=store)
    finally:
        store.close()

    assert [kf.timestamp_ns for kf, _ in ranked] == [111, 222]
    assert [kf.frame_id for kf, _ in ranked] == [f1, f2]
    assert [s for _, s in ranked] == [0.9, 0.8]
    assert [kf.place_id for kf, _ in ranked] == ["p1", "p2"]
    assert ranked[0][0].calib is ranked[1][0].calib


class _FakeDB:
    """Records execute() calls; returns canned existing-object rows for the
    dedup lookup, empty otherwise."""

    def __init__(self, existing=None):
        self._existing = existing or []
        self.calls = []

    def execute(self, query, **params):
        self.calls.append((query, params))
        if "RETURN o.nodeSymbol AS sym" in query:
            return (
                [
                    {"sym": s, "c": [c[0], c[1], c[2]], "cls": cls, "name": cls}
                    for s, c, cls in self._existing
                ],
                None,
                None,
            )
        return ([], None, None)


class _FakeWriter:
    """Captures add_object calls; shares the FakeDB so follow-up SETs land there."""

    def __init__(self, db):
        self._db = db
        self.added = []
        self._n = 0

    async def add_object(
        self,
        class_label,
        position,
        name=None,
        bbox=None,
        source="agent",
        confidence=0.5,
        embedding=None,
        parent_place=None,
    ):
        self._n += 1
        sym = f"O({self._n})"
        self.added.append(
            dict(
                sym=sym,
                class_label=class_label,
                position=position,
                bbox=bbox,
                source=source,
                confidence=confidence,
                parent_place=parent_place,
            )
        )
        return sym


@pytest.mark.adt4
def test_upsert_via_graph_writer_new_and_dedup(monkeypatch):
    """A far object -> add_object (NEW); a near same-label object -> SET update
    (no add_object). Provenance, color, and observations go through writer._db."""
    import asyncio
    import heracles.graph_interface as gi

    obs_calls, edge_calls = [], []
    monkeypatch.setattr(
        gi, "insert_observations_to_db", lambda db, obs: obs_calls.append(obs)
    )
    monkeypatch.setattr(
        gi, "insert_edges", lambda db, *a, **k: edge_calls.append((a, k))
    )

    db = _FakeDB(existing=[("O5", [1.0, 2.0, 3.0], "chair")])
    writer = _FakeWriter(db)
    config = Sam3FrontendConfig(dedup_centroid_dist_m=0.75)

    near = _discovered([1.1, 2.0, 3.0], label="chair")  # dedups to O5
    far = _discovered([20.0, 20.0, 20.0], label="chair")  # new object
    far.obj.color = (7, 8, 9)

    touched = asyncio.run(
        od.upsert_via_graph_writer([near, far], writer, config=config)
    )

    # exactly one NEW object created via add_object; the other deduped to O5
    assert len(writer.added) == 1
    new = writer.added[0]
    assert new["source"] == "sam3_query"
    assert (
        new["bbox"] is None
    )  # canonical bbox_center/bbox_dim written in follow-up SET
    assert set(touched) == {"O5", new["sym"]}
    # both objects got the canonical bbox the viewer reads (bbox_center/bbox_dim)
    assert sum(1 for q, _ in db.calls if "o.bbox_dim=point" in q) == 2

    # the deduped one issued a center-update SET against O5
    assert any(
        "nodeSymbol: $sym" in q
        and "o.source='sam3_query'" in q
        and p.get("sym") == "O5"
        for q, p in db.calls
    )
    # both objects got the color + attr_type follow-up SET
    assert sum(1 for q, _ in db.calls if "o.color_r=$cr" in q) == 2
    # observations + HAS_OBSERVATION edges flushed once
    assert len(obs_calls) == 1 and len(obs_calls[0]) == 2
    assert len(edge_calls) == 1


@pytest.mark.adt4
def test_record_observations_in_memory_mask_gets_rle_and_a_kept_png(tmp_path):
    """No mask file (no mask_output_dir): the in-memory mask is stored as RLE and as
    the store's ``masks/r<robot>/<obs_id>.png``."""
    from PIL import Image as PILImage

    from hydra_python.image_store import ImageStore

    from store_helpers import node_id

    store_dir, kf = _store_and_kf(tmp_path)
    m = np.zeros((20, 20), bool)
    m[2:7, 1:5] = True
    o = od.Observation(
        42, "", (1, 2, 4, 6), kf.rgb_path, score=0.7, keyframe=kf, mask=m
    )
    w = od.open_sam3_writer(str(store_dir))
    od.record_observations(w, "O1", [o])
    w.close()
    assert o.mask_path and o.mask_path.endswith(f"masks/r0/{o.obs_id}.png")
    assert np.array_equal(np.asarray(PILImage.open(o.mask_path)) > 0, m)
    with ImageStore.open(store_dir) as store:
        (ob,) = store.observations(node_id("O1"))
        assert ob.mask_rle and np.array_equal(store.mask(ob), m)
