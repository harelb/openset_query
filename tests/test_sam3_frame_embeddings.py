"""SigLIP2 keyframe embeddings on Neo4j nodes (frame_embeddings.py), the vectorised
frustum test and the per-frame reprojection prefilter. No GPU, no database."""

import numpy as np
import pytest

from openset_query import frame_embeddings as fe
from openset_query import object_discovery as od
from openset_query.config import Sam3FrontendConfig
from openset_query.embedding_filter import EmbeddingFilter
from openset_query.reprojection import CameraCalib
from openset_query.sam3_segmenter import Detection


class FakeDB:
    """Keyframe nodes {label: {frame_id: props}}; understands the module's queries."""

    def __init__(self, nodes):
        self.nodes = nodes
        self.queries = []

    def execute(self, q, **kw):
        self.queries.append(q)
        if q.startswith("CREATE"):
            return [], None, None
        label = q.split("(a:")[1].split(")")[0].split(" ")[0]
        if q.startswith("MATCH"):
            rows = [
                {"fid": fid, "v": p["siglip2_emb"]}
                for fid, p in self.nodes[label].items()
                if p.get("siglip2_emb") is not None
                and p.get("siglip2_model") == kw["m"]
            ]
            return rows, None, None
        n = 0
        for r in kw["rows"]:
            p = self.nodes[label].get(r["fid"])
            if p is not None:
                p.update(siglip2_emb=r["v"], siglip2_model=kw["m"])
                n += 1
        return [{"n": n}], None, None


class _KF:
    def __init__(self, fid, ns=None):
        self.frame_id, self.cache_ns, self.rgb_path = fid, ns, f"/x/{fid}.jpg"


class FakeEF:
    model_id = "google/siglip2-test"
    frame_key = staticmethod(EmbeddingFilter.frame_key)

    def __init__(self, cache=None):
        self._cache = dict(cache or {})
        self.computed = []

    def embed_images(self, paths, keys=None):
        for k in keys:
            if k not in self._cache:
                v = np.ones(4, np.float32) * (len(self.computed) + 1)
                self._cache[k] = v / np.linalg.norm(v)
                self.computed.append(k)


def _unit(*v):
    v = np.asarray(v, np.float64)
    return v / np.linalg.norm(v)


def test_sync_reads_neo4j_then_npz_then_computes_and_writes_back():
    db = FakeDB(
        {
            "Agent": {
                1: {
                    "siglip2_emb": list(_unit(1, 0, 0, 0) * 3),
                    "siglip2_model": FakeEF.model_id,
                },
                2: {},
            },
            "SubKeyframe": {
                3: {"siglip2_emb": [0, 1, 0, 0], "siglip2_model": "other"},
                4: {},
            },
        }
    )
    ef = FakeEF(cache={"f2": _unit(0, 0, 1, 0)})  # frame 2 is in the npz cache
    kfs = [_KF(1), _KF(2), _KF(3), _KF(4), _KF(9, ns="live/robot1/1")]
    st = fe.sync_pool_embeddings(db, ef, kfs, ef.frame_key)
    assert st == {"neo4j": 1, "npz": 1, "computed": 3, "written": 3}
    assert np.allclose(ef._cache["f1"], [1, 0, 0, 0])  # normalised on load
    assert sorted(ef.computed) == ["f3", "f4", "live/robot1/1:f9"]
    # written back: 2 (npz), 3 (other model), 4 (computed); never the live frame
    assert db.nodes["Agent"][2]["siglip2_model"] == FakeEF.model_id
    assert db.nodes["SubKeyframe"][3]["siglip2_model"] == FakeEF.model_id
    assert np.allclose(db.nodes["SubKeyframe"][4]["siglip2_emb"], ef._cache["f4"])
    assert any("CREATE VECTOR INDEX agent_siglip2_emb" in q for q in db.queries)
    assert any("subkeyframe_siglip2_emb" in q and "cosine" in q for q in db.queries)
    # a second server start reads everything from Neo4j and computes nothing
    ef2 = FakeEF()
    st2 = fe.sync_pool_embeddings(db, ef2, kfs[:4], ef2.frame_key)
    assert st2["neo4j"] == 4 and st2["computed"] == 0 and st2["written"] == 0


def test_sync_without_db_and_db_errors_are_not_fatal():
    ef = FakeEF()
    assert fe.sync_pool_embeddings(None, ef, [_KF(1)], ef.frame_key)["computed"] == 1

    class Broken:
        def execute(self, *a, **k):
            raise RuntimeError("db down")

    ef = FakeEF()
    st = fe.sync_pool_embeddings(Broken(), ef, [_KF(1)], ef.frame_key)
    assert st["computed"] == 1 and st["written"] == 0


def test_query_vector_index_converts_scores():
    class DB:
        def execute(self, q, **kw):
            if "agent_siglip2_emb" in q:
                return [{"fid": 1, "score": 0.9}, {"fid": 2, "score": 0.6}], None, None
            return [{"fid": 3, "score": 0.8}], None, None

    out = fe.query_vector_index(DB(), [1.0, 0.0], k=2)
    assert [f for f, _ in out] == [1, 3] and out[0][1] == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# vectorised frustum test == _in_frustum
# ---------------------------------------------------------------------------


class _PKF:
    def __init__(self, T):
        self.world_T_body = T
        self.calib = CameraCalib(
            fx=300,
            fy=300,
            cx=160,
            cy=120,
            width=320,
            height=240,
            depth_scale=1e-3,
            body_T_sensor=np.eye(4),
        )


def _pose(rng):
    from scipy.spatial.transform import Rotation

    T = np.eye(4)
    T[:3, :3] = Rotation.random(random_state=rng).as_matrix()
    T[:3, 3] = rng.normal(0, 3, 3)
    return T


def test_any_in_frustum_matches_in_frustum():
    pytest.importorskip("scipy")
    rng = np.random.default_rng(1)
    agree = 0
    for _ in range(300):
        kf = _PKF(_pose(rng))
        pts = rng.normal(0, 6, (int(rng.integers(1, 6)), 3))
        for depth in (None, 8.0):
            ref = any(od._in_frustum(p, kf, max_depth_m=depth) for p in pts)
            agree += ref == od._any_in_frustum(pts, kf, depth)
    assert agree == 600
    assert od._any_in_frustum(np.empty((0, 3)), _PKF(np.eye(4))) is False


# ---------------------------------------------------------------------------
# per-frame reprojection prefilter keeps exactly what the full path keeps
# ---------------------------------------------------------------------------


def test_views_from_detections_prefilter_is_exact():
    calib = CameraCalib(
        fx=100,
        fy=100,
        cx=32,
        cy=24,
        width=64,
        height=48,
        depth_scale=1e-3,
        body_T_sensor=np.eye(4),
    )

    class KF:
        world_T_body = np.eye(4)
        timestamp_ns = 1
        rgb_path = ""
        frame_id = 1
        cache_ns = None

    KF.calib = calib
    rng = np.random.default_rng(0)
    depth = rng.uniform(0.5, 20.0, (48, 64))
    depth[rng.random((48, 64)) < 0.3] = np.nan
    rgb = np.zeros((48, 64, 3), np.uint8)
    cfg = Sam3FrontendConfig(min_mask_depth_pixels=40)
    dets = []
    for k in range(25):
        m = np.zeros((48, 64), bool)
        y, x = rng.integers(0, 40), rng.integers(0, 56)
        h, w = rng.integers(2, 24), rng.integers(2, 24)
        m[y : y + h, x : x + w] = True
        dets.append(Detection(label="x", mask=m, box=np.zeros(4), score=0.5))
    got = od._views_from_detections(dets, rgb, depth, KF, cfg)
    from openset_query.reprojection import mask_to_points

    want = [
        d
        for d in dets
        if mask_to_points(d.mask, depth, calib, np.eye(4)).shape[0] >= 40
    ]
    assert len(got) == len(want) and 0 < len(want) < len(dets)
    assert all(
        np.array_equal(g.observations[0].mask, w.mask) for g, w in zip(got, want)
    )
