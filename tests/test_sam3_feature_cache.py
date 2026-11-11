"""SAM3 image-feature cache (feature_cache.py) and its query-server wiring.

No GPU, no database: numpy "features", a fake segmenter split into
``image_features`` / ``segment_features``, a stub Neo4j wrapper.
"""

import json
import os
import threading
import time
import urllib.parse
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from openset_query import feature_cache as fc
from openset_query import query_server as qs
from openset_query.config import Sam3FrontendConfig
from openset_query.embedding_filter import EmbeddingFilter
from openset_query.sam3_segmenter import Detection
from store_helpers import H, W, build_store

image_store = pytest.importorskip("hydra_python.image_store")

MASK_BOX = (20, 10, 40, 30)


class NumpyCodec:
    """FeatureCache codec over numpy arrays ("gpu" and "cpu" are both host)."""

    def __init__(self):
        self.loads = 0

    @staticmethod
    def nbytes(f):
        return int(f.nbytes)

    @staticmethod
    def to_gpu(f):
        return f

    @staticmethod
    def to_cpu(f):
        return f.copy()

    @staticmethod
    def save(path, f, hw):
        with open(path, "wb") as fh:
            np.savez(fh, feats=f, hw=np.asarray(hw))

    def load(self, path, to_gpu=True):
        self.loads += 1
        d = np.load(path)
        return d["feats"], tuple(int(v) for v in d["hw"])


def _feat(v, n=100):
    return np.full(n, float(v), dtype=np.float32)  # 400 bytes


def _cache(tmp_path, gpu=1000, cpu=1000, **kw):
    return fc.FeatureCache(
        str(tmp_path / "feats"),
        NumpyCodec(),
        gpu_budget_bytes=gpu,
        cpu_budget_bytes=cpu,
        **kw,
    )


# ---------------------------------------------------------------------------
# FeatureCache
# ---------------------------------------------------------------------------


def test_put_get_and_lru_tiers(tmp_path):
    c = _cache(tmp_path, gpu=800, cpu=400)  # 2 frames on the "gpu", 1 in RAM
    for i in range(4):
        c.put(f"f{i}", _feat(i), (H, W))
    c.flush()
    s = c.stats()
    assert (s["gpu"], s["cpu"], s["disk"]) == (2, 1, 4)
    assert c.tier("f3") == "gpu" and c.tier("f1") == "cpu" and c.tier("f0") == "disk"
    f, hw, tier = c.get("f0")  # from disk -> promoted
    assert tier == "disk" and hw == (H, W) and f[0] == 0.0
    assert c.tier("f0") == "gpu"
    f, _, tier = c.get("f0")
    assert tier == "gpu"
    assert c.get("nope") is None and not c.has("nope") and c.has("f2")


def test_disk_persists_across_instances_and_keys_are_safe(tmp_path):
    c = _cache(tmp_path)
    key = "live/robot1/1760000000:f12"
    c.put(key, _feat(7), (3, 4))
    c.put("f1", _feat(1), (3, 4))
    c.close()
    names = os.listdir(tmp_path / "feats")
    assert len(names) == 2 and all(
        "/" not in n and n.endswith(".safetensors") for n in names
    )
    c2 = _cache(tmp_path)
    assert c2.on_disk(key) and c2.tier(key) == "disk"
    f, hw, tier = c2.get(key)
    assert tier == "disk" and f[0] == 7.0 and hw == (3, 4)


def test_unreadable_file_is_dropped(tmp_path):
    c = _cache(tmp_path)
    c.put("f1", _feat(1), (1, 1))
    c.close()
    with open(c.path("f1"), "wb") as fh:
        fh.write(b"garbage")
    c2 = _cache(tmp_path)
    assert c2.get("f1") is None and not c2.has("f1")


def test_disk_cap_and_memory_only(tmp_path):
    c = _cache(tmp_path, max_disk_bytes=0)
    c.put("f1", _feat(1), (1, 1))
    c.flush()
    assert c.stats()["disk"] == 0 and c.tier("f1") == "gpu"
    m = fc.FeatureCache(None, NumpyCodec(), gpu_budget_bytes=1000)
    m.put("f1", _feat(1), (1, 1))
    assert m.get("f1")[2] == "gpu" and m.stats()["disk"] == 0


def test_preload_fills_gpu_tier_from_disk(tmp_path):
    c = _cache(tmp_path)
    for i in range(3):
        c.put(f"f{i}", _feat(i), (1, 1))
    c.close()
    c2 = _cache(tmp_path, gpu=800)
    assert c2.preload(["f0", "f1", "f2"]) == 2  # room for two
    assert c2.tier("f0") == "gpu" and c2.tier("f1") == "gpu" and c2.tier("f2") == "disk"


def test_cache_stats_json():
    s = fc.CacheStats()
    for t in ("gpu", "gpu", "disk", "miss"):
        s.add(t, 0.01)
    j = s.to_json()
    assert (
        j["frames"] == 4 and j["hits"] == 3 and j["hit_rate"] == 0.75 and j["miss"] == 1
    )
    assert fc.CacheStats().to_json()["hit_rate"] is None


# ---------------------------------------------------------------------------
# GpuGate
# ---------------------------------------------------------------------------


def test_gate_lets_a_waiting_query_go_before_background_work():
    gate = fc.GpuGate()
    order = []
    started = threading.Event()

    def background():
        for i in range(20):
            with gate.background():
                started.set()
                order.append("b")
                time.sleep(0.005)

    t = threading.Thread(target=background)
    t.start()
    started.wait()
    with gate:
        assert gate.locked()
        order.append("q")
    t.join()
    assert not gate.locked()
    # the query got in within one background unit, not after all twenty
    assert order.index("q") <= 3


# ---------------------------------------------------------------------------
# CachedSegmenter / builder
# ---------------------------------------------------------------------------


def _mask():
    m = np.zeros((H, W), dtype=bool)
    x0, y0, x1, y1 = MASK_BOX
    m[y0:y1, x0:x1] = True
    return m


class SplitSegmenter:
    """Fake SAM3 split: the 'trunk' is the image mean, the 'head' a fixed detection
    whose score depends on the features (so cached == uncached is checkable)."""

    def __init__(self):
        self.trunk = 0
        self.head = 0
        self.lock = threading.Lock()

    def feature_version(self):
        return "fake1"

    def image_features(self, image):
        with self.lock:
            self.trunk += 1
        return np.asarray(image, dtype=np.float32).mean(axis=(0, 1)), image.shape[:2]

    def segment_features(self, feats, hw, prompts):
        self.head += 1
        score = 0.5 + float(np.asarray(feats).sum()) / 1e4
        return [
            Detection(
                label=prompts[0], mask=_mask(), box=np.array(MASK_BOX), score=score
            )
        ]

    def segment(self, image, prompts):
        return self.segment_features(*self.image_features(image), prompts)


class _KF:
    def __init__(self, i):
        self.frame_id, self.cache_ns, self.rgb_path = i, None, f"/x/{i}.jpg"


def _rgb(i):
    return np.full((4, 5, 3), i, dtype=np.uint8)


def test_cached_segmenter_matches_uncached_and_counts_hits(tmp_path):
    seg = SplitSegmenter()
    cs = fc.CachedSegmenter(seg, _cache(tmp_path, gpu=10_000), qs.frame_key)
    kf = _KF(3)
    a = cs.segment_keyframe(kf, _rgb(3), ["mug"])
    b = cs.segment_keyframe(kf, _rgb(3), ["mug"])
    ref = seg.segment(_rgb(3), ["mug"])
    assert a[0].score == b[0].score == ref[0].score
    assert cs.stats.to_json()["miss"] == 1 and cs.stats.to_json()["gpu"] == 1
    assert seg.trunk == 2  # one miss + the reference call; the hit skipped the trunk


def test_builder_fills_the_pool_and_skips_cached(tmp_path):
    seg = SplitSegmenter()
    cache = _cache(tmp_path, gpu=10_000)
    cs = fc.CachedSegmenter(seg, cache, qs.frame_key)
    b = fc.FeatureCacheBuilder(
        cs, fc.GpuGate(), qs.frame_key, read_rgb=lambda kf: _rgb(kf.frame_id)
    )
    kfs = [_KF(i) for i in range(5)]
    assert b.enqueue(kfs) == 5
    assert b.wait(5)
    assert (
        seg.trunk == 5 and b.built == 5 and all(cache.has(qs.frame_key(k)) for k in kfs)
    )
    assert b.enqueue(kfs + [_KF(9)]) == 1  # only the new frame
    assert b.wait(5) and seg.trunk == 6
    st = b.status()
    assert st["built"] == 6 and st["queued"] == 0 and not st["building"]


def test_builder_survives_unreadable_frames(tmp_path):
    seg = SplitSegmenter()
    cs = fc.CachedSegmenter(seg, _cache(tmp_path), qs.frame_key)

    def read(kf):
        if kf.frame_id == 1:
            raise FileNotFoundError(kf.rgb_path)
        return _rgb(kf.frame_id)

    b = fc.FeatureCacheBuilder(cs, fc.GpuGate(), qs.frame_key, read_rgb=read)
    b.enqueue([_KF(0), _KF(1), _KF(2)])
    assert b.wait(5) and b.built == 2 and b.failed == 1


# ---------------------------------------------------------------------------
# query server wiring
# ---------------------------------------------------------------------------


class RankEF:
    frame_key = staticmethod(EmbeddingFilter.frame_key)
    model_id = "fake/siglip2"

    def embed_images(self, paths, keys=None):
        return None

    def rank(self, pool, prompts):
        return sorted(
            ((kf, 1e-4 * kf.timestamp_ns) for kf in pool), key=lambda t: -t[1]
        )


class StubDB:
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, q, **kw):
        return [], None, None


def _pool(tmp_path, n=3):
    from hydra_python.image_store import ImageStore

    from openset_query import keyframe_store as ks

    b = build_store(tmp_path / "prior" / "image_store")
    for i in range(n):
        T = np.eye(4)
        T[0, 3] = 0.05 * i
        b.frame(
            1000 + i,
            color=(40 * i, 200, 30),
            anchor=image_store.make_node_id("a", i),
            pose=T,
        )
    d = b.close()
    with ImageStore.open(d) as s:
        return d, ks.load_keyframes_from_store(s)


@pytest.fixture
def engine(tmp_path):
    store_dir, kfs = _pool(tmp_path)
    cfg = Sam3FrontendConfig(device="cpu", conf_threshold=0.3)
    cache = fc.FeatureCache(
        str(tmp_path / "feats"), NumpyCodec(), gpu_budget_bytes=10_000
    )
    eng = qs.QueryEngine(
        "bolt://unused",
        ("u", "p"),
        cfg,
        image_store=[str(store_dir)],
        segmenter=SplitSegmenter(),
        embedding_filter=RankEF(),
        keyframes=kfs,
        feature_cache=cache,
        build_feature_cache=False,
    )
    eng._Neo4jWrapper = StubDB
    return eng


@pytest.mark.adt4
def test_query_reports_cache_hits_and_results_do_not_change(engine):
    r1 = engine.query("mug", mode="topk", dry_run=True, render="none")
    assert r1["cache"]["miss"] == 3 and r1["cache"]["hit_rate"] == 0.0
    r2 = engine.query("mug", mode="topk", dry_run=True, render="none")
    assert r2["cache"]["hits"] == 3 and r2["timings"]["cache_hit"] == 1.0

    def strip(r):
        return [(c["score"], c["centroid"], c["num_views"]) for c in r["candidates"]]

    assert strip(r1) == strip(r2)
    trunk = engine.segmenter.trunk
    engine.cached = None  # uncached path gives the same candidates
    r3 = engine.query("mug", mode="topk", dry_run=True, render="none")
    assert strip(r3) == strip(r1) and engine.segmenter.trunk == trunk + 3
    assert "cache" not in r3


@pytest.mark.adt4
def test_background_build_then_warm_query(tmp_path, engine):
    assert engine.builder.enqueue(engine.keyframes) == 3
    assert engine.builder.wait(5)
    st = engine.feature_cache_status()
    assert st["pool_cached"] == 3 and st["pool"] == 3 and st["built"] == 3
    r = engine.query("mug", mode="frustum", dry_run=True, render="none", patience=3)
    assert r["cache"]["miss"] == 0 and r["cache"]["frames"] >= 1


@pytest.mark.adt4
def test_reload_queues_new_frames(engine, monkeypatch):
    engine.builder.enqueue(engine.keyframes)
    engine.builder.wait(5)
    extra = list(engine.keyframes)
    import copy

    k = copy.copy(extra[0])
    k.frame_id, k.cache_ns = 77, "live/robot1/1"
    monkeypatch.setattr(engine, "_load_pool", lambda db: extra + [k])
    engine.reload_keyframes()
    assert engine.builder.wait(5)
    assert engine.cached.cache.has("live/robot1/1:f77")


@pytest.mark.adt4
def test_render_modes_and_cluster_fields(engine):
    r = engine.query("mug", mode="topk", dry_run=True, render="none", max_views=2)
    (c,) = r["candidates"]
    assert c["num_views"] == 3 and c["num_frames"] == 3 and len(c["frame_keys"]) == 3
    assert len(c["views"]) == 2 and all(v["image_uri"] is None for v in c["views"])
    assert c["image_uri"] is None and [v["view"] for v in c["views"]] == [0, 1]
    assert c["views"][0]["frame_key"] == c["frame_keys"][0]
    b = engine.query("mug", mode="topk", dry_run=True, render="best", max_views=2)
    v0, v1 = b["candidates"][0]["views"]
    assert v0["image_uri"].startswith("data:image/jpeg") and v1["image_uri"] is None
    with pytest.raises(ValueError):
        engine.query("mug", dry_run=True, render="some")


@pytest.mark.adt4
def test_frame_image_draws_a_pending_view(engine):
    r = engine.query("mug", mode="topk", dry_run=True, render="none")
    pid, c = r["pending_id"], r["candidates"][0]
    out = engine.frame_image(None, max_dim=64, pending_id=pid, index=0, view=1)
    assert out["annotated"] and out["frame_key"] == c["views"][1]["frame_key"]
    assert out["uri"].startswith("data:image/jpeg")
    with pytest.raises(IndexError):
        engine.frame_image(None, pending_id=pid, index=5)
    with pytest.raises(qs.PendingNotFound):
        engine.frame_image(None, pending_id="nope", index=0)
    v = c["views"][0]
    out = engine.frame_image(
        v["frame_key"],
        mask_rle=v["mask_rle"],
        mask_shape=v["mask_shape"],
        bbox=v["bbox_2d"],
    )
    assert out["annotated"]
    assert not engine.frame_image(v["frame_key"])["annotated"]


@pytest.fixture
def http(engine):
    server = ThreadingHTTPServer(("127.0.0.1", 0), qs.make_handler(engine))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield engine, f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def _req(url, path, body=None):
    import urllib.error

    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        url + path,
        data=data,
        method="GET" if body is None else "POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


@pytest.mark.adt4
def test_http_render_frame_and_health(http):
    engine, url = http
    code, q = _req(
        url,
        "/query",
        {
            "prompt": "mug",
            "mode": "topk",
            "dry_run": True,
            "render": "none",
            "max_views": 3,
        },
    )
    assert code == 200 and q["cache"]["frames"] == 3 and q["association"] == "cluster"
    c = q["candidates"][0]
    assert c["num_frames"] == 3 and c["views"][0]["image_uri"] is None
    code, f = _req(
        url, f"/frame?pending_id={q['pending_id']}&index=0&view=2&max_dim=64"
    )
    assert (
        code == 200 and f["annotated"] and f["frame_key"] == c["views"][2]["frame_key"]
    )
    v = c["views"][0]
    qstr = urllib.parse.urlencode(
        {
            "key": v["frame_key"],
            "mask": v["mask_rle"],
            "mask_shape": ",".join(map(str, v["mask_shape"])),
            "bbox": ",".join(map(str, v["bbox_2d"])),
        }
    )
    code, f = _req(url, "/frame?" + qstr)
    assert code == 200 and f["annotated"]
    assert _req(url, "/frame?pending_id=nope&index=0")[0] == 404
    assert _req(url, f"/frame?pending_id={q['pending_id']}")[0] == 400
    assert _req(url, f"/frame?pending_id={q['pending_id']}&index=9")[0] == 404
    assert _req(url, "/query", {"prompt": "mug", "render": "bad"})[0] == 400
    assert _req(url, "/query", {"prompt": "mug", "association": "bad"})[0] == 400
    code, cq = _req(
        url,
        "/query",
        {
            "prompt": "mug",
            "mode": "topk",
            "dry_run": True,
            "association": "centroid",
            "render": "none",
        },
    )
    assert code == 200 and cq["association"] == "centroid"
    code, h = _req(url, "/health")
    assert code == 200 and h["feature_cache"]["pool"] == 3 and h["busy"] is False
