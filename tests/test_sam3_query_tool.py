"""SAM3 query server additions for the console "Search images" tool: pool sources,
per-request SAM3 confidence, SigLIP2-only /rank, raw /frame, richer /health.

No GPU, no database: fake segmenter + embedding filter, and a stub Neo4j wrapper
(the dry-run path only reads existing objects for the merge hint).
"""

import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from openset_query import query_server as qs
from openset_query.config import Sam3FrontendConfig
from openset_query.embedding_filter import EmbeddingFilter
from openset_query.sam3_segmenter import Detection
from store_helpers import H, W, build_store

image_store = pytest.importorskip("hydra_python.image_store")

MASK_BOX = (20, 10, 40, 30)


def _mask():
    m = np.zeros((H, W), dtype=bool)
    x0, y0, x1, y1 = MASK_BOX
    m[y0:y1, x0:x1] = True
    return m


class ScoredSegmenter:
    """One detection per frame with a fixed score; records the threshold it ran at."""

    def __init__(self, config, score=0.4):
        self.config, self.score, self.thresholds = config, score, []

    def set_conf_threshold(self, t):
        self.config.conf_threshold = float(t)

    def segment(self, image, prompts):
        self.thresholds.append(self.config.conf_threshold)
        if self.score < self.config.conf_threshold:
            return []
        return [
            Detection(
                label=prompts[0], mask=_mask(), box=np.array(MASK_BOX), score=self.score
            )
        ]


class FixedSegmenter:
    """No ``set_conf_threshold``: always reports its one detection."""

    def __init__(self, score):
        self.score = score

    def segment(self, image, prompts):
        return [
            Detection(
                label=prompts[0], mask=_mask(), box=np.array(MASK_BOX), score=self.score
            )
        ]


class RankEF:
    """Ranks by capture time (later frame = better match)."""

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


def _pool(tmp_path):
    prior = build_store(tmp_path / "prior" / "image_store")
    for i in range(3):
        T = np.eye(4)
        T[0, 3] = 0.05 * i
        prior.frame(
            1000 + i,
            color=(40 * i, 200, 30),
            anchor=image_store.make_node_id("a", i),
            pose=T,
        )
    prior_dir = prior.close()
    live_root = tmp_path / "live"
    lb = build_store(live_root / "robot1" / "1760000000" / "image_store")
    lb.frame(5000, color=(10, 10, 200), anchor=image_store.make_node_id("a", 9))
    lb.close()
    from hydra_python.image_store import ImageStore

    from openset_query import keyframe_store as ks

    with ImageStore.open(prior_dir) as s:
        kfs = ks.load_keyframes_from_store(s)
    return prior_dir, kfs + qs.load_live_keyframes(live_root), str(live_root)


def write_live_state(path, live_root, relocalized=True, T=(0, 0, 0, 1, 0, 0, 0)):
    """The console's live state: robot1's current store, relocalized with ``T``."""
    import time

    store = f"{live_root}/robot1/1760000000/image_store"
    state = {
        "version": 1,
        "written_ns": time.time_ns(),
        "live_store_root": str(live_root),
        "robots": {
            "robot1": {
                "store_id": "1760000000",
                "path": store,
                "relocalized": relocalized,
                "reason": None if relocalized else "relocalizing: 2 / 5",
                "frame_align": {
                    "ok": True,
                    "parent": "map",
                    "child": "robot1/map",
                    "T": list(T),
                },
            }
        },
    }
    with open(path, "w") as f:
        json.dump(state, f)
    return str(path)


@pytest.fixture
def engine(tmp_path):
    prior_dir, kfs, live_root = _pool(tmp_path)
    cfg = Sam3FrontendConfig(device="cpu", conf_threshold=0.5)
    eng = qs.QueryEngine(
        "bolt://unused",
        ("u", "p"),
        cfg,
        image_store=[str(prior_dir)],
        segmenter=ScoredSegmenter(cfg),
        embedding_filter=RankEF(),
        keyframes=kfs,
        live_state=write_live_state(tmp_path / "console_live.json", live_root),
    )
    eng._Neo4jWrapper = StubDB
    return eng


@pytest.mark.adt4
def test_sources_and_filter_pool(engine):
    src = engine.sources()
    assert src == [
        {"id": "prior", "keyframes": 3},
        {"id": "live/robot1/1760000000", "keyframes": 1},
    ]
    kfs = engine.keyframes
    assert len(qs.filter_pool(kfs)) == 4 and len(qs.filter_pool(kfs, "all")) == 4
    assert {qs.keyframe_source(k) for k in qs.filter_pool(kfs, "prior")} == {"prior"}
    assert len(qs.filter_pool(kfs, "live")) == 1
    assert len(qs.filter_pool(kfs, "live/robot1")) == 1  # prefix of a mirror id
    assert len(qs.filter_pool(kfs, "live/robot1/1760000000")) == 1
    with pytest.raises(qs.UnknownSource):
        qs.filter_pool(kfs, "live/nobody")
    newest = max(k.timestamp_ns for k in kfs if qs.keyframe_source(k) == "prior")
    assert len(qs.filter_pool(kfs, "prior", min_ts=newest)) == 1


@pytest.mark.adt4
def test_conf_override_is_per_request(engine):
    seg = engine.segmenter
    r = engine.query("mug", mode="topk", dry_run=True)  # 0.4 < server conf 0.5
    assert r["candidates"] == [] and r["conf"] == 0.5
    r = engine.query("mug", mode="topk", dry_run=True, conf=0.3, source="prior")
    assert len(r["candidates"]) == 1 and r["conf"] == 0.3 and r["source"] == "prior"
    assert r["pool"] == 3 and seg.thresholds[-1] == 0.3
    assert engine.config.conf_threshold == 0.5  # restored
    c = r["candidates"][0]
    assert {v["source"] for v in c["views"]} == {"prior"}
    assert all(v["frame_key"].startswith("f") for v in c["views"])
    # a segmenter that cannot change its own threshold is still filtered
    engine.segmenter = FixedSegmenter(score=0.6)
    assert engine.query("mug", mode="topk", dry_run=True, conf=0.7)["candidates"] == []
    assert len(engine.query("mug", mode="topk", dry_run=True)["candidates"]) == 1


@pytest.mark.adt4
def test_max_views_caps_rendered_views(engine):
    r = engine.query(
        "mug", mode="topk", dry_run=True, conf=0.3, source="prior", max_views=1
    )
    (c,) = r["candidates"]
    assert c["num_views"] == 3 and len(c["views"]) == 1


@pytest.mark.adt4
def test_rank_returns_scores_and_thumbs_without_sam3(engine):
    seg = engine.segmenter
    r = engine.rank("mug", top_k=2)
    assert seg.thresholds == []  # SAM3 never ran
    assert r["pool"] == 4 and len(r["frames"]) == 2
    f0, f1 = r["frames"]
    assert f0["rank"] == 0 and f0["score"] >= f1["score"]
    assert f0["source"].startswith("live/") and f0["frame_key"].startswith(
        "live/robot1"
    )
    assert f0["thumb_uri"].startswith("data:image/jpeg;base64,")
    assert len(f0["camera_position"]) == 3 and set(r["timings"]) == {
        "embed_rank",
        "render",
    }
    r = engine.rank("mug", top_k=10, source="prior", thumbs=False)
    assert len(r["frames"]) == 3 and all(f["thumb_uri"] is None for f in r["frames"])
    json.dumps(r)


@pytest.mark.adt4
def test_frame_image(engine):
    key = qs.frame_key(engine.keyframes[0])
    out = engine.frame_image(key, max_dim=32)
    assert out["uri"].startswith("data:image/jpeg") and (
        out["width"],
        out["height"],
    ) == (W, H)
    with pytest.raises(KeyError):
        engine.frame_image("f999999")


@pytest.fixture
def http(engine):
    server = ThreadingHTTPServer(("127.0.0.1", 0), qs.make_handler(engine))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield engine, f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def _req(url, path, body=None):
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
def test_http_routes(http):
    engine, url = http
    code, h = _req(url, "/health")
    assert code == 200 and h["keyframes"] == 4 and h["conf_threshold"] == 0.5
    assert h["sources"][0] == {"id": "prior", "keyframes": 3} and h["busy"] is False
    assert h["embed_model"] == "fake/siglip2"

    code, r = _req(url, "/rank", {"prompt": "mug", "top_k": 3, "source": "prior"})
    assert code == 200 and len(r["frames"]) == 3
    assert _req(url, "/rank", {"prompt": ""})[0] == 400
    assert _req(url, "/rank", {"prompt": "mug", "source": "live/nobody"})[0] == 400

    code, f = _req(url, "/frame?key=" + urllib.parse.quote(r["frames"][0]["frame_key"]))
    assert code == 200 and f["uri"].startswith("data:image/jpeg")
    assert _req(url, "/frame?key=nope")[0] == 404 and _req(url, "/frame")[0] == 400

    code, q = _req(
        url,
        "/query",
        {
            "prompt": "mug",
            "mode": "topk",
            "dry_run": True,
            "conf": 0.3,
            "source": "prior",
            "max_views": 2,
        },
    )
    assert (
        code == 200
        and len(q["candidates"]) == 1
        and len(q["candidates"][0]["views"]) == 2
    )
    assert q["pending_id"] and q["conf"] == 0.3
    assert _req(url, "/query", {"prompt": "mug", "conf": 1.5})[0] == 400
    assert _req(url, "/query", {"prompt": "mug", "mode": "nope"})[0] == 400
    assert (
        _req(
            url, "/query", {"prompt": "mug", "source": "live/nobody", "dry_run": True}
        )[0]
        == 400
    )
