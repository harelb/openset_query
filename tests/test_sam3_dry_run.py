"""SAM3 query server dry-run / commit / discard (console "find -> show -> confirm").

Unit tests need no GPU and no database (fake segmenter + embedding filter). The DB
tests write through the real ``upsert_objects`` / ``admit_object`` path and need a
THROWAWAY Neo4j (``NEO4J_URI`` / ``NEO4J_AUTH``, same convention as
tests/test_sim_live_tier_db.py); they only delete the objects they create.
"""

import io
import json
import os
import threading
import urllib.error
import uuid
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from openset_query import keyframe_store as ks
from openset_query import object_discovery as od
from openset_query import query_client as qc
from openset_query import query_server as qs
from openset_query.config import Sam3FrontendConfig
from openset_query.embedding_filter import EmbeddingFilter
from openset_query.reprojection import Object3D
from openset_query.sam3_segmenter import Detection
from store_helpers import H, W, build_store

image_store = pytest.importorskip("hydra_python.image_store")
from hydra_python.image_store import ImageStore, rle_decode  # noqa: E402

MASK_BOX = (20, 10, 40, 30)  # x0, y0, x1, y1 (exclusive end)


def _mask():
    m = np.zeros((H, W), dtype=bool)
    x0, y0, x1, y1 = MASK_BOX
    m[y0:y1, x0:x1] = True
    return m


class FakeSegmenter:
    """One detection of whatever was prompted, the same box in every frame."""

    def __init__(self):
        self.calls = 0

    def segment(self, image, prompts):
        self.calls += 1
        return [
            Detection(label=prompts[0], mask=_mask(), box=np.array(MASK_BOX), score=0.8)
        ]


class FakeEF:
    frame_key = staticmethod(EmbeddingFilter.frame_key)

    def embed_images(self, paths, keys=None):
        return None

    def rank(self, pool, prompts):
        return [(kf, 1.0) for kf in pool]


def _store(tmp_path, n=3):
    b = build_store(tmp_path / "run" / "image_store")
    for i in range(n):
        T = np.eye(4)
        T[0, 3] = 0.05 * i  # small baseline: all views fuse into one object
        b.frame(
            1000 + i,
            color=(40 * i, 200, 30),
            anchor=image_store.make_node_id("a", i),
            pose=T,
        )
    return b.close()


def _keyframes(store_dir):
    with ImageStore.open(store_dir) as s:
        return ks.load_keyframes_from_store(s)


# ---------------------------------------------------------------------------
# unit
# ---------------------------------------------------------------------------


def test_pending_cache_ttl_and_bound():
    now = [0.0]
    c = qs.PendingCache(ttl_s=10, max_entries=2, clock=lambda: now[0])
    a = c.put({"x": 1})
    b = c.put({"x": 2})
    assert c.get(a)["x"] == 1 and len(c) == 2
    c.put({"x": 3})  # bound: the oldest goes
    with pytest.raises(qs.PendingNotFound):
        c.get(a)
    now[0] = 11.0  # ttl
    with pytest.raises(qs.PendingNotFound):
        c.get(b)
    assert len(c) == 0
    d = c.put({"x": 4})
    assert c.discard(d) and not c.discard(d)


def test_observation_mask_compaction_roundtrip():
    o = od.Observation(timestamp_ns=1, mask_file="", bbox_2d=(0, 0, 1, 1), mask=_mask())
    o.compact_mask()
    assert o.mask is None and o.mask_shape == (H, W)
    assert np.array_equal(o.mask_array(), _mask())


def test_build_candidates_payload(tmp_path):
    store_dir = _store(tmp_path, 1)
    kf = _keyframes(store_dir)[0]
    obs = od.Observation(
        timestamp_ns=kf.timestamp_ns,
        mask_file="",
        bbox_2d=(20, 10, 39, 29),
        rgb_path=kf.rgb_path,
        score=0.8,
        keyframe=kf,
        mask=_mask(),
    )
    obj = Object3D(
        label="mug",
        centroid=np.array([1.0, 2.0, 0.5]),
        bbox_min=np.array([0.9, 1.9, 0.4]),
        bbox_max=np.array([1.1, 2.1, 0.6]),
        num_points=400,
        score=0.8,
    )
    existing = [
        ("O7", np.array([1.2, 2.0, 0.5]), "mug"),
        ("O8", np.array([1.0, 2.0, 0.5]), "chair"),
    ]
    (c,) = qs.build_candidates(
        [od.DiscoveredObject(obj=obj, observations=[obs])], existing
    )
    assert c["index"] == 0 and c["label"] == "mug" and c["num_points"] == 400
    assert c["centroid"] == [1.0, 2.0, 0.5] and c["bbox_max"] == [1.1, 2.1, 0.6]
    assert c["would_merge_into"] == {"symbol": "O7", "distance_m": 0.2}
    (v,) = c["views"]
    assert v["frame_id"] == kf.frame_id and v["timestamp_ns"] == kf.timestamp_ns
    assert v["bbox_2d"] == [20, 10, 39, 29] and v["mask_shape"] == [H, W]
    assert np.array_equal(rle_decode(v["mask_rle"], (H, W)), _mask())
    assert (
        v["image_uri"].startswith("data:image/jpeg;base64,")
        and c["image_uri"] == v["image_uri"]
    )
    json.dumps(c)  # JSON-serializable as is


def test_claim_object_index_falls_back_on_a_stub_db():
    class Stub:
        def execute(self, q, **kw):
            return [], None, None

    assert od.claim_object_index(Stub(), 7, "t") == 7


def test_live_store_keyframes_are_tagged(tmp_path):
    live = tmp_path / "live"
    b = build_store(live / "robot1" / "1760000000" / "image_store")
    b.frame(5, anchor=image_store.make_node_id("a", 0))
    b.close()
    assert qs.find_live_stores(live) == [
        str(live / "robot1" / "1760000000" / "image_store")
    ]
    (kf,) = qs.load_live_keyframes(live)
    assert (
        kf.store_dir.endswith("image_store") and kf.cache_ns == "live/robot1/1760000000"
    )
    assert EmbeddingFilter.frame_key(kf) == f"live/robot1/1760000000:f{kf.frame_id}"
    assert (
        qs.find_live_stores(None) == [] and qs.find_live_stores(tmp_path / "nope") == []
    )


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_client_dry_run_commit_discard_payloads(monkeypatch):
    sent = []

    def fake(req, timeout=0):
        sent.append((req.full_url, json.loads(req.data)))
        return _Resp(json.dumps({"ok": True}).encode())

    monkeypatch.setattr(qc.urllib.request, "urlopen", fake)
    qc.sam3_query("mug", base_url="http://x:1", dry_run=True)
    qc.sam3_query("mug", base_url="http://x:1")
    qc.sam3_commit("abc", 2, base_url="http://x:1")
    qc.sam3_discard("abc", base_url="http://x:1")
    assert sent[0][1]["dry_run"] is True and "dry_run" not in sent[1][1]
    assert sent[2] == ("http://x:1/commit", {"pending_id": "abc", "index": 2})
    assert sent[3] == ("http://x:1/discard", {"pending_id": "abc"})


def test_client_commit_404_carries_status(monkeypatch):
    def fake(req, timeout=0):
        raise urllib.error.HTTPError(
            req.full_url, 404, "nf", {}, io.BytesIO(b'{"error": "x"}')
        )

    monkeypatch.setattr(qc.urllib.request, "urlopen", fake)
    with pytest.raises(qc.Sam3QueryError) as e:
        qc.sam3_commit("gone", 0, base_url="http://x:1")
    assert e.value.status == 404


class _EngineStub:
    keyframes = []
    admission_policy = None
    live_store_root = None

    def __init__(self):
        self.pending = qs.PendingCache()
        self.seen = []

    def query(self, prompt, **kw):
        self.seen.append(kw)
        return {"prompt": prompt, "objects": 0, "symbols": [], "elapsed_s": 0.0}

    def commit(self, pid, index):
        if pid != "p1":
            raise qs.PendingNotFound(pid)
        if index > 0:
            raise IndexError("index 3 out of range")
        return {"symbol": "O1", "created": True}

    def discard(self, pid):
        return pid == "p1"


@pytest.fixture
def http_stub():
    engine = _EngineStub()
    server = ThreadingHTTPServer(("127.0.0.1", 0), qs.make_handler(engine))
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield engine, f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_http_routes(http_stub):
    engine, url = http_stub
    qc.sam3_query("mug", base_url=url, dry_run=True)
    assert engine.seen[-1]["dry_run"] is True
    qc.sam3_query("mug", base_url=url)
    assert engine.seen[-1]["dry_run"] is False
    assert qc.sam3_commit("p1", 0, base_url=url) == {"symbol": "O1", "created": True}
    with pytest.raises(qc.Sam3QueryError) as e:
        qc.sam3_commit("nope", 0, base_url=url)
    assert e.value.status == 404
    with pytest.raises(qc.Sam3QueryError) as e:
        qc.sam3_commit("p1", 3, base_url=url)
    assert e.value.status == 400
    assert qc.sam3_discard("p1", base_url=url) == {"discarded": True}


# ---------------------------------------------------------------------------
# Neo4j (throwaway)
# ---------------------------------------------------------------------------

_NEO4J_URI = os.environ.get("NEO4J_URI", "")
needs_db = pytest.mark.skipif(
    not _NEO4J_URI, reason="NEO4J_URI not set (throwaway DB required)"
)
LABEL = "dry run test mug"
LABEL2 = "dry run test kettle"


def _auth():
    user, password = os.environ.get("NEO4J_AUTH", "neo4j/neo4j").split("/", 1)
    return user, password


@pytest.fixture
def db():
    from heracles.query_interface import Neo4jWrapper

    wrapper = Neo4jWrapper(_NEO4J_URI, _auth(), atomic_queries=True)
    wrapper.connect()
    _wipe(wrapper)
    try:
        yield wrapper
    finally:
        _wipe(wrapper)
        wrapper.close()


def _wipe(db):
    db.execute(
        "MATCH (o:Object)-[:HAS_OBSERVATION]->(x:Observation) WHERE o.class IN $c "
        "DETACH DELETE x",
        c=[LABEL, LABEL2],
    )
    db.execute(
        "MATCH (o:Object) WHERE o.class IN $c DETACH DELETE o", c=[LABEL, LABEL2]
    )


def _versions(db):
    from openset_query.heracles_compat import read_map_version, read_state_version

    return read_state_version(db), read_map_version(db)


def _engine(tmp_path):
    store_dir = _store(tmp_path)
    eng = qs.QueryEngine(
        _NEO4J_URI,
        _auth(),
        Sam3FrontendConfig(device="cpu"),
        image_store=[str(store_dir)],
        segmenter=FakeSegmenter(),
        embedding_filter=FakeEF(),
        keyframes=_keyframes(store_dir),
    )
    return eng, store_dir


def _objects(db, label):
    rows, _, _ = db.execute(
        "MATCH (o:Object {class: $c}) RETURN o.nodeSymbol AS s, "
        "o.admission_status AS st, o.admission_policy_version AS pv",
        c=label,
    )
    return rows


@needs_db
def test_dry_run_writes_nothing_then_commit_writes_admits_and_bumps(db, tmp_path):
    eng, store_dir = _engine(tmp_path)
    v0 = _versions(db)
    with ImageStore.open(store_dir) as s:
        obs0 = s.info()[0]["observations"]

    r = eng.query(LABEL, mode="topk", dry_run=True)
    assert r["dry_run"] and r["pending_id"] and r["symbols"] == []
    assert len(r["candidates"]) == 1
    c = r["candidates"][0]
    assert c["label"] == LABEL and c["num_views"] == 3 and len(c["views"]) == 3
    assert all(v["image_uri"].startswith("data:image/jpeg") for v in c["views"])
    assert np.array_equal(rle_decode(c["views"][0]["mask_rle"], (H, W)), _mask())
    assert c["would_merge_into"] is None
    assert _objects(db, LABEL) == [] and _versions(db) == v0
    with ImageStore.open(store_dir) as s:
        assert s.info()[0]["observations"] == obs0

    out = eng.commit(r["pending_id"], 0)
    assert out["created"] and not out["merged"] and not out["already_committed"]
    sym = out["symbol"]
    (row,) = _objects(db, LABEL)
    assert (
        row["s"] == sym
        and row["st"] == "admitted"
        and row["pv"] == qs.HUMAN_CONSOLE_POLICY
    )
    assert (out["state_version"], out["map_version"]) == _versions(db)
    assert out["map_version"] == v0[1] + 1
    if v0[0] is not None:
        assert out["state_version"] == v0[0] + 1
    n_obs, _, _ = db.execute(
        "MATCH (:Object {nodeSymbol: $s})-[:HAS_OBSERVATION]->(x) RETURN count(x) AS n",
        s=sym,
    )
    assert n_obs[0]["n"] == 3 and len(out["observation_ids"]) == 3
    # the masks reached the image store (RLE + kept PNG)
    with ImageStore.open(store_dir) as s:
        obs = s.observations(od._node_value(sym))
        assert len(obs) == 3 and all(
            o.kept and o.mask_path and o.mask_path.exists() for o in obs
        )
        assert np.array_equal(s.mask(obs[0]), _mask())
        assert {o.frame_id for o in obs} == {kf.frame_id for kf in eng.keyframes}
    # no allocator placeholder left behind
    assert (
        db.execute(
            "MATCH (o:Object) WHERE o._alloc_token IS NOT NULL RETURN count(o) AS n"
        )[0][0]["n"]
        == 0
    )

    again = eng.commit(r["pending_id"], 0)
    assert again["already_committed"] and again["symbol"] == sym
    # the map version always moves; the console's _State counter only when the
    # database has one (heracles master only bumps an existing _State)
    v1 = _versions(db)
    assert v1[1] == v0[1] + 1
    assert v1[0] == (None if v0[0] is None else v0[0] + 1) or (
        v0[0] is None and v1[0] == 1
    )

    # the same object found again: previewed and committed as a merge
    r2 = eng.query(LABEL, mode="topk", dry_run=True)
    assert r2["candidates"][0]["would_merge_into"]["symbol"] == sym
    out2 = eng.commit(r2["pending_id"], 0)
    assert out2["symbol"] == sym and out2["merged"] and not out2["created"]
    assert len(_objects(db, LABEL)) == 1

    assert eng.discard(r2["pending_id"])
    with pytest.raises(qs.PendingNotFound):
        eng.commit(r2["pending_id"], 0)
    with pytest.raises(IndexError):
        eng.commit(eng.query(LABEL, mode="topk", dry_run=True)["pending_id"], 5)


@needs_db
def test_plain_query_still_writes(db, tmp_path):
    eng, _ = _engine(tmp_path)
    r = eng.query(LABEL2)
    assert "dry_run" not in r and r["objects"] == 1
    (row,) = _objects(db, LABEL2)
    assert row["s"] == r["symbols"][0] and row["st"] == "candidate"


@needs_db
def test_claim_object_index_is_unique_under_concurrency(db):
    from heracles.query_interface import Neo4jWrapper

    db.execute("CREATE (o:Object {nodeSymbol: 'O90001', class: $c})", c=LABEL)
    tokens = [uuid.uuid4().hex for _ in range(8)]
    got, errors = [], []

    def worker(tok):
        try:
            with Neo4jWrapper(_NEO4J_URI, _auth(), atomic_queries=True) as w:
                got.append(od.claim_object_index(w, 0, tok))
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(t,)) for t in tokens]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    try:
        assert not errors and len(set(got)) == 8 and min(got) > 90001
    finally:
        db.execute(
            "MATCH (o:Object) WHERE o._alloc_token IN $t DETACH DELETE o", t=tokens
        )
