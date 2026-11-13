"""Live images: only the console's current run, posed into the prior map, and only
while the robot is relocalized (the Oct 19 phone booth: an old run's mirror, in its
own odometry frame, searched as if it were the map)."""

import json
import time

import numpy as np
import pytest

from openset_query import live_pool
from openset_query import query_server as qs
from openset_query.config import Sam3FrontendConfig
from store_helpers import build_store

image_store = pytest.importorskip("hydra_python.image_store")

from test_sam3_query_tool import (  # noqa: E402
    RankEF,
    ScoredSegmenter,
    StubDB,
    write_live_state,
)


def _mirror(root, robot, sid, ts, x=0.0):
    b = build_store(root / robot / sid / "image_store")
    T = np.eye(4)
    T[0, 3] = x
    b.frame(ts, color=(10, 10, 200), anchor=image_store.make_node_id("a", 9), pose=T)
    return b.close()


def _engine(tmp_path, state_path, prior_kfs=()):
    cfg = Sam3FrontendConfig(device="cpu", conf_threshold=0.3)
    eng = qs.QueryEngine(
        "bolt://unused",
        ("u", "p"),
        cfg,
        image_store=[str(tmp_path / "prior")],
        segmenter=ScoredSegmenter(cfg),
        embedding_filter=RankEF(),
        keyframes=list(prior_kfs),
        live_state=state_path,
    )
    eng._Neo4jWrapper = StubDB
    return eng


def test_read_state_missing_stale_and_ok(tmp_path):
    p = tmp_path / "s.json"
    assert not live_pool.read_state(p)["ok"]
    assert "not mirroring" in live_pool.read_state(p)["reason"]
    p.write_text(json.dumps({"written_ns": time.time_ns() - int(60e9), "robots": {}}))
    info = live_pool.read_state(p)
    assert not info["ok"] and "old" in info["reason"]
    p.write_text(json.dumps({"written_ns": time.time_ns(), "robots": {}}))
    assert live_pool.read_state(p)["ok"]
    assert not live_pool.read_state(None)["ok"]


def test_only_the_current_runs_store_is_loaded(tmp_path):
    root = tmp_path / "run_today" / "live"
    _mirror(root, "robot1", "1760000000", 5000)
    # an earlier run's mirror of the same robot, and one in another (shared) folder
    _mirror(root, "robot1", "old0000000", 4000)
    _mirror(tmp_path / "shared_live", "robot1", "oct19", 3000)
    state = write_live_state(tmp_path / "console_live.json", root)
    eng = _engine(tmp_path, state)
    with eng._lock:
        eng._refresh_live()
    ids = {qs.keyframe_source(kf) for kf in eng.keyframes}
    assert ids == {"live/robot1/1760000000"}
    # refresh is a no-op while nothing changed
    before = eng.keyframes
    with eng._lock:
        eng._refresh_live()
    assert eng.keyframes is before


def test_live_keyframes_are_posed_into_the_map(tmp_path):
    root = tmp_path / "live"
    _mirror(root, "robot1", "1760000000", 5000, x=1.0)
    # map <- robot frame: +10 m in x, 90 deg about z
    qz = np.sin(np.pi / 4)
    state = write_live_state(
        tmp_path / "s.json", root, T=(10, 0, 0, np.cos(np.pi / 4), 0, 0, qz)
    )
    eng = _engine(tmp_path, state)
    with eng._lock:
        eng._refresh_live()
    pool, live = eng._posed()
    (kf,) = pool
    assert np.allclose(kf.world_T_body[:3, 3], [10.0, 1.0, 0.0], atol=1e-6)
    assert np.allclose(kf.robot_T_body[:3, 3], [1.0, 0.0, 0.0])
    # the loaded keyframe itself keeps its robot-frame pose
    assert np.allclose(eng.keyframes[0].world_T_body[:3, 3], [1.0, 0.0, 0.0])
    assert live["robots"] == {"robot1": {"aligned": True, "reason": None}}


def test_not_relocalized_robot_is_left_out(tmp_path):
    root = tmp_path / "live"
    _mirror(root, "robot1", "1760000000", 5000)
    state = write_live_state(tmp_path / "s.json", root, relocalized=False)
    eng = _engine(tmp_path, state)
    r = eng.rank("mug", source="live/robot1")
    assert r["pool"] == 0
    assert r["live"]["robots"]["robot1"]["aligned"] is False
    assert "not relocalized" in r["live"]["robots"]["robot1"]["reason"]
    q = eng.query("mug", mode="topk", dry_run=True)
    assert q["candidates"] == []


def test_commit_refused_when_relocalization_is_lost_or_moved(tmp_path):
    root = tmp_path / "live"
    _mirror(root, "robot1", "1760000000", 5000)
    path = tmp_path / "s.json"
    write_live_state(path, root)
    eng = _engine(tmp_path, str(path))
    r = eng.query("mug", mode="topk", dry_run=True, source="live/robot1")
    (c,) = r["candidates"]
    assert c["sources"] == ["live/robot1/1760000000"]
    # relocalization lost before the click
    write_live_state(path, root, relocalized=False)
    with pytest.raises(live_pool.LiveNotAligned, match="not relocalized"):
        eng.commit(r["pending_id"], 0)
    # relocalized again, but the alignment moved 2 m since the search
    write_live_state(path, root, T=(2, 0, 0, 1, 0, 0, 0))
    with pytest.raises(live_pool.LiveNotAligned, match="moved"):
        eng.commit(r["pending_id"], 0)
    # same alignment as the search: the check passes (the write itself needs Neo4j)
    write_live_state(path, root)
    kfs = [
        o.keyframe
        for o in eng.pending.get(r["pending_id"])["discovered"][0].observations
    ]
    live_pool.check_commit(kfs, live_pool.read_state(path))


def test_no_state_file_means_no_live_images(tmp_path):
    root = tmp_path / "live"
    _mirror(root, "robot1", "1760000000", 5000)
    eng = _engine(tmp_path, str(tmp_path / "missing.json"))
    with eng._lock:
        eng._refresh_live()
    assert eng.keyframes == []


def test_store_change_invalidates_commit(tmp_path):
    root = tmp_path / "live"
    _mirror(root, "robot1", "1760000000", 5000)
    path = tmp_path / "s.json"
    write_live_state(path, root)
    eng = _engine(tmp_path, str(path))
    r = eng.query("mug", mode="topk", dry_run=True, source="live/robot1")
    st = json.loads(path.read_text())
    st["robots"]["robot1"]["store_id"] = "newrun0000"
    st["written_ns"] = time.time_ns()
    path.write_text(json.dumps(st))
    with pytest.raises(live_pool.LiveNotAligned, match="store changed"):
        eng.commit(r["pending_id"], 0)
