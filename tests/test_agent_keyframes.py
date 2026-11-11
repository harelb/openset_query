"""Agents' own keyframes vs the other keyframes (the old KEYFRAME / SUBKEYFRAME split)."""
import datetime

import numpy as np
import pytest

from openset_query.perception import agent_keyframes as ak
from openset_query import keyframe_store as ks
from openset_query import object_discovery as od
from store_helpers import build_store, node_id, off_anchor

image_store = pytest.importorskip("hydra_python.image_store")
ImageStore = image_store.ImageStore


def _two_agents(tmp_path):
    """a0: own frame at 10 + an other keyframe at 15; a1: own at 20 + other at 22."""
    sb = build_store(tmp_path / "store")
    a0, a1 = node_id("a0"), node_id("a1")
    ids = {
        10: sb.frame(10, anchor=a0),
        15: sb.frame(15, anchor=a0, anchor_T_body=off_anchor()),
        20: sb.frame(20, anchor=a1),
        22: sb.frame(22, anchor=a1, anchor_T_body=off_anchor(0.1)),
    }
    return sb.close(), ids


def test_own_keyframes_without_stamps_are_the_identity_anchored_ones(tmp_path):
    store_dir, _ = _two_agents(tmp_path)
    with ImageStore.open(store_dir) as s:
        assert [f.timestamp_ns for f in ak.keyframes(s, False)] == [10, 20]
        assert [f.timestamp_ns for f in ak.keyframes(s, True)] == [10, 15, 20, 22]
        assert ak.own_keyframe(s, node_id("a0")).timestamp_ns == 10


def test_own_keyframe_with_the_node_stamp_is_agent_frame(tmp_path):
    store_dir, _ = _two_agents(tmp_path)
    with ImageStore.open(store_dir) as s:
        # the stamp wins over the identity fallback
        assert ak.own_keyframe(s, node_id("a0"), 15).timestamp_ns == 15
        assert ak.own_keyframe(s, node_id("a0"), 10 + 10**9) is None   # nothing within 50 ms
        stamps = {node_id("a0"): 15, node_id("a1"): None}.get
        assert [f.timestamp_ns for f in ak.keyframes(s, False, stamps)] == [15, 20]


def test_stamps_from_a_spark_dsg_graph():
    spark_dsg = pytest.importorskip("spark_dsg")
    G = spark_dsg.DynamicSceneGraph()
    attrs = spark_dsg.AgentNodeAttributes()
    attrs.timestamp = datetime.timedelta(microseconds=15)
    G.add_node(spark_dsg.LayerKey(2, ord("a")), node_id("a0"), attrs)
    stamp = ak.stamps_from_graph(G)
    assert stamp(node_id("a0")) == 15_000 and stamp(node_id("a9")) is None
    assert ak.stamps_from_graph({1: np.eye(4)}) is None


def test_load_keyframes_from_store_subkeyframe_flag(tmp_path):
    store_dir, _ = _two_agents(tmp_path)
    with ImageStore.open(store_dir) as s:
        assert [k.timestamp_ns for k in ks.load_keyframes_from_store(s)] == [10, 20]
        assert len(ks.load_keyframes_from_store(s, include_subkeyframes=True)) == 4


def test_sam3_write_back_survives_reindexing(tmp_path):
    """SAM3 observations recorded into an indexed run's store keep their external
    tracks when the indexer refreshes the store."""
    run = tmp_path / "run"
    (run / "tracks").mkdir(parents=True)              # a run the indexer knows
    sb = build_store(run / "image_store", size=(20, 20))
    sb.frame(42, anchor=node_id("a0"))
    sb.close()
    with ImageStore.open(run, index=False) as s:
        (kf,) = ks.load_keyframes_from_store(s)

    w = od.open_sam3_writer(str(run))
    od.record_observations(w, "O1", [od.Observation(42, "", (1, 2, 4, 6), kf.rgb_path,
                                                    score=0.7, keyframe=kf)])
    w.close()
    image_store.index(run)
    with ImageStore.open(run, index=False) as s:
        (o,) = s.observations(node_id("O1"))
        from openset_query.store_writeback import sam3_track_id

        assert o.track_id == sam3_track_id(node_id("O1"))
        assert o.frame_id == kf.frame_id and o.kept and tuple(o.bbox2d) == (1, 2, 4, 6)
