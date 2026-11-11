"""Observation provenance threading: detection score/detector/mechanism on
Observation dicts, and OBSERVED_IN/DEPICTS linking after upserts."""

import asyncio

import numpy as np
import pytest

from openset_query import object_discovery as od
from openset_query.config import Sam3FrontendConfig


def _discovered(centroid, label="chair", score=0.9, rgb_path="", ts=1):
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
            od.Observation(
                timestamp_ns=ts,
                mask_file="",
                bbox_2d=(0, 0, 10, 10),
                rgb_path=rgb_path,
                score=score,
            )
        ],
    )


def test_obs_dict_carries_provenance():
    o = od.Observation(
        timestamp_ns=5,
        mask_file="m.png",
        bbox_2d=(0, 0, 2, 2),
        rgb_path="/x/agent_5_rgb.jpg",
        score=0.83,
    )
    d = od._obs_dict("O7", o)
    assert d["nodeSymbol"] == "O7_5"
    assert d["score"] == 0.83
    assert d["detector"] == "sam3"
    assert d["mechanism"] == "sam3_query"


def test_depicts_pair_matches_frame_index_symbol():
    from openset_query.perception.frame_index import _frame_node_symbol

    o = od.Observation(
        timestamp_ns=5,
        mask_file="",
        bbox_2d=(0, 0, 1, 1),
        rgb_path="/x/agent_5_rgb.jpg",
    )
    pair = od._depicts_pair("O7", o)
    assert pair == (_frame_node_symbol(5, "/x/agent_5_rgb.jpg"), "O7")

    # No source path -> no DEPICTS edge candidate.
    o_nopath = od.Observation(timestamp_ns=5, mask_file="", bbox_2d=(0, 0, 1, 1))
    assert od._depicts_pair("O7", o_nopath) is None


class _FakeDB:
    def __init__(self):
        self.calls = []

    def execute(self, query, **params):
        self.calls.append((query, params))
        return ([], None, None)


class _FakeWriter:
    def __init__(self, db):
        self._db = db
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
        return f"O({self._n})"


@pytest.mark.adt4
def test_upsert_via_graph_writer_persists_scores_and_links(monkeypatch):
    import heracles.graph_interface as gi
    from openset_query.perception.frame_index import _frame_node_symbol

    obs_calls, frame_edge_calls, link_calls = [], [], []
    monkeypatch.setattr(
        gi, "insert_observations_to_db", lambda db, obs: obs_calls.append(obs)
    )
    monkeypatch.setattr(gi, "insert_edges", lambda db, *a, **k: None)
    monkeypatch.setattr(
        gi,
        "insert_frame_edges",
        lambda db, place_pairs, obj_pairs: frame_edge_calls.append(obj_pairs),
    )
    monkeypatch.setattr(
        gi, "link_observations_to_frames", lambda db: link_calls.append(db) or 0
    )

    db = _FakeDB()
    writer = _FakeWriter(db)
    disc = _discovered(
        [1.0, 2.0, 3.0], score=0.77, rgb_path="/data/agents/agent_9_rgb.jpg", ts=9
    )

    touched = asyncio.run(
        od.upsert_via_graph_writer([disc], writer, config=Sam3FrontendConfig())
    )
    assert touched == ["O(1)"]

    # Detection score + provenance markers reach the Observation dicts.
    assert len(obs_calls) == 1
    (obs,) = obs_calls[0]
    assert obs["score"] == 0.77
    assert obs["detector"] == "sam3"
    assert obs["mechanism"] == "sam3_query"

    # OBSERVED_IN linking ran, and DEPICTS pairs use the frame-index symbol.
    assert link_calls == [db]
    assert frame_edge_calls == [
        [(_frame_node_symbol(9, "/data/agents/agent_9_rgb.jpg"), "O(1)")]
    ]
