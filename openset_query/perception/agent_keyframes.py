"""Agents' own keyframes in a Hydra image store (``hydra_python.image_store``).

Every keyframe archive is a ``KEYFRAME`` anchored to the agent node nearest in time.
An agent node's *own* keyframe is the one taken at the node's stamp
(``store.agent_frame(node, stamp)``). Without the stamp (store-only code paths) it is
the anchored keyframe whose ``anchor_T_body`` is the identity -- a frame taken at the
node's own stamp has exactly the node's odometric pose -- at most one per anchor. All
other keyframes are what ADT4 calls sub-keyframes (Neo4j ``SubKeyframe``).
"""

from __future__ import annotations

import datetime
from typing import Callable, Iterable, Optional

import numpy as np

#: max |anchor_T_body - I| of a frame taken at its anchor's own stamp
IDENTITY_TOL = 1e-3
#: an agent node's own keyframe is within this of the node's stamp (``agent_frame``)
MAX_DT_NS = 50_000_000

#: ``node_id -> stamp_ns`` (None when unknown)
StampFn = Callable[[int], Optional[int]]


def anchor_deviation(frame) -> float:
    """max |anchor_T_body - I| of ``frame`` (inf when it has no anchor transform)."""
    T = getattr(frame, "anchor_T_body", None)
    if T is None:
        return float("inf")
    return float(np.max(np.abs(np.asarray(T, dtype=np.float64).reshape(4, 4) - np.eye(4))))


def _to_ns(ts) -> Optional[int]:
    if ts is None:
        return None
    if isinstance(ts, datetime.timedelta):
        return (ts.days * 86_400 + ts.seconds) * 1_000_000_000 + ts.microseconds * 1_000
    try:
        return int(ts)
    except (TypeError, ValueError):
        return None


def stamps_from_graph(graph) -> Optional[StampFn]:
    """``node_id -> stamp_ns`` of the agent nodes of a spark_dsg graph or a
    ``hydra_python.image_store.JsonDsg`` (None for other pose sources, e.g. mappings)."""
    if graph is None:
        return None
    if hasattr(graph, "get_node") and hasattr(graph, "has_node"):  # spark_dsg

        def stamp(node_id: int) -> Optional[int]:
            try:
                if not graph.has_node(int(node_id)):
                    return None
                attrs = graph.get_node(int(node_id)).attributes
            except Exception:
                return None
            return _to_ns(getattr(attrs, "timestamp", None)) or None

        return stamp
    if hasattr(graph, "attributes") and hasattr(graph, "has_node"):  # JsonDsg

        def stamp(node_id: int) -> Optional[int]:
            attrs = graph.attributes(int(node_id)) or {}
            return _to_ns(attrs.get("timestamp")) or None

        return stamp
    return None


def own_keyframes(frames: Iterable, node_stamp: Optional[StampFn] = None) -> list:
    """The agents' own keyframes among ``frames`` (keyframes of any anchors), time-ordered.

    Per anchor: with a known stamp (``node_stamp(anchor)``), the keyframe nearest to it
    within :data:`MAX_DT_NS` (as ``ImageStore.agent_frame``); otherwise the one with an
    identity ``anchor_T_body``. Unanchored keyframes (a store indexed without agent
    nodes) are kept: they are no other node's frame.
    """
    by_anchor: dict[int, list] = {}
    out = []
    for f in frames:
        if f.anchor_node_id is None:
            out.append(f)
        else:
            by_anchor.setdefault(int(f.anchor_node_id), []).append(f)
    for anchor, cands in by_anchor.items():
        stamp = node_stamp(anchor) if node_stamp is not None else None
        if stamp is not None:
            best = min(cands, key=lambda f: (abs(f.timestamp_ns - stamp), f.frame_id))
            if abs(best.timestamp_ns - stamp) <= MAX_DT_NS:
                out.append(best)
            continue
        ident = [f for f in cands if anchor_deviation(f) < IDENTITY_TOL]
        if ident:
            out.append(min(ident, key=lambda f: (anchor_deviation(f), f.timestamp_ns,
                                                f.frame_id)))
    out.sort(key=lambda f: (f.timestamp_ns, f.frame_id))
    return out


def own_keyframe(store, node_id: int, timestamp_ns: Optional[int] = None):
    """The own keyframe of agent node ``node_id`` (``agent_frame`` when its stamp is
    known, else the identity-anchored fallback), or None."""
    if timestamp_ns is not None:
        return store.agent_frame(int(node_id), int(timestamp_ns), max_dt_ns=MAX_DT_NS)
    own = own_keyframes(store.frames_for_agent(int(node_id)))
    return own[0] if own else None


def keyframes(store, include_subkeyframes: bool, node_stamp: Optional[StampFn] = None):
    """Keyframes of ``store``, time-ordered: all of them with ``include_subkeyframes``,
    else only the agents' own (:func:`own_keyframes`)."""
    from hydra_python.image_store import FrameKind

    frames = store.frames(kind=FrameKind.KEYFRAME)
    return frames if include_subkeyframes else own_keyframes(frames, node_stamp)
