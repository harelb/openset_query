"""Same-object association of a query's per-view detections (3D clustering).

Every SAM3 detection of a query is reprojected into a world-frame point set + AABB
(``object_discovery``). Views of one object land on overlapping boxes; distinct
objects (two chairs 2 m apart, a mug on a table) do not. This module groups them:

1. **spatial hash** -- each detection's (slightly inflated) AABB is inserted into the
   cells of a uniform grid (cell ~ typical object size), so only detections sharing a
   cell are compared: O(n) on average instead of the greedy fusion's O(n^2);
2. **pairwise test** -- two detections join when their labels agree AND
   (3D IoU >= ``iou_min`` OR containment >= ``contain_min`` with a bounded size
   ratio OR centroid distance <= ``k_extent`` x object extent, clipped to
   [``min_dist_m``, ``max_dist_m``]), AND (when both carry an L2-normalised
   appearance embedding, e.g. the SigLIP2 crop embedding) cosine >= ``emb_min``;
3. **union-find** over the joined pairs; each component is one object.

Each cluster becomes one ``DiscoveredObject``: its geometry is the robust
(percentile-trimmed) reduction of the POOLED member points (a close, dense view
outweighs a far, grazing one), its score the best member score, its observations
sorted best first. Pure numpy; unit-tested with synthetic detections.
"""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from openset_query.reprojection import points_to_object


@dataclass
class AssociationParams:
    #: grid cell edge (m); ~ the size of the queried objects
    cell_m: float = 1.0
    #: inflate each AABB by this much (m) before hashing / overlap tests
    margin_m: float = 0.05
    iou_min: float = 0.2
    #: fraction of the SMALLER box inside the larger one
    contain_min: float = 0.6
    #: containment only counts up to this volume ratio (a view smeared over three
    #: chairs must not swallow each of them)
    max_volume_ratio: float = 8.0
    k_extent: float = 0.5
    min_dist_m: float = 0.25
    max_dist_m: float = 0.75
    #: appearance gate (only applied when both detections carry an embedding)
    emb_min: float = 0.75


def label_key(label: Optional[str]) -> str:
    return " ".join(str(label or "").lower().split())


class _UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, a: int) -> int:
        p = self.p
        while p[a] != a:
            p[a] = p[p[a]]
            a = p[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def _box_stats(lo: np.ndarray, hi: np.ndarray):
    size = np.maximum(hi - lo, 0.0)
    return float(np.prod(size)), size


def overlap_terms(lo_a, hi_a, lo_b, hi_b):
    """``(iou, containment of the smaller box, volume ratio big/small)``."""
    inter = np.maximum(np.minimum(hi_a, hi_b) - np.maximum(lo_a, lo_b), 0.0)
    vi = float(np.prod(inter))
    va, _ = _box_stats(lo_a, hi_a)
    vb, _ = _box_stats(lo_b, hi_b)
    if vi <= 0.0:
        return 0.0, 0.0, (max(va, vb) / max(min(va, vb), 1e-12))
    union = va + vb - vi
    small = max(min(va, vb), 1e-12)
    return vi / max(union, 1e-12), vi / small, max(va, vb) / small


def same_object(a: dict, b: dict, p: AssociationParams) -> bool:
    """The pairwise join test on two prepared detections (see :func:`_prepare`)."""
    if a["label"] != b["label"]:
        return False
    ea, eb = a.get("emb"), b.get("emb")
    if ea is not None and eb is not None and float(np.dot(ea, eb)) < p.emb_min:
        return False
    iou, contain, ratio = overlap_terms(a["lo"], a["hi"], b["lo"], b["hi"])
    if iou >= p.iou_min:
        return True
    if contain >= p.contain_min and ratio <= p.max_volume_ratio:
        return True
    extent = min(a["extent"], b["extent"])
    thr = float(np.clip(p.k_extent * extent, p.min_dist_m, p.max_dist_m))
    return float(np.linalg.norm(a["c"] - b["c"])) <= thr


def _prepare(d, emb=None, margin=0.0) -> dict:
    o = d.obj
    lo = np.asarray(o.bbox_min, dtype=np.float64) - margin
    hi = np.asarray(o.bbox_max, dtype=np.float64) + margin
    size = np.maximum(hi - lo, 0.0)
    return {
        "label": label_key(o.label),
        "lo": lo,
        "hi": hi,
        "c": np.asarray(o.centroid, dtype=np.float64),
        "extent": float(np.max(size)) if size.size else 0.0,
        "emb": None if emb is None else np.asarray(emb, dtype=np.float64),
    }


def cluster_indices(
    items: Sequence,
    params: Optional[AssociationParams] = None,
    embeddings: Optional[Sequence] = None,
) -> List[List[int]]:
    """Group per-view detections (``DiscoveredObject``-like: ``.obj`` with
    ``bbox_min``/``bbox_max``/``centroid``/``label``) into same-object clusters.
    Returns lists of indices into ``items``, each sorted by descending score;
    clusters ordered by their best score."""
    p = params or AssociationParams()
    n = len(items)
    if n == 0:
        return []
    embs = list(embeddings) if embeddings is not None else [None] * n
    prep = [_prepare(d, embs[i], p.margin_m) for i, d in enumerate(items)]
    cell = max(float(p.cell_m), 1e-3)
    # The centroid rule can join boxes up to max_dist_m apart: pad the hashed
    # extent so such pairs share a cell.
    pad = p.max_dist_m / 2.0
    grid: Dict[tuple, List[int]] = {}
    for i, d in enumerate(prep):
        lo = np.floor((np.minimum(d["lo"], d["c"] - pad)) / cell).astype(int)
        hi = np.floor((np.maximum(d["hi"], d["c"] + pad)) / cell).astype(int)
        # cap pathological (huge, smeared) boxes so one detection cannot touch
        # millions of cells: they still hash by their centroid neighbourhood
        if np.prod(hi - lo + 1) > 4096:
            cc = np.floor(d["c"] / cell).astype(int)
            lo, hi = cc - 1, cc + 1
        for key in itertools.product(*(range(a, b + 1) for a, b in zip(lo, hi))):
            grid.setdefault(key, []).append(i)
    uf = _UnionFind(n)
    seen = set()
    for members in grid.values():
        if len(members) < 2:
            continue
        for x in range(len(members)):
            i = members[x]
            for y in range(x + 1, len(members)):
                j = members[y]
                pair = (i, j) if i < j else (j, i)
                if pair in seen:
                    continue
                seen.add(pair)
                if uf.find(i) == uf.find(j):
                    continue
                if same_object(prep[i], prep[j], p):
                    uf.union(i, j)
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(uf.find(i), []).append(i)

    def score(i):
        return float(getattr(items[i].obj, "score", 0.0) or 0.0)

    out = [sorted(g, key=lambda i: -score(i)) for g in groups.values()]
    out.sort(key=lambda g: -score(g[0]))
    return out


def _obs_frame_key(o) -> str:
    kf = getattr(o, "keyframe", None)
    if kf is not None:
        fid, ns = getattr(kf, "frame_id", None), getattr(kf, "cache_ns", None)
        if fid is not None:
            return f"{ns}:f{int(fid)}" if ns else f"f{int(fid)}"
    return str(getattr(o, "timestamp_ns", id(o)))


def merge_cluster(
    members: Sequence, percentile_lo: float = 5.0, percentile_hi: float = 95.0
):
    """One ``DiscoveredObject`` from a cluster (members best first): pooled-point
    geometry, best score, observations sorted by score, ``num_frames``."""
    from openset_query.object_discovery import DiscoveredObject

    best = members[0]
    o = copy.copy(best.obj)
    pool = [
        m.obj.points
        for m in members
        if getattr(m.obj, "points", None) is not None and len(m.obj.points)
    ]
    if pool:
        reduced = points_to_object(
            np.vstack(pool), o.label, o.score, percentile_lo, percentile_hi
        )
        if reduced is not None:
            o.centroid, o.bbox_min, o.bbox_max = (
                reduced.centroid,
                reduced.bbox_min,
                reduced.bbox_max,
            )
            o.num_points = reduced.num_points
            o.points = reduced.points
    elif len(members) > 1:  # synthetic detections without points: score-weighted
        w = np.array([max(float(m.obj.score), 1e-6) for m in members])
        o.centroid = (np.stack([m.obj.centroid for m in members]) * w[:, None]).sum(
            0
        ) / w.sum()
        o.bbox_min = np.min(np.stack([m.obj.bbox_min for m in members]), axis=0)
        o.bbox_max = np.max(np.stack([m.obj.bbox_max for m in members]), axis=0)
        o.num_points = int(sum(m.obj.num_points for m in members))
    o.score = max(float(m.obj.score) for m in members)
    obs = [ob for m in members for ob in m.observations]
    obs.sort(key=lambda ob: -(ob.score if ob.score is not None else 0.0))
    disc = DiscoveredObject(obj=o, observations=obs)
    disc.num_frames = len({_obs_frame_key(ob) for ob in obs})
    return disc


def cluster_detections(
    items: Sequence,
    params: Optional[AssociationParams] = None,
    embeddings: Optional[Sequence] = None,
    percentile_lo: float = 5.0,
    percentile_hi: float = 95.0,
) -> list:
    """Per-view detections -> one ``DiscoveredObject`` per same-object cluster,
    best cluster first. Never mutates ``items``."""
    groups = cluster_indices(items, params, embeddings)
    return [
        merge_cluster([items[i] for i in g], percentile_lo, percentile_hi)
        for g in groups
    ]


def params_from_config(config) -> AssociationParams:
    """:class:`AssociationParams` from a ``Sam3FrontendConfig`` (``assoc_*`` fields;
    ``fuse_centroid_dist_m`` caps the centroid rule)."""
    p = AssociationParams()
    for f in p.__dataclass_fields__:
        v = getattr(config, f"assoc_{f}", None)
        if v is not None:
            setattr(p, f, v)
    cap = getattr(config, "fuse_centroid_dist_m", None)
    if cap is not None and getattr(config, "assoc_max_dist_m", None) is None:
        p.max_dist_m = float(cap)
        p.min_dist_m = min(p.min_dist_m, p.max_dist_m)
    return p
