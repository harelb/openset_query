"""SigLIP2 keyframe embeddings stored on the Neo4j keyframe nodes.

Each prior-map pool frame (``Agent`` / ``SubKeyframe`` node with a ``frame_id``) gets

* ``siglip2_emb``: its L2-normalised image embedding (float list),
* ``siglip2_model``: the embedding model id (a different model never matches),

plus one cosine vector index per label (``agent_siglip2_emb``,
``subkeyframe_siglip2_emb``) for ``db.index.vector.queryNodes``. The query server
loads them at startup (then the on-disk npz cache, then computes what is still
missing and writes it back), so a fresh server on a seeded map skips the SigLIP2
pass. Live-mirror frames are not in Neo4j and stay in the npz cache.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("openset_query.frame_embeddings")

EMB_PROP = "siglip2_emb"
MODEL_PROP = "siglip2_model"
FRAME_LABELS = ("Agent", "SubKeyframe")


def index_name(label: str, prop: str = EMB_PROP) -> str:
    return f"{label.lower()}_{prop}"


def ensure_vector_indexes(
    db, dim: int, labels: Sequence[str] = FRAME_LABELS, prop: str = EMB_PROP
) -> list:
    """Create the cosine vector index on ``prop`` for each label (idempotent)."""
    names = []
    for label in labels:
        name = index_name(label, prop)
        db.execute(
            f"CREATE VECTOR INDEX {name} IF NOT EXISTS FOR (n:{label}) ON (n.{prop}) "
            "OPTIONS {indexConfig: {`vector.dimensions`: $dim, "
            "`vector.similarity_function`: 'cosine'}}",
            dim=int(dim),
        )
        names.append(name)
    return names


def load_frame_embeddings(
    db, model_id: str, labels: Sequence[str] = FRAME_LABELS
) -> Dict[int, np.ndarray]:
    """``{frame_id: embedding (float32, L2-normalised)}`` of the keyframe nodes
    embedded with ``model_id``."""
    out: Dict[int, np.ndarray] = {}
    for label in labels:
        records, _, _ = db.execute(
            f"MATCH (a:{label}) WHERE a.frame_id IS NOT NULL AND a.{EMB_PROP} IS NOT NULL "
            f"AND a.{MODEL_PROP} = $m RETURN a.frame_id AS fid, a.{EMB_PROP} AS v",
            m=model_id,
        )
        for r in records:
            v = np.asarray(r["v"], dtype=np.float32)
            n = float(np.linalg.norm(v))
            if n > 0:
                out[int(r["fid"])] = v / n
    return out


def write_frame_embeddings(
    db,
    rows: Iterable[Tuple[int, np.ndarray]],
    model_id: str,
    labels: Sequence[str] = FRAME_LABELS,
    batch: int = 256,
) -> int:
    """Set ``siglip2_emb`` / ``siglip2_model`` on the keyframe nodes of ``rows``
    (``(frame_id, vector)``). Returns the number of nodes written."""
    rows = [
        {"fid": int(f), "v": [float(x) for x in np.asarray(v, dtype=np.float64)]}
        for f, v in rows
    ]
    n = 0
    for i in range(0, len(rows), batch):
        chunk = rows[i : i + batch]
        for label in labels:
            records, _, _ = db.execute(
                f"UNWIND $rows AS r MATCH (a:{label} {{frame_id: r.fid}}) "
                f"SET a.{EMB_PROP} = r.v, a.{MODEL_PROP} = $m RETURN count(a) AS n",
                rows=chunk,
                m=model_id,
            )
            n += int(records[0]["n"]) if records else 0
    return n


def ensure_frame_id_indexes(db, labels: Sequence[str] = FRAME_LABELS) -> None:
    """Range index on ``frame_id`` so the write-back MATCH is not a label scan."""
    for label in labels:
        db.execute(
            f"CREATE INDEX {label.lower()}_frame_id IF NOT EXISTS "
            f"FOR (n:{label}) ON (n.frame_id)"
        )


def query_vector_index(db, vec, k: int, labels: Sequence[str] = FRAME_LABELS) -> list:
    """Top-``k`` ``(frame_id, score)`` over the keyframe vector indexes (merged).
    Neo4j's cosine score is ``(1 + cos) / 2``; returned as the raw cosine."""
    v = [float(x) for x in np.asarray(vec, dtype=np.float64)]
    out = []
    for label in labels:
        records, _, _ = db.execute(
            f"CALL db.index.vector.queryNodes('{index_name(label)}', $k, $v) "
            "YIELD node, score RETURN node.frame_id AS fid, score",
            k=int(k),
            v=v,
        )
        out.extend((int(r["fid"]), 2.0 * float(r["score"]) - 1.0) for r in records)
    out.sort(key=lambda t: -t[1])
    return out[:k]


def sync_pool_embeddings(
    db,
    ef,
    keyframes,
    key_fn,
    write_back: bool = True,
    log: Optional[logging.Logger] = None,
) -> dict:
    """Fill ``ef``'s embedding cache for ``keyframes``: Neo4j first, then its npz
    cache (already loaded), then compute the rest; write the prior-map frames
    Neo4j lacks back onto their nodes and ensure the vector indexes.

    ``ef`` is an ``EmbeddingFilter`` (``model_id``, ``_cache``, ``embed_images``).
    Returns counts ``{"neo4j", "npz", "computed", "written"}``.
    """
    log = log or logger
    prior = [
        kf
        for kf in keyframes
        if getattr(kf, "cache_ns", None) is None
        and getattr(kf, "frame_id", None) is not None
    ]
    stats = {"neo4j": 0, "npz": 0, "computed": 0, "written": 0}
    db_emb: Dict[int, np.ndarray] = {}
    if db is not None and prior:
        try:
            db_emb = load_frame_embeddings(db, ef.model_id)
        except Exception as e:
            log.warning("could not read SigLIP2 embeddings from Neo4j: %s", e)
    for kf in prior:
        v = db_emb.get(int(kf.frame_id))
        if v is not None:
            key = key_fn(kf)
            if key not in ef._cache:
                ef._cache[key] = v
            stats["neo4j"] += 1
    keys = [key_fn(kf) for kf in keyframes]
    before = {k for k in keys if k in ef._cache}
    stats["npz"] = max(0, len(before) - stats["neo4j"])
    if keyframes:
        ef.embed_images([kf.rgb_path for kf in keyframes], keys=keys)
    stats["computed"] = sum(1 for k in keys if k in ef._cache and k not in before)
    if db is not None and write_back:
        missing = [
            (int(kf.frame_id), ef._cache[key_fn(kf)])
            for kf in prior
            if int(kf.frame_id) not in db_emb and key_fn(kf) in ef._cache
        ]
        try:
            if missing:
                ensure_frame_id_indexes(db)
                stats["written"] = write_frame_embeddings(db, missing, ef.model_id)
            any_vec = next(iter(ef._cache.values()), None)
            if any_vec is not None and prior:
                ensure_vector_indexes(db, int(np.asarray(any_vec).shape[-1]))
        except Exception as e:
            log.warning("could not write SigLIP2 embeddings to Neo4j: %s", e)
    log.info(
        "SigLIP2 embeddings: %d from Neo4j, %d from npz, %d computed, %d written back",
        stats["neo4j"],
        stats["npz"],
        stats["computed"],
        stats["written"],
    )
    return stats
