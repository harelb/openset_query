"""Trajectory frame indexing for free-flight mission images.

Backed by Neo4j via heracles helpers; uses the TrajectoryFrame node label.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
from PIL import Image

from openset_query.perception.embedders import ImageEmbedder

logger = logging.getLogger(__name__)

ImageRef = Union[str, Path, "Image.Image"]

_EMBED_BATCH_SIZE = 16


# ---------------------------------------------------------------------------
# Stable node-symbol hash
# ---------------------------------------------------------------------------


def _frame_node_symbol(timestamp_ns: int, path: str) -> str:
    """Return a stable nodeSymbol ``F(<hash>)`` for a (timestamp_ns, path) pair.

    Uses the first 8 hex digits of SHA-256(``<timestamp_ns>:<path>``) interpreted
    as a uint32, modulo 2^31 (so the value is always non-negative).
    """
    raw = f"{timestamp_ns}:{path}".encode("utf-8")
    i = int(hashlib.sha256(raw).hexdigest()[:8], 16) % (2**31)
    return f"F({i})"


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class TrajectoryFrameIndex:
    """Discovers, embeds, stores, and queries free-flight trajectory images.

    Backed by Neo4j via heracles helpers; uses TrajectoryFrame node label.
    """

    def __init__(self, db, embedder: ImageEmbedder):
        """
        Parameters
        ----------
        db:
            heracles Neo4jWrapper (has ``.query`` / ``.execute`` methods).
        embedder:
            ImageEmbedder for image and text embedding.
        """
        self._db = db
        self._embedder = embedder

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(
        self,
        image_root: Union[str, Path],
        sample_stride: int = 5,
        place_attribution: bool = True,
    ) -> int:
        """Walk *image_root* for trajectory frames, embed them, and write
        TrajectoryFrame nodes to Neo4j.  Returns the number of frames written.

        Frame discovery
        ~~~~~~~~~~~~~~~
        - Looks for files matching ``**/*_rgb.jpg`` and ``**/*_rgb.png``.
        - Sub-samples to every ``sample_stride``-th file, sorted by mtime.
        - Attempts to read a sibling ``*_meta.json`` for pose/timestamp when
          present; otherwise falls back to:

          * ``timestamp_ns`` — ``int(os.path.getmtime(path) * 1e9)``
          * ``pos_x/y/z``     — ``0.0``
          * ``pose_qw``        — ``1.0``; ``pose_qx/y/z`` — ``0.0``

        Place attribution (when ``place_attribution=True``)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Queries Neo4j for all MeshPlace nodes, builds a KD-tree, and for each
        frame finds the nearest place.  Stores its ``nodeSymbol`` as
        ``place_id`` and writes an ``OBSERVED_AT`` edge.

        Parameters
        ----------
        image_root:
            Directory to walk for ``*_rgb.{jpg,png}`` files.
        sample_stride:
            Keep every *n*-th image (sorted by mtime ascending).  1 keeps all.
        place_attribution:
            Whether to attach frames to nearby MeshPlace nodes.

        Returns
        -------
        int
            Number of TrajectoryFrame nodes written to Neo4j.
        """
        image_root = Path(image_root)

        # --- Discover images ---
        candidates: list[Path] = []
        for pattern in ("**/*_rgb.jpg", "**/*_rgb.png"):
            candidates.extend(image_root.glob(pattern))

        # Sort by mtime, then subsample
        candidates.sort(key=lambda p: p.stat().st_mtime)
        candidates = candidates[::sample_stride]

        if not candidates:
            logger.warning("ingest: no *_rgb.{jpg,png} images found in %s", image_root)
            return 0

        # --- Read metadata and build raw frame dicts ---
        raw_frames: list[dict] = []
        for path in candidates:
            meta_path = path.with_name(path.stem.replace("_rgb", "") + "_meta.json")
            meta: dict = {}
            if meta_path.exists():
                try:
                    with open(meta_path) as fh:
                        meta = json.load(fh)
                except Exception as exc:
                    logger.warning("ingest: could not read %s: %s", meta_path, exc)

            ts_ns: int = int(meta.get("timestamp_ns", int(os.path.getmtime(path) * 1e9)))
            pos_x: float = float(meta.get("pos_x", 0.0))
            pos_y: float = float(meta.get("pos_y", 0.0))
            pos_z: float = float(meta.get("pos_z", 0.0))
            pose_qw: float = float(meta.get("pose_qw", 1.0))
            pose_qx: float = float(meta.get("pose_qx", 0.0))
            pose_qy: float = float(meta.get("pose_qy", 0.0))
            pose_qz: float = float(meta.get("pose_qz", 0.0))

            raw_frames.append(
                {
                    "path": str(path),
                    "timestamp_ns": ts_ns,
                    "pos_x": pos_x,
                    "pos_y": pos_y,
                    "pos_z": pos_z,
                    "pose_qw": pose_qw,
                    "pose_qx": pose_qx,
                    "pose_qy": pose_qy,
                    "pose_qz": pose_qz,
                }
            )

        return self._ingest_raw(raw_frames, place_attribution)

    def _ingest_raw(self, raw_frames: list[dict], place_attribution: bool) -> int:
        """Embed ``raw_frames`` (dicts with path/timestamp_ns/pose) and write them."""
        from heracles.graph_interface import (
            insert_frame_edges,
            insert_trajectory_frames_to_db,
        )

        # --- Assign nodeSymbols ---
        for rf in raw_frames:
            rf["nodeSymbol"] = _frame_node_symbol(rf["timestamp_ns"], rf["path"])

        # --- Place attribution (optional) ---
        place_lookup: dict[str, str] = {}  # nodeSymbol -> place_id
        if place_attribution:
            place_lookup = self._build_place_lookup(raw_frames)

        # --- Embed in batches of 16 ---
        paths_for_embed = [rf["path"] for rf in raw_frames]
        all_embeddings: list[list[float]] = []
        for batch_start in range(0, len(paths_for_embed), _EMBED_BATCH_SIZE):
            batch_paths = paths_for_embed[batch_start : batch_start + _EMBED_BATCH_SIZE]
            vecs = self._embedder.embed_images(batch_paths)  # (B, dim) float32
            for v in vecs:
                all_embeddings.append(v.tolist())

        # --- Build final frame dicts for DB ---
        frame_dicts: list[dict] = []
        for rf, emb in zip(raw_frames, all_embeddings):
            frame_dicts.append(
                {
                    "nodeSymbol": rf["nodeSymbol"],
                    "timestamp_ns": rf["timestamp_ns"],
                    "path": rf["path"],
                    "pos_x": rf["pos_x"],
                    "pos_y": rf["pos_y"],
                    "pos_z": rf["pos_z"],
                    "pose_qw": rf["pose_qw"],
                    "pose_qx": rf["pose_qx"],
                    "pose_qy": rf["pose_qy"],
                    "pose_qz": rf["pose_qz"],
                    "place_id": place_lookup.get(rf["nodeSymbol"]),
                    "embedding": emb,
                    "embedding_dim": len(emb),
                    "embedding_model": self._embedder.name,
                }
            )

        n_written = insert_trajectory_frames_to_db(self._db, frame_dicts)

        # --- Write OBSERVED_AT edges ---
        if place_attribution and place_lookup:
            frame_to_place_edges = [
                (ns, pid) for ns, pid in place_lookup.items() if pid is not None
            ]
            if frame_to_place_edges:
                insert_frame_edges(self._db, frame_to_place_edges, [])

        return n_written

    def ingest_store(
        self,
        store,
        pose_lookup=None,
        sample_stride: int = 1,
        place_attribution: bool = True,
        include_subkeyframes: bool = True,
    ) -> int:
        """Embed the keyframe images of a Hydra image store (all of them, or only the
        agent nodes' own with ``include_subkeyframes=False``) and write TrajectoryFrame
        nodes. Poses are composed from the anchor node via ``pose_lookup`` (a DSG,
        ``load_dsg(...)``, mapping or callable) with the odometry as fallback. Returns
        the number of frames written.
        """
        from openset_query.perception.agent_keyframes import keyframes, stamps_from_graph

        frames = [f for f in keyframes(store, include_subkeyframes, stamps_from_graph(pose_lookup))
                  if f.rgb_path is not None]
        frames = frames[::sample_stride]
        if not frames:
            logger.warning("ingest_store: no frames with images in %s", store.roots)
            return 0
        from openset_query.ingest_keyframes import _quat_from_R

        raw_frames: list[dict] = []
        for f in frames:
            T, _ = store.body_pose(f, pose_lookup)
            qw, qx, qy, qz = _quat_from_R(T[:3, :3])
            raw_frames.append({
                "path": str(f.rgb_path),
                "timestamp_ns": int(f.timestamp_ns),
                "pos_x": float(T[0, 3]), "pos_y": float(T[1, 3]), "pos_z": float(T[2, 3]),
                "pose_qw": qw, "pose_qx": qx, "pose_qy": qy, "pose_qz": qz,
            })
        return self._ingest_raw(raw_frames, place_attribution)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def query_by_text(
        self,
        text: str,
        k: int = 10,
        radius_m: Optional[float] = None,
        center_xyz: Optional[Sequence[float]] = None,
    ) -> list[dict]:
        """Hybrid spatial + semantic query using a text prompt.

        If *radius_m* and *center_xyz* are provided, pre-filters to frames
        within *radius_m* of *center_xyz*, then ranks by cosine similarity to
        the text embedding.  Otherwise ranks all frames by cosine similarity.

        Parameters
        ----------
        text:
            Natural-language query string.
        k:
            Maximum number of results to return.
        radius_m:
            Spatial pre-filter radius in metres.  Ignored unless *center_xyz*
            is also provided.
        center_xyz:
            3-D centre point ``[x, y, z]`` for spatial pre-filtering.

        Returns
        -------
        list[dict]
            Each dict has keys: ``nodeSymbol``, ``score``, ``path``,
            ``pos_x``, ``pos_y``, ``pos_z``, ``place_id``, ``timestamp_ns``.
        """
        query_vec = self._embedder.embed_text([text])[0]
        return self._semantic_query(query_vec, k=k, radius_m=radius_m, center_xyz=center_xyz)

    def query_by_image(
        self,
        image: ImageRef,
        k: int = 10,
        radius_m: Optional[float] = None,
        center_xyz: Optional[Sequence[float]] = None,
    ) -> list[dict]:
        """Hybrid spatial + semantic query using an image.

        Same return shape as :meth:`query_by_text` but the query vector comes
        from embedding the provided image.

        Parameters
        ----------
        image:
            Path to an image file or a ``PIL.Image.Image`` instance.
        k:
            Maximum number of results to return.
        radius_m:
            Spatial pre-filter radius in metres.  Ignored unless *center_xyz*
            is also provided.
        center_xyz:
            3-D centre point ``[x, y, z]`` for spatial pre-filtering.
        """
        query_vec = self._embedder.embed_images([image])[0]
        return self._semantic_query(query_vec, k=k, radius_m=radius_m, center_xyz=center_xyz)

    def lookup_near(
        self,
        center_xyz: Sequence[float],
        radius_m: float,
        k: int = 10,
    ) -> list[dict]:
        """Purely spatial query — frames within *radius_m* of *center_xyz*.

        Results are ordered by distance ascending.

        Parameters
        ----------
        center_xyz:
            3-D query centre ``[x, y, z]``.
        radius_m:
            Search radius in metres.
        k:
            Maximum number of results to return.

        Returns
        -------
        list[dict]
            Each dict has keys: ``nodeSymbol``, ``distance``, ``path``,
            ``pos_x``, ``pos_y``, ``pos_z``, ``place_id``, ``timestamp_ns``.
        """
        cx, cy, cz = float(center_xyz[0]), float(center_xyz[1]), float(center_xyz[2])
        cypher = (
            "WITH point({x: $cx, y: $cy, z: $cz}) AS c, $r AS r, $k AS k\n"
            "MATCH (f:TrajectoryFrame)\n"
            "WHERE point.distance(f.center, c) <= r\n"
            "WITH f, point.distance(f.center, c) AS dist\n"
            "RETURN f.nodeSymbol AS nodeSymbol, dist AS distance, f.path AS path,\n"
            "       f.pos_x AS pos_x, f.pos_y AS pos_y, f.pos_z AS pos_z,\n"
            "       f.place_id AS place_id, f.timestamp_ns AS timestamp_ns\n"
            "ORDER BY dist ASC LIMIT $k"
        )
        records, _, _ = self._db.execute(cypher, cx=cx, cy=cy, cz=cz, r=float(radius_m), k=k)
        return [dict(r) for r in records]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _semantic_query(
        self,
        query_vec: np.ndarray,
        k: int,
        radius_m: Optional[float],
        center_xyz: Optional[Sequence[float]],
    ) -> list[dict]:
        """Dispatch to vector-index or spatial+cosine depending on filters."""
        from heracles.graph_interface import query_similar_nodes

        use_spatial = radius_m is not None and center_xyz is not None

        if not use_spatial:
            # Fast path: use the pre-built vector index
            pairs = query_similar_nodes(self._db, "TrajectoryFrame", query_vec, k=k)
            if not pairs:
                return []
            symbols = [ns for ns, _ in pairs]
            scores_map = {ns: sc for ns, sc in pairs}

            # Fetch full properties for the returned symbols
            cypher = (
                "UNWIND $symbols AS sym\n"
                "MATCH (f:TrajectoryFrame {nodeSymbol: sym})\n"
                "RETURN f.nodeSymbol AS nodeSymbol, f.path AS path,\n"
                "       f.pos_x AS pos_x, f.pos_y AS pos_y, f.pos_z AS pos_z,\n"
                "       f.place_id AS place_id, f.timestamp_ns AS timestamp_ns"
            )
            records, _, _ = self._db.execute(cypher, symbols=symbols)
            results = []
            for r in records:
                d = dict(r)
                d["score"] = scores_map.get(d["nodeSymbol"], 0.0)
                results.append(d)
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:k]

        # Spatial + cosine path (slower; no index but works on filtered sets)
        cx = float(center_xyz[0])
        cy = float(center_xyz[1])
        cz = float(center_xyz[2])
        qv = list(map(float, query_vec))
        cypher = (
            "WITH point({x: $cx, y: $cy, z: $cz}) AS c, $qv AS qv, $r AS r, $k AS k\n"
            "MATCH (f:TrajectoryFrame)\n"
            "WHERE point.distance(f.center, c) <= r AND f.embedding IS NOT NULL\n"
            "WITH f, vector.similarity.cosine(qv, f.embedding) AS score\n"
            "RETURN f.nodeSymbol AS nodeSymbol, score, f.path AS path,\n"
            "       f.pos_x AS pos_x, f.pos_y AS pos_y, f.pos_z AS pos_z,\n"
            "       f.place_id AS place_id, f.timestamp_ns AS timestamp_ns\n"
            "ORDER BY score DESC LIMIT $k"
        )
        records, _, _ = self._db.execute(
            cypher, cx=cx, cy=cy, cz=cz, qv=qv, r=float(radius_m), k=k
        )
        return [dict(r) for r in records]

    def _build_place_lookup(self, raw_frames: list[dict]) -> dict[str, str]:
        """Return ``{nodeSymbol -> place_nodeSymbol}`` for nearest MeshPlace.

        Queries Neo4j for all MeshPlace positions, builds a cKDTree, and
        assigns each frame to its nearest place.  Returns an empty dict if
        there are no MeshPlace nodes in the database.
        """
        from scipy.spatial import cKDTree

        records, _, _ = self._db.execute(
            "MATCH (p:MeshPlace)\n"
            "RETURN p.nodeSymbol AS ns, p.pos_x AS x, p.pos_y AS y, p.pos_z AS z"
        )
        if not records:
            return {}

        place_symbols = [r["ns"] for r in records]
        place_coords = np.array(
            [[float(r["x"]), float(r["y"]), float(r["z"])] for r in records],
            dtype=np.float64,
        )
        tree = cKDTree(place_coords)

        frame_coords = np.array(
            [[rf["pos_x"], rf["pos_y"], rf["pos_z"]] for rf in raw_frames],
            dtype=np.float64,
        )
        _, idxs = tree.query(frame_coords, k=1)

        return {
            rf["nodeSymbol"]: place_symbols[idx]
            for rf, idx in zip(raw_frames, idxs)
        }
