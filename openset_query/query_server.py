"""Persistent warm SAM3 query service (runs in the SAM3 env).

Loads SAM3 + the embedding model once and keeps them resident, preloads the
keyframes and their CLIP/SigLIP2 embeddings, and answers text queries over HTTP.
This removes the ~20 s per-query model-load cost that the subprocess CLI pays, so
queries run warm in ~1-2 s (warm models + a compiled image model + frustum
early-stop).

Clients (e.g. the ADT4 web console, which does not load SAM3) POST to this server.

Run:
    HF_HUB_OFFLINE=1 HERACLES_NEO4J_URI=bolt://localhost:7687 \
    HERACLES_NEO4J_USERNAME=neo4j HERACLES_NEO4J_PASSWORD=<password> \
    $ADT4_ENV/sam3/bin/python \
        -m openset_query.query_server \
        --port 8077 [--image-store <dir>...] [--mask-output-dir <dir>] \
        [--live-state ~/adt4_output/console_live.json]

API:
    GET  /health           -> {"status": "ready", "keyframes": N, "pending": N,
                               "sources": [{"id", "keyframes"}], "conf_threshold",
                               "busy", ...}
    POST /query            <- {"prompt": str, "frustum": bool, "top_k": int,
                               "max_frames": int, "patience": int, "min_ts": int,
                               "dry_run": bool, "source": str, "conf": float}
         source (optional): keyframe pool to search -- "all" (default), "prior"
                           (the Neo4j/prior-map store), "live" (every live mirror),
                           or one mirror id "live/<robot>/<epoch>" (see /health).
         conf (optional): SAM3 confidence threshold for this request only.
         render (optional, dry run): "all" (default) | "best" | "none" -- which
                           returned views carry a rendered image_uri ("none": fetch
                           them lazily with GET /frame?pending_id=&index=&view=).
         association (optional): "cluster" (default, 3D same-object clustering)
                           | "centroid" (legacy greedy fusion) for this request.
         The response carries "cache": {frames, hits, hit_rate, gpu, cpu, disk,
                           miss, feat_s} (SAM3 feature cache use) and dry runs
                           "detections" (per-view detections before clustering);
                           candidates carry num_frames + frame_keys.
                           -> {"prompt", "objects", "symbols", "elapsed_s",
                               "evidence": {sym: {"observations": [raw score
                               terms], "admission": decision|null}}}
         dry_run=true: detect only, write nothing; masks stay in memory under
                           -> {"prompt", "dry_run": true, "pending_id"|null,
                               "expires_in_s", "objects", "symbols": [],
                               "candidates": [see build_candidates], "elapsed_s",
                               "timings"}
    POST /commit           <- {"pending_id": str, "index": int}
                           -> {"pending_id", "index", "symbol", "created": bool,
                               "merged": bool, "label", "observation_ids",
                               "admission_status": "admitted", "policy_version",
                               "state_version", "map_version",
                               "already_committed": bool}
                              (404 unknown/expired pending_id, 400 bad index,
                               409 a live image whose robot is not relocalized)
    POST /discard          <- {"pending_id": str} -> {"discarded": bool}
    POST /reload           re-read the keyframe pool (Neo4j + the current run's live
                           mirrors; those are also re-read before each query)
    POST /rank             <- {"prompt": str, "top_k": int, "source": str,
                               "min_ts": int, "thumbs": bool, "thumb_dim": int}
                           -> {"prompt", "source", "pool", "frames": [{"rank",
                               "frame_key", "frame_id", "timestamp_ns", "source",
                               "score", "camera_position", "thumb_uri"}],
                               "elapsed_s", "timings": {"embed_rank", "render"}}
         SigLIP2 ranking only (cached image embeddings + one text encode): no
         SAM3, nothing cached or written.
    GET  /frame?key=<frame_key>&max_dim=N
                           -> {"frame_key", "uri" (raw JPEG data URI), "width",
                               "height", "timestamp_ns", "source", "annotated"}
                              (404 unknown key)
         &mask=<RLE JSON>&mask_shape=h,w[&bbox=x0,y0,x1,y1]: draw that mask/box;
         ?pending_id=&index=[&view=]: a dry-run candidate view (mask + box drawn).

SAM3 image-feature cache (feature_cache.py): the trunk output of every pool frame
is cached (GPU / RAM LRU + <image_store>/cache/sam3_feats_<version>/ on disk),
built in the background at startup and for new frames on /reload, so a warm query
runs only text + grounding per frame (~0.04 s vs ~0.15 s). /health reports
"feature_cache". SigLIP2 frame embeddings are read from / written to the Neo4j
keyframe nodes (frame_embeddings.py); precompute both offline with
``python -m openset_query.precompute``.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import logging
import os
import secrets
import threading
import time
import urllib.parse
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from openset_query.config import Sam3FrontendConfig
from openset_query.live_pool import DEFAULT_STATE_FILE, LiveNotAligned

logger = logging.getLogger("openset_query.query_server")

#: admission policy recorded when a human confirms a candidate in the console
HUMAN_CONSOLE_POLICY = "human_console_v1"
#: ``_State.dirty_layers`` entry of the objects layer (layer 2, partition 0)
OBJECTS_DIRTY_LAYER = "2:0"
PENDING_TTL_S = 600.0
PENDING_MAX = 32


class PendingNotFound(KeyError):
    pass


class UnknownSource(ValueError):
    pass


#: ``source`` of the prior-map (Neo4j) keyframes; live mirrors use their cache_ns
PRIOR_SOURCE = "prior"


def keyframe_source(kf) -> str:
    """Pool a keyframe belongs to: ``"prior"`` or ``"live/<robot>/<epoch>"``."""
    return getattr(kf, "cache_ns", None) or PRIOR_SOURCE


def frame_key(kf) -> str:
    """Stable id of a keyframe across stores (the embedding-cache key)."""
    from openset_query.embedding_filter import EmbeddingFilter

    return EmbeddingFilter.frame_key(kf)


def describe_sources(keyframes) -> list:
    """``[{"id", "keyframes"}]``: the prior pool first, then each live mirror."""
    counts = collections.Counter(keyframe_source(kf) for kf in keyframes)
    order = sorted(counts, key=lambda s: (s != PRIOR_SOURCE, s))
    return [{"id": s, "keyframes": counts[s]} for s in order]


def filter_pool(keyframes, source=None, min_ts=None) -> list:
    """Keyframes of ``source`` (None/"all": every pool; "live": every mirror; a
    mirror id or a prefix of one, e.g. "live/robot1") at/after ``min_ts``.
    Raises :class:`UnknownSource` for a source with no keyframes in the pool."""
    pool = list(keyframes)
    if source and source != "all":
        if source == "live":
            pool = [kf for kf in pool if keyframe_source(kf).startswith("live/")]
        else:
            pool = [
                kf
                for kf in pool
                if keyframe_source(kf) == source
                or keyframe_source(kf).startswith(source.rstrip("/") + "/")
            ]
        if not pool and source not in (PRIOR_SOURCE, "live"):
            raise UnknownSource(f"unknown source {source!r}")
    if min_ts is not None:
        pool = [kf for kf in pool if kf.timestamp_ns >= int(min_ts)]
    return pool


def _filter_posed(posed, live, source=None, min_ts=None) -> list:
    """:func:`filter_pool` over the posed pool; a live source whose robot was left
    out (not relocalized, not in the current run) is an empty pool, not an error."""
    try:
        return filter_pool(posed, source, min_ts)
    except UnknownSource:
        robots = (live or {}).get("robots") or {}
        parts = (source or "").split("/")
        if len(parts) >= 2 and parts[0] == "live" and parts[1] in robots:
            return []
        raise


class _ConfFiltered:
    """Segmenter view that drops detections below ``conf`` (the per-request SAM3
    threshold; also covers segmenters that cannot change their own threshold)."""

    def __init__(self, seg, conf):
        self._seg, self._conf = seg, float(conf)

    def segment(self, image, prompts):
        return [d for d in self._seg.segment(image, prompts) if d.score >= self._conf]

    def segment_batch(self, images, prompts, fixed_batch=None):
        return [
            [d for d in dets if d.score >= self._conf]
            for dets in self._seg.segment_batch(
                images, prompts, fixed_batch=fixed_batch
            )
        ]

    def segment_keyframe(self, kf, rgb, prompts):
        fn = getattr(self._seg, "segment_keyframe", None)
        dets = (
            fn(kf, rgb, prompts) if fn is not None else self._seg.segment(rgb, prompts)
        )
        return [d for d in dets if d.score >= self._conf]

    def __getattr__(self, name):
        return getattr(self._seg, name)


class PendingCache:
    """Dry-run results awaiting a human decision: TTL'd and bounded (oldest out)."""

    def __init__(
        self,
        ttl_s: float = PENDING_TTL_S,
        max_entries: int = PENDING_MAX,
        clock=time.monotonic,
    ):
        self.ttl_s, self.max_entries, self._clock = (
            float(ttl_s),
            int(max_entries),
            clock,
        )
        self._d: "collections.OrderedDict[str, dict]" = collections.OrderedDict()
        self._lock = threading.Lock()

    def _evict(self) -> None:
        now = self._clock()
        for pid in [
            p for p, e in self._d.items() if now - e["created_at"] > self.ttl_s
        ]:
            del self._d[pid]
        while len(self._d) > self.max_entries:
            self._d.popitem(last=False)

    def put(self, entry: dict) -> str:
        pid = secrets.token_hex(8)
        with self._lock:
            entry = dict(entry, created_at=self._clock(), committed={})
            self._d[pid] = entry
            self._evict()
        return pid

    def get(self, pid: str) -> dict:
        with self._lock:
            self._evict()
            if pid not in self._d:
                raise PendingNotFound(pid)
            return self._d[pid]

    def expires_in(self, pid: str) -> float:
        with self._lock:
            e = self._d.get(pid)
            return (
                0.0
                if e is None
                else max(0.0, self.ttl_s - (self._clock() - e["created_at"]))
            )

    def discard(self, pid: str) -> bool:
        with self._lock:
            return self._d.pop(pid, None) is not None

    def __len__(self) -> int:
        with self._lock:
            self._evict()
            return len(self._d)


def _dedup_match(disc, existing, dist_m: float):
    """The existing object ``upsert_objects`` would merge ``disc`` into (same rule)."""
    import numpy as np

    match, best = None, dist_m
    for sym, pos, name in existing:
        d = float(np.linalg.norm(pos - disc.obj.centroid))
        if d <= best and (name is None or name == disc.obj.label):
            best, match = d, {"symbol": sym, "distance_m": round(d, 3)}
    return match


#: ``render`` values of a dry-run query: annotated JPEG data URIs for every returned
#: view ("all"), only each candidate's best view ("best"), or none ("none": the client
#: fetches them lazily with ``GET /frame?pending_id=&index=&view=``)
RENDER_MODES = ("all", "best", "none")


def _view_frame_key(o):
    return None if o.keyframe is None else frame_key(o.keyframe)


def build_candidates(
    discovered,
    existing=(),
    dedup_dist_m: float = 0.75,
    max_views: int = 3,
    max_dim: int = 640,
    render: str = "all",
) -> list:
    """Per-candidate payload of a dry-run query (pure; no GPU, no DB).

    ``{"index", "label", "score", "centroid": [x,y,z], "bbox_min", "bbox_max",
    "num_points", "num_views", "num_frames", "frame_keys": [every member view's
    frame key, best first], "would_merge_into": {"symbol", "distance_m"}|null,
    "image_uri", "views": [{"view", "frame_key", "source", "frame_id", "timestamp_ns",
    "bbox_2d": [x0,y0,x1,y1], "score", "mask_rle" (hydra_python.image_store RLE JSON),
    "mask_shape": [h, w], "image_uri"}]}``, views best first (at most ``max_views``);
    ``image_uri`` is a JPEG data URI of the view with the mask tinted and the box
    drawn (``overlay.render_annotated_data_uri``) per ``render`` (see RENDER_MODES).
    """
    import json as _json

    from hydra_python.image_store import rle_encode

    from openset_query.overlay import render_annotated_data_uri

    out = []
    for i, disc in enumerate(discovered):
        obj = disc.obj
        ranked = sorted(disc.observations, key=lambda o: -(o.score or 0.0))
        views = []
        for j, o in enumerate(ranked[:max_views]):
            mask = o.mask_array()
            rle = o.mask_rle
            if rle is None and mask is not None:
                rle = _json.dumps(rle_encode(mask), separators=(",", ":"))
            uri = None
            if (
                (render == "all" or (render == "best" and j == 0))
                and o.rgb_path
                and os.path.exists(o.rgb_path)
            ):
                try:
                    uri = render_annotated_data_uri(
                        o.rgb_path, max_dim=max_dim, mask=mask, bbox=o.bbox_2d
                    )["uri"]
                except Exception:
                    logger.exception("could not render %s", o.rgb_path)
            kf = o.keyframe
            views.append(
                {
                    "view": j,
                    "frame_key": _view_frame_key(o),
                    "source": None if kf is None else keyframe_source(kf),
                    "frame_id": None
                    if kf is None or kf.frame_id is None
                    else int(kf.frame_id),
                    "timestamp_ns": int(o.timestamp_ns),
                    "bbox_2d": [int(v) for v in o.bbox_2d],
                    "score": None if o.score is None else float(o.score),
                    "mask_rle": rle,
                    "mask_shape": (
                        list(mask.shape[:2])
                        if mask is not None
                        else (list(o.mask_shape) if o.mask_shape else None)
                    ),
                    "image_uri": uri,
                }
            )
        keys = list(dict.fromkeys(k for k in (_view_frame_key(o) for o in ranked) if k))
        num_frames = getattr(disc, "num_frames", None)
        out.append(
            {
                "index": i,
                "label": obj.label,
                "score": float(obj.score),
                "centroid": [float(v) for v in obj.centroid],
                "bbox_min": [float(v) for v in obj.bbox_min],
                "bbox_max": [float(v) for v in obj.bbox_max],
                "num_points": int(obj.num_points),
                "num_views": len(disc.observations),
                "num_frames": int(num_frames)
                if num_frames is not None
                else (len(keys) or len(disc.observations)),
                "frame_keys": keys,
                "would_merge_into": _dedup_match(disc, existing, dedup_dist_m),
                "image_uri": views[0]["image_uri"] if views else None,
                "views": views,
                "sources": sorted(
                    {
                        keyframe_source(o.keyframe)
                        for o in disc.observations
                        if o.keyframe is not None
                    }
                ),
            }
        )
    return out


def find_live_stores(root) -> list:
    """``<root>/<robot>/<epoch>/image_store`` mirrors (the console's live store sink)."""
    if not root:
        return []
    pattern = os.path.join(
        os.path.expanduser(str(root)), "*", "*", "image_store", "store.sqlite"
    )
    return sorted(os.path.dirname(p) for p in glob.glob(pattern))


def load_live_keyframes(stores, include_subkeyframes: bool = False) -> list:
    """Keyframes of the live mirrors ``stores`` (``[(robot, store_id, store_dir)]``,
    the current run's, from the console's live state; a directory string is read as
    ``<dir>/<robot>/<store_id>/image_store`` for tools and tests), posed in the
    robot's own scene-graph frame (the mirror has no prior-map scene graph: see
    :func:`live_pool.posed_pool`). Tagged with their store so their ids are never
    mistaken for prior-map ids (frame reuse, embedding-cache keys)."""
    from hydra_python.image_store import ImageStore

    from openset_query import keyframe_store as ks

    if stores is None:
        return []
    if isinstance(stores, (str, os.PathLike)):
        stores = [(*d.split(os.sep)[-3:-1], d) for d in find_live_stores(stores)]
    out = []
    for robot, epoch, store_dir in stores:
        try:
            with ImageStore.open(store_dir) as store:
                kfs = ks.load_keyframes_from_store(
                    store, include_subkeyframes=include_subkeyframes
                )
        except Exception as e:
            logger.warning("live store %s unreadable: %s", store_dir, e)
            continue
        for kf in kfs:
            kf.store_dir = store_dir
            kf.cache_ns = f"live/{robot}/{epoch}"
            kf.robot_T_body = kf.world_T_body
        out.extend(kfs)
    return out


class QueryEngine:
    """Holds warm SAM3 + embedding models and the preloaded keyframes."""

    def __init__(
        self,
        uri,
        auth,
        config,
        image_store=None,
        mask_output_dir=None,
        embed_model="siglip2",
        batch_size=16,
        include_subkeyframes=False,
        admission_policy=None,
        live_store_root=None,
        live_state=None,
        segmenter=None,
        embedding_filter=None,
        keyframes=None,
        pending_ttl_s=PENDING_TTL_S,
        pending_max=PENDING_MAX,
        feature_cache=True,
        feature_cache_dir=None,
        gpu_cache_gb=None,
        cpu_cache_gb=4.0,
        max_disk_cache_gb=64.0,
        build_feature_cache=True,
        neo4j_embeddings=True,
        warm_batched=False,
    ):
        """``segmenter`` / ``embedding_filter`` / ``keyframes`` inject ready objects
        (tests, or a caller that already holds them) instead of loading SAM3, the
        embedding model and the Neo4j keyframe pool.

        ``feature_cache``: cache SAM3 image features per pool frame (True; or a ready
        :class:`~feature_cache.FeatureCache`), on disk under ``feature_cache_dir``
        (default ``<image_store[0]>/cache``), ``gpu_cache_gb`` on the GPU (None:
        min(4, free - 2) GB at startup), ``cpu_cache_gb`` in RAM;
        ``build_feature_cache`` fills it in the background.
        ``neo4j_embeddings``: read/write the SigLIP2 frame embeddings on the Neo4j
        keyframe nodes (see frame_embeddings.py)."""
        from openset_query import keyframe_store as ks
        from openset_query.feature_cache import GpuGate
        from heracles.query_interface import Neo4jWrapper

        self.uri, self.auth = uri, auth
        self.config = config
        self.image_store = image_store  # list of store dirs (None: read from Neo4j)
        self.mask_output_dir = mask_output_dir
        self.batch_size = (
            batch_size  # fixed batch for the batched mode (avoids recompiles)
        )
        # Serializes GPU work; queries go before the feature-cache builder.
        self._lock = GpuGate()
        self._ks = ks
        self._Neo4jWrapper = Neo4jWrapper
        self._include_subkeyframes = include_subkeyframes
        if live_store_root:
            logger.warning(
                "--live-store-root %s is ignored: live images come only from the "
                "console's current run (--live-state)",
                live_store_root,
            )
        #: the console's live state file (see live_pool); None = no live images
        self.live_state = live_state
        self._live_sig = None
        self.pending = PendingCache(pending_ttl_s, pending_max)
        self.neo4j_embeddings = neo4j_embeddings
        self.embedding_sources = {}
        self.cached = None  # CachedSegmenter when the feature cache is on
        self.builder = None
        # The admission policy that grades this server's candidate writes
        # (None = grading disabled; writes stay candidates).
        self.admission_policy = admission_policy
        if admission_policy is not None:
            logger.info("Admission policy active: %s", admission_policy.policy_version)
        else:
            logger.info("Admission policy OFF: detector writes stay candidates")

        if segmenter is not None:
            self.segmenter = segmenter
        else:
            from openset_query.sam3_segmenter import Sam3Segmenter

            logger.info(
                "Loading SAM3 (warm%s)...", ", compiled" if config.compile else ""
            )
            self.segmenter = Sam3Segmenter(config)
            self.segmenter._ensure_loaded()
        self.warmup_s = None
        if segmenter is None and config.compile:
            logger.info(
                "Warming up torch.compile (one-time, can take a few minutes)..."
            )
            t = time.time()
            self.segmenter.warm_up(n=3)
            if warm_batched:  # compile the batched path at its fixed size
                self.segmenter.warm_up_batched(batch_size)
            self.warmup_s = round(time.time() - t, 1)
            logger.info("torch.compile warm-up done in %.1fs", self.warmup_s)

        if keyframes is not None:
            self.keyframes = list(keyframes)
        else:
            logger.info("Connecting to Neo4j + preloading keyframes...")
            with Neo4jWrapper(uri, auth, atomic_queries=True) as db:
                if not self.image_store:
                    self.image_store = ks.image_store_paths_from_db(db) or None
                self.keyframes = self._load_pool(db)
            logger.info(
                "Loaded %d keyframes%s",
                len(self.keyframes),
                " (incl. subkeyframes)" if include_subkeyframes else "",
            )

        if feature_cache:
            self._init_feature_cache(
                feature_cache,
                feature_cache_dir,
                gpu_cache_gb,
                cpu_cache_gb,
                max_disk_cache_gb,
            )

        if embedding_filter is not None:
            self.ef = embedding_filter
        else:
            from openset_query.embedding_filter import EmbeddingFilter

            # Embedding cache lives next to the image store, keyed by frame_id.
            cache = (
                ks.embed_cache_path(self.image_store[0], embed_model)
                if self.image_store
                else None
            )
            self.ef = EmbeddingFilter(
                model=embed_model, device=config.device, cache_path=cache
            )
        if self.keyframes:
            logger.info(
                "Warming embedding cache over %d keyframes...", len(self.keyframes)
            )
            self._warm_embeddings()
        if self.builder is not None and build_feature_cache and self.keyframes:
            n = self.builder.enqueue(self._build_order(self.keyframes))
            logger.info(
                "SAM3 feature cache: %d of %d frames to build in the background",
                n,
                len(self.keyframes),
            )
        logger.info("Query engine ready.")

    def _init_feature_cache(self, feature_cache, cache_dir, gpu_gb, cpu_gb, disk_gb):
        from openset_query.feature_cache import (
            GB,
            CachedSegmenter,
            FeatureCache,
            FeatureCacheBuilder,
            TorchCodec,
        )

        seg = self.segmenter
        if not (hasattr(seg, "image_features") and hasattr(seg, "segment_features")):
            logger.info("SAM3 feature cache off (segmenter has no feature split)")
            return
        sup = getattr(seg, "supports_feature_cache", None)
        if sup is not None and not sup():
            return
        if isinstance(feature_cache, FeatureCache):
            cache = feature_cache
        else:
            version = seg.feature_version() if hasattr(seg, "feature_version") else "v0"
            if cache_dir is None and self.image_store:
                cache_dir = os.path.join(str(self.image_store[0]), "cache")
            sub = (
                os.path.join(cache_dir, f"sam3_feats_{version}") if cache_dir else None
            )
            if gpu_gb is None:
                gpu_gb = 4.0
                try:
                    import torch

                    free, _ = torch.cuda.mem_get_info()
                    gpu_gb = max(0.0, min(4.0, free / GB - 2.0))
                except Exception:
                    pass
            cache = FeatureCache(
                sub,
                TorchCodec(self.config.device),
                gpu_budget_bytes=int(gpu_gb * GB),
                cpu_budget_bytes=int(cpu_gb * GB),
                max_disk_bytes=int(disk_gb * GB),
            )
        self.cached = CachedSegmenter(seg, cache, frame_key)
        self.builder = FeatureCacheBuilder(self.cached, self._lock, frame_key)
        logger.info("SAM3 feature cache: %s", cache.stats())

    @staticmethod
    def _build_order(keyframes):
        """Build the prior map first, then the live mirrors (newest frames first:
        the ones a live query most likely wants)."""
        prior = [kf for kf in keyframes if keyframe_source(kf) == PRIOR_SOURCE]
        live = [kf for kf in keyframes if keyframe_source(kf) != PRIOR_SOURCE]
        live.sort(key=lambda kf: -int(kf.timestamp_ns))
        return prior + live

    def _warm_embeddings(self):
        """Fill the SigLIP2 cache: Neo4j node embeddings, the npz cache, then
        compute (and write back) the missing ones."""
        if not self.neo4j_embeddings or not hasattr(self.ef, "_cache"):
            self.ef.embed_images(
                [kf.rgb_path for kf in self.keyframes],
                keys=[self.ef.frame_key(kf) for kf in self.keyframes],
            )
            return
        from openset_query.frame_embeddings import sync_pool_embeddings

        try:
            with self._Neo4jWrapper(self.uri, self.auth, atomic_queries=True) as db:
                self.embedding_sources = sync_pool_embeddings(
                    db, self.ef, self.keyframes, self.ef.frame_key
                )
        except Exception as e:
            logger.warning("Neo4j embedding sync failed (%s); embedding locally", e)
            self.ef.embed_images(
                [kf.rgb_path for kf in self.keyframes],
                keys=[self.ef.frame_key(kf) for kf in self.keyframes],
            )

    def feature_cache_status(self):
        if self.cached is None:
            return None
        out = dict(self.cached.cache.stats())
        out.update(self.builder.status() if self.builder is not None else {})
        keys = [frame_key(kf) for kf in self.keyframes]
        out["pool_cached"] = sum(1 for k in keys if self.cached.cache.has(k))
        out["pool"] = len(keys)
        return out

    def _load_pool(self, db):
        """Keyframe pool: Neo4j Agent rows (prior map) + the current run's live
        mirrors (the console's live state)."""
        pool = self._ks.load_keyframes_from_db(
            db, include_subkeyframes=self._include_subkeyframes
        )
        self._live_sig = None
        return pool + self._load_live()

    def _live_stores(self):
        from openset_query import live_pool

        info = live_pool.read_state(self.live_state) if self.live_state else None
        stores = live_pool.current_stores(info["state"]) if info else []
        return info, stores

    @staticmethod
    def _stores_sig(stores):
        sig = []
        for robot, sid, d in stores:
            mt = []
            for f in ("store.sqlite", "store.sqlite-wal"):
                try:
                    mt.append(os.stat(os.path.join(d, f)).st_mtime_ns)
                except OSError:
                    mt.append(None)
            sig.append((robot, sid, d, *mt))
        return tuple(sig)

    def _load_live(self):
        """The current run's live keyframes (and remember what was read)."""
        _, stores = self._live_stores()
        live = load_live_keyframes(stores, self._include_subkeyframes)
        # after reading: opening a store can create its -wal file
        self._live_sig = self._stores_sig(stores)
        if stores or live:
            logger.info(
                "live mirrors of the current run (%s): %d keyframes",
                ", ".join(f"{r}/{s}" for r, s, _ in stores) or "none",
                len(live),
            )
        return live

    def _refresh_live(self):
        """Swap in the current run's live keyframes when the console's current
        stores changed or got new frames (caller holds the GPU lock). Old runs'
        mirrors never enter: only stores the console lists are read."""
        if not self.live_state:
            return
        _, stores = self._live_stores()
        if self._stores_sig(stores) == self._live_sig:
            return
        prior = [kf for kf in self.keyframes if keyframe_source(kf) == PRIOR_SOURCE]
        old = {frame_key(kf) for kf in self.keyframes}
        self.keyframes = prior + self._load_live()
        new = [kf for kf in self.keyframes if frame_key(kf) not in old]
        if new:
            try:
                self.ef.embed_images(
                    [kf.rgb_path for kf in new], keys=[frame_key(kf) for kf in new]
                )
            except Exception:
                logger.exception("embedding %d new live keyframes", len(new))
            if self.builder is not None:
                self.builder.enqueue(self._build_order(new))

    def _posed(self):
        """``(pool, live)``: the pool with live keyframes posed into the map (only
        relocalized robots of the current run), and ``live`` = per-robot status."""
        from openset_query import live_pool

        info = (
            live_pool.read_state(self.live_state)
            if self.live_state
            else {
                "ok": False,
                "reason": "live images off (no --live-state)",
                "state": None,
            }
        )
        pool, excluded = live_pool.posed_pool(self.keyframes, info)
        used = sorted({live_pool.robot_of(kf) for kf in pool if live_pool.is_live(kf)})
        live = {
            "state": "ok" if info.get("ok") else (info.get("reason") or "off"),
            "robots": {
                **{r: {"aligned": True, "reason": None} for r in used},
                **{r: {"aligned": False, "reason": why} for r, why in excluded.items()},
            },
        }
        return pool, live

    def reload_keyframes(self):
        """Re-read the keyframe pool from Neo4j (+ live mirrors) and warm the new embeddings.

        A LIVE mission ingests fresh Agent rows mid-run (acquire_view: the
        robot drove for new views); the pool is otherwise loaded once at
        startup. Under the GPU lock so a concurrent /query never sees a
        half-swapped pool. Returns (before, after) counts.
        """
        with self._lock:
            before = len(self.keyframes)
            with self._Neo4jWrapper(self.uri, self.auth, atomic_queries=True) as db:
                self.keyframes = self._load_pool(db)
            if self.keyframes:
                self._warm_embeddings()
            logger.info("reload_keyframes: %d -> %d", before, len(self.keyframes))
        # new frames (live mirrors, fresh Agent rows) join the feature cache in the
        # background, outside the GPU lock
        if self.builder is not None and self.keyframes:
            n = self.builder.enqueue(self._build_order(self.keyframes))
            if n:
                logger.info("SAM3 feature cache: %d new frames queued", n)
        return before, len(self.keyframes)

    # patience=1 stops after the FIRST frame that yields no new object, i.e. a
    # one-frame search over a pool of thousands -- it reports 0 for objects that
    # are plainly present. Measured on an indoor tour map (pool 4499, conf 0.25):
    # "a vending machine" processed 1 frame -> 0 objects at the old defaults, and
    # 45 frames -> 2 objects at these. Easy queries still stop early, because
    # patience only counts frames that add nothing.
    def sources(self) -> list:
        return describe_sources(self.keyframes)

    def busy(self) -> bool:
        return self._lock.locked()

    def _frame_index(self) -> dict:
        """``frame_key -> keyframe`` of the current pool (rebuilt after a reload)."""
        kfs = self.keyframes
        idx = getattr(self, "_fidx", None)
        if idx is None or idx[0] is not kfs:
            idx = (kfs, {frame_key(kf): kf for kf in kfs})
            self._fidx = idx
        return idx[1]

    @contextmanager
    def _segmenter_at(self, conf):
        """The segmenter with SAM3's confidence threshold at ``conf`` for this call
        (restored afterwards; caller holds the GPU lock). ``None``: as configured."""
        base = self.cached if self.cached is not None else self.segmenter
        if conf is None:
            yield base
            return
        setter = getattr(self.segmenter, "set_conf_threshold", None)
        old = self.config.conf_threshold
        if setter is not None:
            setter(conf)
        try:
            yield _ConfFiltered(base, conf)
        finally:
            if setter is not None:
                setter(old)
            self.config.conf_threshold = old

    def rank(
        self, prompt, top_k=24, source=None, min_ts=None, thumbs=True, thumb_dim=320
    ):
        """SigLIP2-only ranking of the pool against ``prompt`` (no SAM3): the top
        ``top_k`` keyframes with their cosine score and a JPEG thumbnail."""
        from openset_query.overlay import render_annotated_data_uri

        t0 = time.perf_counter()
        with self._lock:
            self._refresh_live()
        posed, live = self._posed()
        pool = _filter_posed(posed, live, source, min_ts)
        if not pool:
            return {
                "prompt": prompt,
                "source": source or "all",
                "pool": 0,
                "frames": [],
                "elapsed_s": 0.0,
                "timings": {},
                "note": "empty pool",
                "live": live,
            }
        with self._lock:  # the embedding model shares the GPU with SAM3
            t = time.perf_counter()
            ranked = self.ef.rank(pool, [prompt])
            t_rank = time.perf_counter() - t
        t = time.perf_counter()
        frames = []
        for i, (kf, score) in enumerate(ranked[: max(1, int(top_k))]):
            uri = None
            if thumbs and kf.rgb_path and os.path.exists(kf.rgb_path):
                try:
                    uri = render_annotated_data_uri(
                        kf.rgb_path, max_dim=int(thumb_dim)
                    )["uri"]
                except Exception:
                    logger.exception("could not render %s", kf.rgb_path)
            try:
                cam = [float(v) for v in kf.camera_position]
            except Exception:
                cam = None
            frames.append(
                {
                    "rank": i,
                    "frame_key": frame_key(kf),
                    "source": keyframe_source(kf),
                    "frame_id": None if kf.frame_id is None else int(kf.frame_id),
                    "timestamp_ns": int(kf.timestamp_ns),
                    "score": round(float(score), 5),
                    "camera_position": cam,
                    "thumb_uri": uri,
                }
            )
        t_render = time.perf_counter() - t
        total = time.perf_counter() - t0
        logger.info(
            "RANK '%s': %d/%d frames in %.2fs", prompt, len(frames), len(pool), total
        )
        return {
            "prompt": prompt,
            "source": source or "all",
            "pool": len(pool),
            "frames": frames,
            "elapsed_s": round(total, 3),
            "timings": {"embed_rank": round(t_rank, 3), "render": round(t_render, 3)},
        }

    def frame_image(
        self,
        key=None,
        max_dim=1280,
        pending_id=None,
        index=None,
        view=0,
        mask_rle=None,
        mask_shape=None,
        bbox=None,
    ):
        """Keyframe ``key`` as a JPEG data URI (KeyError: not in the pool / unknown).

        With ``pending_id`` + ``index`` (+ ``view``, default the best) the frame is
        that dry-run candidate view's, drawn with its mask and box (the lazy
        thumbnails of the console); ``mask_rle`` + ``mask_shape`` (+ ``bbox``) draw
        an explicit mask instead. Otherwise the raw frame."""
        from openset_query.overlay import render_annotated_data_uri

        mask = None
        if pending_id:
            entry = self.pending.get(pending_id)  # PendingNotFound
            disc = entry["discovered"][int(index)]  # IndexError
            ranked = sorted(disc.observations, key=lambda o: -(o.score or 0.0))
            o = ranked[int(view)]
            if o.keyframe is None:
                raise KeyError(f"{pending_id}[{index}] view {view} has no keyframe")
            kf = o.keyframe
            key = frame_key(kf)
            mask, bbox = o.mask_array(), o.bbox_2d
        else:
            kf = self._frame_index()[key]
            if mask_rle:
                from hydra_python.image_store import rle_decode

                if not mask_shape or len(mask_shape) < 2:
                    raise ValueError("mask needs mask_shape=h,w")
                mask = rle_decode(mask_rle, (int(mask_shape[0]), int(mask_shape[1])))
        if not kf.rgb_path or not os.path.exists(kf.rgb_path):
            raise KeyError(key)
        from PIL import Image

        with Image.open(kf.rgb_path) as im:
            w, h = im.size
        return {
            "frame_key": key,
            "uri": render_annotated_data_uri(
                kf.rgb_path,
                max_dim=int(max_dim),
                mask=mask,
                bbox=None if bbox is None else tuple(int(v) for v in bbox),
            )["uri"],
            "width": w,
            "height": h,
            "timestamp_ns": int(kf.timestamp_ns),
            "frame_id": None if kf.frame_id is None else int(kf.frame_id),
            "source": keyframe_source(kf),
            "annotated": mask is not None,
        }

    def query(
        self,
        prompt,
        mode="frustum",
        top_k=60,
        max_frames=60,
        patience=25,
        min_ts=None,
        dry_run=False,
        source=None,
        conf=None,
        max_views=3,
        render="all",
        association=None,
    ):
        """``source`` restricts the pool (see :func:`filter_pool`), ``conf``
        overrides SAM3's confidence threshold for this call, ``max_views`` caps the
        views per dry-run candidate and ``render`` which of them come with an image
        (RENDER_MODES); ``association`` overrides ``config.association``
        ("cluster" | "centroid") for this call."""
        if render not in RENDER_MODES:
            raise ValueError(f"render must be one of {RENDER_MODES}")
        if association not in (None, "cluster", "centroid"):
            raise ValueError("association must be 'cluster' or 'centroid'")
        with self._lock:
            old_assoc = getattr(self.config, "association", "cluster")
            if association:
                self.config.association = association
            if self.cached is not None:
                self.cached.reset_stats()
            self._refresh_live()
            try:
                with self._segmenter_at(conf) as seg:
                    out = self._query(
                        prompt,
                        mode,
                        top_k,
                        max_frames,
                        patience,
                        min_ts,
                        dry_run,
                        source,
                        seg,
                        max_views,
                        render,
                    )
            finally:
                self.config.association = old_assoc
            out.setdefault("source", source or "all")
            out["conf"] = self.config.conf_threshold if conf is None else float(conf)
            out["association"] = association or old_assoc
            if self.cached is not None:
                cs = self.cached.stats.to_json()
                out["cache"] = cs
                if cs["frames"]:
                    out.setdefault("timings", {})["cache_hit"] = cs["hit_rate"]
            return out

    def _query(
        self,
        prompt,
        mode,
        top_k,
        max_frames,
        patience,
        min_ts,
        dry_run,
        source,
        segmenter,
        max_views,
        render="all",
    ):
        from openset_query import object_discovery as od

        t0 = time.perf_counter()
        # Context-managed so the driver is closed after each query (no per-query
        # connection leak). __enter__ connect()s and returns the connected wrapper.
        with self._Neo4jWrapper(self.uri, self.auth, atomic_queries=True) as db:
            t_connect = time.perf_counter() - t0

            # min_ts restricts the pool to frames captured at/after the
            # given ns timestamp — the acquire/verify contract: "find NEW
            # evidence". Without it, a prior archive's perfect vantage
            # frames outrank fresh mediocre views forever and every
            # re-query re-reads the stale evidence (ranking starvation;
            # measured on a revisited simulated world).
            posed, live = self._posed()
            pool = _filter_posed(posed, live, source, min_ts)
            if not pool:
                out = {
                    "prompt": prompt,
                    "objects": 0,
                    "symbols": [],
                    "elapsed_s": 0.0,
                    "timings": {},
                    "note": "empty pool after source/min_ts filter",
                    "live": live,
                }
                if dry_run:
                    out.update(
                        dry_run=True, pending_id=None, candidates=[], expires_in_s=0.0
                    )
                return out

            t = time.perf_counter()
            ranked = self.ef.rank(pool, [prompt])
            t_rank = time.perf_counter() - t

            t = time.perf_counter()
            views = []
            if mode == "batched":
                sel = [kf for kf, _ in ranked[:top_k]]
                discovered = od.detect_batched(
                    sel,
                    segmenter,
                    [prompt],
                    self.config,
                    mask_output_dir=self.mask_output_dir,
                    fixed_batch=self.batch_size,
                )
            elif mode == "frustum":
                discovered = od.detect_frustum_aware(
                    ranked,
                    segmenter,
                    [prompt],
                    self.config,
                    max_frames=max_frames,
                    patience=patience,
                    mask_output_dir=self.mask_output_dir,
                    views_out=views,
                )
                if (
                    views
                    and getattr(self.config, "cluster_crop_embeddings", False)
                    and getattr(self.config, "association", "cluster") == "cluster"
                ):
                    discovered = od.associate(
                        views, self.config, od.crop_embeddings(views, self.ef)
                    )
            else:  # "topk": sequential over the top-K frames
                sel = [kf for kf, _ in ranked[:top_k]]
                discovered = od.detect_in_keyframes(
                    sel,
                    segmenter,
                    [prompt],
                    self.config,
                    mask_output_dir=self.mask_output_dir,
                )
            t_detect = time.perf_counter() - t

            # Score each detection crop against the target
            # prompt + the calibrated negative controls BEFORE the write,
            # so the raw terms persist onto the :Observation rows.
            t = time.perf_counter()
            if self.admission_policy is not None and discovered:
                od.attach_crop_evidence(
                    discovered,
                    self.ef,
                    prompt,
                    self.admission_policy.calibration.control_prompts,
                )
            t_evidence = time.perf_counter() - t

            if dry_run:
                out = self._dry_run_result(
                    db,
                    prompt,
                    discovered,
                    t0,
                    {
                        "connect": round(t_connect, 3),
                        "embed_rank": round(t_rank, 3),
                        "detect": round(t_detect, 3),
                        "evidence": round(t_evidence, 3),
                    },
                    max_views=max_views,
                    render=render,
                )
                out["pool"] = len(pool)
                out["live"] = live
                if views:
                    out["detections"] = len(views)
                return out

            t = time.perf_counter()
            touched = od.upsert_objects(
                discovered,
                G=None,
                db=db,
                config=self.config,
                image_store=self.image_store,
            )
            t_upsert = time.perf_counter() - t

            # Grade this query's candidates. Reads the object's
            # ACCUMULATED persisted evidence (same-prompt), so two one-view
            # queries can jointly reach min_views. Trusted priors /
            # already-decided objects are skipped inside grade_object.
            decisions = {}
            if self.admission_policy is not None:
                from openset_query.evidence.admission_policy import grade_object

                for sym in dict.fromkeys(touched):
                    decisions[sym] = grade_object(
                        db, sym, self.admission_policy, prompt=prompt
                    )
                    logger.info(
                        "admission %s: %s (%s)",
                        sym,
                        decisions[sym].status,
                        decisions[sym].reason,
                    )

        total = time.perf_counter() - t0
        timings = {
            "connect": round(t_connect, 3),
            "embed_rank": round(t_rank, 3),
            "detect": round(t_detect, 3),
            "evidence": round(t_evidence, 3),
            "upsert": round(t_upsert, 3),
        }
        logger.info("PROFILE '%s': total=%.2fs %s", prompt, total, timings)
        return {
            "prompt": prompt,
            "objects": len(touched),
            "symbols": touched,
            "elapsed_s": round(total, 2),
            "timings": timings,
            "evidence": build_evidence_payload(discovered, touched, decisions),
            "live": live,
        }

    def _dry_run_result(
        self, db, prompt, discovered, t0, timings, max_views=3, render="all"
    ):
        """Cache the detections (masks compacted to RLE) and describe them."""
        from openset_query import object_discovery as od

        t = time.perf_counter()
        candidates = build_candidates(
            discovered,
            od._existing_objects(db, None),
            self.config.dedup_centroid_dist_m,
            max_views=max(1, int(max_views)),
            render=render,
        )
        for disc in discovered:
            for o in disc.observations:
                o.compact_mask()
        pid = None
        if discovered:
            pid = self.pending.put({"prompt": prompt, "discovered": discovered})
        timings["render"] = round(time.perf_counter() - t, 3)
        total = time.perf_counter() - t0
        logger.info(
            "DRY RUN '%s': %d candidates (pending %s) in %.2fs",
            prompt,
            len(discovered),
            pid,
            total,
        )
        return {
            "prompt": prompt,
            "dry_run": True,
            "pending_id": pid,
            "expires_in_s": self.pending.ttl_s if pid else 0.0,
            "objects": len(discovered),
            "symbols": [],
            "candidates": candidates,
            "elapsed_s": round(total, 2),
            "timings": timings,
        }

    def commit(self, pending_id, index):
        """Write candidate ``index`` of a dry run: image-store observations + Neo4j
        object (dedup/merge as a normal query), admitted by the human, then both map
        version counters. Committing the same candidate again returns the first
        result (``already_committed``)."""
        from openset_query.heracles_compat import notify_map_change

        from openset_query.evidence.admission import ADMITTED, admit_object
        from openset_query import object_discovery as od

        entry = self.pending.get(pending_id)  # PendingNotFound -> 404
        index = int(index)
        if not 0 <= index < len(entry["discovered"]):
            raise IndexError(
                f"index {index} out of range (0..{len(entry['discovered']) - 1})"
            )
        with self._lock:
            if index in entry["committed"]:
                return dict(entry["committed"][index], already_committed=True)
            disc = entry["discovered"][index]
            # a detection from a live image lands in the prior map only while its
            # robot is still relocalized with about the alignment it was searched with
            from openset_query import live_pool

            kfs = [o.keyframe for o in disc.observations if o.keyframe is not None]
            if any(live_pool.is_live(kf) for kf in kfs):
                live_pool.check_commit(
                    kfs,
                    live_pool.read_state(self.live_state)
                    if self.live_state
                    else {"ok": False, "reason": "live images off", "state": None},
                )
            with self._Neo4jWrapper(self.uri, self.auth, atomic_queries=True) as db:
                result = {}
                touched = od.upsert_objects(
                    [disc],
                    G=None,
                    db=db,
                    config=self.config,
                    image_store=self.image_store,
                    result=result,
                )
                sym = touched[0]
                obs_ids = [f"{sym}_{o.timestamp_ns}" for o in disc.observations]
                admit_object(
                    db,
                    sym,
                    policy_version=HUMAN_CONSOLE_POLICY,
                    observation_ids=obs_ids,
                )
                state_version, map_version = notify_map_change(
                    db, [OBJECTS_DIRTY_LAYER], [sym]
                )
            created = sym in result.get("created", [])
            out = {
                "pending_id": pending_id,
                "index": index,
                "symbol": sym,
                "created": created,
                "merged": not created,
                "label": disc.obj.label,
                "observation_ids": obs_ids,
                "admission_status": ADMITTED,
                "policy_version": HUMAN_CONSOLE_POLICY,
                "state_version": state_version,
                "map_version": map_version,
                "already_committed": False,
            }
            entry["committed"][index] = out
            logger.info(
                "COMMIT %s[%d] -> %s (%s), map_version %d",
                pending_id,
                index,
                sym,
                "created" if created else "merged",
                map_version,
            )
            return out

    def discard(self, pending_id):
        return self.pending.discard(pending_id)


def build_evidence_payload(discovered, touched, decisions) -> dict:
    """Per-symbol evidence for the /query response: the raw
    per-observation score terms this query produced, plus the admission
    decision (or ``None`` when grading is off).

    ``touched[i]`` is the symbol ``discovered[i]`` landed on (one append per
    discovered object in ``_upsert_objects_write``); two discovered objects
    deduped onto one symbol merge their observation lists. Pure — unit-tested
    without a GPU or a database.
    """
    from openset_query.evidence.margins import ObservationEvidence

    by_sym: dict = {}
    for sym, disc in zip(touched, discovered):
        rows = by_sym.setdefault(sym, [])
        for o in disc.observations:
            controls = dict(o.control_scores or {})
            rows.append(
                ObservationEvidence(
                    obs_id=f"{sym}_{o.timestamp_ns}",
                    timestamp_ns=o.timestamp_ns,
                    sam3_score=o.score,
                    target_score=o.target_score,
                    control_scores=controls,
                    depth_valid_px=o.depth_valid_px,
                    view_xyz=o.view_xyz,
                    target_prompt=o.target_prompt,
                ).to_json()
            )
    return {
        sym: {
            "observations": rows,
            "admission": (
                decisions[sym].to_json() if decisions.get(sym) is not None else None
            ),
        }
        for sym, rows in by_sym.items()
    }


def make_handler(engine):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self):
            n = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self):
            url = urllib.parse.urlsplit(self.path)
            if url.path == "/health":
                pol = engine.admission_policy
                health = {
                    "status": "ready",
                    "keyframes": len(engine.keyframes),
                    "admission_policy": (pol.policy_version if pol else None),
                    "pending": len(engine.pending),
                    "live_state": getattr(engine, "live_state", None),
                }
                # optional extras (engines in tests may not have them)
                for key, fn in (("sources", "sources"), ("busy", "busy")):
                    if hasattr(engine, fn):
                        health[key] = getattr(engine, fn)()
                if hasattr(engine, "_posed"):
                    try:
                        health["live"] = engine._posed()[1]
                    except Exception as e:  # noqa: BLE001
                        health["live"] = {"state": f"error: {e}", "robots": {}}
                cfg = getattr(engine, "config", None)
                if cfg is not None:
                    health["conf_threshold"] = cfg.conf_threshold
                ef = getattr(engine, "ef", None)
                if ef is not None and getattr(ef, "model_id", None):
                    health["embed_model"] = ef.model_id
                fc = getattr(engine, "feature_cache_status", None)
                if fc is not None:
                    health["feature_cache"] = fc()
                if getattr(engine, "embedding_sources", None):
                    health["embedding_sources"] = engine.embedding_sources
                if getattr(engine, "warmup_s", None) is not None:
                    health["compile_warmup_s"] = engine.warmup_s
                self._send(200, health)
            elif url.path == "/frame":
                q = urllib.parse.parse_qs(url.query)

                def arg(name, default=None):
                    return (q.get(name) or [default])[0]

                key = arg("key", "")
                pid = arg("pending_id")
                if not key and not pid:
                    self._send(400, {"error": "missing key (or pending_id + index)"})
                    return
                kw = {"max_dim": int(arg("max_dim", 1280))}
                if pid:
                    if arg("index") is None:
                        self._send(400, {"error": "pending_id needs index"})
                        return
                    kw.update(
                        pending_id=pid,
                        index=int(arg("index")),
                        view=int(arg("view", 0)),
                    )
                elif arg("mask"):
                    shape = arg("mask_shape", "")
                    box = arg("bbox")
                    kw.update(
                        mask_rle=arg("mask"),
                        mask_shape=[int(v) for v in shape.split(",") if v],
                        bbox=[int(float(v)) for v in box.split(",")] if box else None,
                    )
                try:
                    self._send(200, engine.frame_image(key or None, **kw))
                except PendingNotFound:
                    self._send(404, {"error": "unknown or expired pending_id"})
                except (KeyError, IndexError):
                    self._send(404, {"error": f"unknown frame {key or pid!r}"})
                except ValueError as e:
                    self._send(400, {"error": str(e)})
                except Exception as e:
                    logger.exception("frame %s", key)
                    self._send(500, {"error": str(e)})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path == "/admission":
                # Runtime policy switch: swap the grading policy without
                # reloading the warm models. Body: {"policy": "margin"|
                # "geometric"|"off", "calibration": <path, optional if one is
                # already loaded>}.
                try:
                    n = int(self.headers.get("Content-Length", 0))
                    req = json.loads(self.rfile.read(n) or b"{}")
                    from openset_query.evidence.admission_policy import (
                        load_calibration,
                        make_admission_policy,
                    )

                    kind = req.get("policy", "off")
                    cal = None
                    if req.get("calibration"):
                        cal = load_calibration(req["calibration"])
                    elif engine.admission_policy is not None:
                        cal = engine.admission_policy.calibration
                    engine.admission_policy = make_admission_policy(kind, cal)
                    pol = engine.admission_policy
                    logger.info(
                        "admission policy -> %s", pol.policy_version if pol else "off"
                    )
                    self._send(
                        200, {"admission_policy": pol.policy_version if pol else None}
                    )
                except (ValueError, FileNotFoundError) as e:
                    self._send(400, {"error": str(e)})
                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    self._send(500, {"error": str(e)})
                return
            if self.path == "/reload":
                try:
                    before, after = engine.reload_keyframes()
                    self._send(
                        200,
                        {
                            "status": "reloaded",
                            "keyframes_before": before,
                            "keyframes": after,
                        },
                    )
                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    self._send(500, {"error": str(e)})
                return
            if self.path in ("/commit", "/discard"):
                try:
                    n = int(self.headers.get("Content-Length", 0))
                    req = json.loads(self.rfile.read(n) or b"{}")
                    pid = req.get("pending_id")
                    if not pid:
                        self._send(400, {"error": "missing pending_id"})
                        return
                    if self.path == "/discard":
                        self._send(200, {"discarded": engine.discard(pid)})
                        return
                    if req.get("index") is None:
                        self._send(400, {"error": "missing index"})
                        return
                    self._send(200, engine.commit(pid, int(req["index"])))
                except PendingNotFound:
                    self._send(404, {"error": "unknown or expired pending_id"})
                except LiveNotAligned as e:
                    self._send(409, {"error": str(e), "live_not_aligned": True})
                except (IndexError, ValueError) as e:
                    self._send(400, {"error": str(e)})
                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    self._send(500, {"error": str(e)})
                return
            if self.path == "/rank":
                try:
                    req = self._body()
                    prompt = (req.get("prompt") or "").strip()
                    if not prompt:
                        self._send(400, {"error": "empty prompt"})
                        return
                    min_ts = req.get("min_ts")
                    self._send(
                        200,
                        engine.rank(
                            prompt,
                            top_k=int(req.get("top_k", 24)),
                            source=req.get("source") or None,
                            min_ts=(int(min_ts) if min_ts is not None else None),
                            thumbs=bool(req.get("thumbs", True)),
                            thumb_dim=int(req.get("thumb_dim", 320)),
                        ),
                    )
                except (UnknownSource, ValueError) as e:
                    self._send(400, {"error": str(e)})
                except Exception as e:
                    logger.exception("rank failed")
                    self._send(500, {"error": str(e)})
                return
            if self.path != "/query":
                self._send(404, {"error": "not found"})
                return
            try:
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n) or b"{}")
                prompt = (req.get("prompt") or "").strip()
                if not prompt:
                    self._send(400, {"error": "empty prompt"})
                    return
                mode = req.get("mode") or (
                    "frustum" if req.get("frustum", True) else "topk"
                )
                if mode not in ("frustum", "batched", "topk"):
                    self._send(400, {"error": f"unknown mode {mode!r}"})
                    return
                min_ts = req.get("min_ts")
                # optional (newer) knobs: only passed when present, so an older
                # engine signature keeps working
                extra = {}
                if req.get("source"):
                    extra["source"] = str(req["source"])
                if req.get("conf") is not None:
                    conf = float(req["conf"])
                    if not 0.0 < conf < 1.0:
                        self._send(400, {"error": "conf must be in (0, 1)"})
                        return
                    extra["conf"] = conf
                if req.get("max_views") is not None:
                    extra["max_views"] = int(req["max_views"])
                if req.get("render") is not None:
                    if req["render"] not in RENDER_MODES:
                        self._send(
                            400, {"error": f"render must be one of {RENDER_MODES}"}
                        )
                        return
                    extra["render"] = str(req["render"])
                if req.get("association") is not None:
                    if req["association"] not in ("cluster", "centroid"):
                        self._send(
                            400, {"error": "association must be cluster|centroid"}
                        )
                        return
                    extra["association"] = str(req["association"])
                result = engine.query(
                    prompt,
                    mode=mode,
                    # keep in sync with QueryEngine.query's defaults -- this is the
                    # path the viewer/chat panel hits, and it does not pass these.
                    top_k=int(req.get("top_k", 60)),
                    max_frames=int(req.get("max_frames", 60)),
                    patience=int(req.get("patience", 25)),
                    min_ts=(int(min_ts) if min_ts is not None else None),
                    dry_run=bool(req.get("dry_run", False)),
                    **extra,
                )
                logger.info(
                    "query '%s' -> %d objects in %.2fs",
                    prompt,
                    result["objects"],
                    result["elapsed_s"],
                )
                self._send(200, result)
            except UnknownSource as e:
                self._send(400, {"error": str(e)})
            except Exception as e:
                import traceback

                traceback.print_exc()
                self._send(500, {"error": str(e)})

        def log_message(self, *a):
            pass  # quiet

    return Handler


def main(argv=None):
    p = argparse.ArgumentParser(description="Warm SAM3 query server")
    p.add_argument("--port", type=int, default=8077)
    p.add_argument("--device", default="cuda")
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="torch.compile the SAM3 image trunk + pixel decoder (warmed at "
        "startup; speeds up feature-cache building and uncached frames)",
    )
    p.add_argument(
        "--warm-batched",
        action="store_true",
        help="with --compile, also compile the batched mode at --batch-size",
    )
    p.add_argument(
        "--feature-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="cache SAM3 image features per keyframe (memory + disk) and "
        "build them in the background",
    )
    p.add_argument(
        "--feature-cache-dir",
        default=os.getenv("SAM3_FEATURE_CACHE_DIR"),
        help="disk cache root (default <first image store>/cache)",
    )
    p.add_argument(
        "--gpu-cache-gb",
        type=float,
        default=None,
        help="GPU budget of the feature cache (default min(4, free-2) GB)",
    )
    p.add_argument(
        "--cpu-cache-gb",
        type=float,
        default=4.0,
        help="RAM budget of the feature cache",
    )
    p.add_argument(
        "--max-disk-cache-gb",
        type=float,
        default=64.0,
        help="stop writing features to disk past this size",
    )
    p.add_argument(
        "--no-build-cache",
        action="store_true",
        help="do not precompute features in the background (cache on use)",
    )
    p.add_argument(
        "--neo4j-embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="read/write the SigLIP2 keyframe embeddings on the Neo4j nodes",
    )
    p.add_argument(
        "--association",
        choices=["cluster", "centroid"],
        default="cluster",
        help="same-object grouping of a query's detections",
    )
    p.add_argument(
        "--cluster-crop-embeddings",
        action="store_true",
        help="gate the clustering by the SigLIP2 cosine of detection crops",
    )
    p.add_argument("--embed-model", default="siglip2")
    p.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Fixed batch size for batched mode (padded; avoids recompiles)",
    )
    p.add_argument(
        "--image-store",
        nargs="+",
        default=None,
        help="Image store directory(ies); default $IMAGE_STORE, else the "
        "paths recorded in Neo4j. SAM3 observations are recorded in "
        "the first one.",
    )
    p.add_argument("--mask-output-dir", default=os.getenv("MASK_OUTPUT_DIR"))
    p.add_argument(
        "--include-subkeyframes",
        action="store_true",
        help="Also load the denser SubKeyframe nodes into the query pool "
        "(~2.9x more frames; longer one-time embedding cache build)",
    )
    p.add_argument(
        "--live-state",
        default=os.getenv("SAM3_LIVE_STATE", DEFAULT_STATE_FILE),
        help="The web console's live state file: the current run's live image-store "
        "mirrors and each robot's relocalization (map <- robot frame). Only those "
        "mirrors are searched, posed into the prior map, and only while the robot is "
        "relocalized. Empty string: no live images.",
    )
    p.add_argument(
        "--live-store-root",
        default=None,
        help="Deprecated and ignored: live mirrors come from --live-state only (a "
        "shared mirror folder holds earlier runs in other frames)",
    )
    p.add_argument(
        "--pending-ttl",
        type=float,
        default=PENDING_TTL_S,
        help="Seconds a dry-run result waits for /commit",
    )
    p.add_argument(
        "--admission-policy",
        default=os.getenv("SAM3_ADMISSION_POLICY", "off"),
        choices=["margin", "geometric", "off"],
        help="Grading of this server's candidate writes: 'margin' "
        "= negative-control relative-margin gate, 'geometric' = "
        "segmentation+geometric validation only, "
        "'off' = writes stay candidates",
    )
    p.add_argument(
        "--admission-calibration",
        default=os.getenv("SAM3_ADMISSION_CALIBRATION"),
        help="Path to the calibration artifact JSON (required unless "
        "--admission-policy off; see openset_query.eval.calibrate_admission)",
    )
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    from openset_query import heracles_compat

    uri, auth = heracles_compat.neo4j_settings()
    heracles_compat.log_optional_features()
    logger.info("map version bumps via %s", heracles_compat.map_change_backend())
    config = Sam3FrontendConfig(
        device=args.device,
        conf_threshold=args.conf,
        compile=args.compile,
        association=args.association,
        cluster_crop_embeddings=args.cluster_crop_embeddings,
    )

    from openset_query.evidence.admission_policy import (
        load_calibration,
        make_admission_policy,
    )

    calibration = (
        load_calibration(args.admission_calibration)
        if args.admission_calibration
        else None
    )
    policy = make_admission_policy(args.admission_policy, calibration)

    image_store = args.image_store or (
        os.environ["IMAGE_STORE"].split(os.pathsep)
        if os.environ.get("IMAGE_STORE")
        else None
    )
    engine = QueryEngine(
        uri,
        auth,
        config,
        image_store=image_store,
        mask_output_dir=args.mask_output_dir,
        embed_model=args.embed_model,
        batch_size=args.batch_size,
        include_subkeyframes=args.include_subkeyframes,
        admission_policy=policy,
        live_store_root=args.live_store_root,
        live_state=args.live_state or None,
        pending_ttl_s=args.pending_ttl,
        feature_cache=args.feature_cache,
        feature_cache_dir=args.feature_cache_dir,
        gpu_cache_gb=args.gpu_cache_gb,
        cpu_cache_gb=args.cpu_cache_gb,
        max_disk_cache_gb=args.max_disk_cache_gb,
        build_feature_cache=not args.no_build_cache,
        neo4j_embeddings=args.neo4j_embeddings,
        warm_batched=args.warm_batched,
    )
    server = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(engine))
    print(f"SAM3 query server ready on http://localhost:{args.port}  (POST /query)")
    server.serve_forever()


if __name__ == "__main__":
    main()
