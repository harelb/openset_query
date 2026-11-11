"""Precompute the SAM3 query server's caches offline for a run / map database.

* SigLIP2 frame embeddings: onto the Neo4j keyframe nodes (``siglip2_emb`` +
  ``siglip2_model`` + cosine vector indexes, see frame_embeddings.py) and the npz
  cache next to the image store;
* SAM3 image features: ``<store>/cache/sam3_feats_<version>/`` (feature_cache.py).

The server then starts warm: it loads the embeddings from Neo4j and reads the
features from disk instead of recomputing them.

Run (SAM3 env):
    HF_HUB_OFFLINE=1 $ADT4_ENV/sam3/bin/python \\
        -m openset_query.precompute \\
        --image-store <run>/image_store --neo4j bolt://localhost:7687 \\
        [--include-subkeyframes] [--live-store-root ~/adt4_output/live] [--no-sam3] [--no-siglip]

Without ``--neo4j`` the pool is every keyframe (+ sub-keyframe) of the image store
and only the on-disk caches are written.
"""

from __future__ import annotations

import argparse
import logging
import os
import time

logger = logging.getLogger("openset_query.precompute")


def load_pool(args, db=None) -> list:
    from openset_query import keyframe_store as ks
    from openset_query.query_server import load_live_keyframes

    if db is not None:
        pool = ks.load_keyframes_from_db(db, include_subkeyframes=args.include_subkeyframes)
    else:
        from hydra_python.image_store import ImageStore

        with ImageStore.open(*args.image_store) as store:
            pool = ks.load_keyframes_from_store(
                store, include_subkeyframes=args.include_subkeyframes)
    if args.live_store_root:
        pool = pool + load_live_keyframes(args.live_store_root, args.include_subkeyframes)
    return pool


def precompute_siglip(args, pool, db) -> dict:
    from openset_query import keyframe_store as ks
    from openset_query.embedding_filter import EmbeddingFilter
    from openset_query.frame_embeddings import sync_pool_embeddings

    cache = ks.embed_cache_path(args.image_store[0], args.embed_model) if args.image_store else None
    ef = EmbeddingFilter(model=args.embed_model, device=args.device, cache_path=cache)
    t = time.time()
    stats = sync_pool_embeddings(db, ef, pool, ef.frame_key)
    stats["seconds"] = round(time.time() - t, 1)
    stats["npz"] = cache
    del ef
    return stats


def precompute_sam3(args, pool) -> dict:
    from openset_query import keyframe_store as ks
    from openset_query.config import Sam3FrontendConfig
    from openset_query.feature_cache import GB, FeatureCache, TorchCodec
    from openset_query.query_server import frame_key
    from openset_query.sam3_segmenter import Sam3Segmenter

    seg = Sam3Segmenter(Sam3FrontendConfig(device=args.device, compile=args.compile))
    seg._ensure_loaded()
    if not seg.supports_feature_cache():
        raise SystemExit("this SAM3 build has no feature split; nothing to precompute")
    root = args.feature_cache_dir or os.path.join(str(args.image_store[0]), "cache")
    cache = FeatureCache(os.path.join(root, f"sam3_feats_{seg.feature_version()}"),
                         TorchCodec(args.device), gpu_budget_bytes=0, cpu_budget_bytes=0,
                         max_disk_bytes=int(args.max_disk_cache_gb * GB))
    todo = [kf for kf in pool if not cache.has(frame_key(kf))]
    logger.info("SAM3 features: %d of %d frames to compute -> %s", len(todo), len(pool),
                cache.cache_dir)
    t0, gpu_s, failed = time.time(), 0.0, 0
    for i, kf in enumerate(todo):
        try:
            rgb = ks.read_rgb(kf)
        except Exception as e:
            logger.warning("cannot read %s: %s", kf.rgb_path, e)
            failed += 1
            continue
        t = time.perf_counter()
        feats, hw = seg.image_features(rgb)
        cache.put(frame_key(kf), feats, hw)
        gpu_s += time.perf_counter() - t
        if (i + 1) % 100 == 0:
            logger.info("  %d/%d (%.3f s/frame)", i + 1, len(todo), gpu_s / (i + 1))
    cache.close()
    st = cache.stats()
    return {"computed": len(todo) - failed, "failed": failed, "pool": len(pool),
            "seconds": round(time.time() - t0, 1),
            "per_frame_s": round(gpu_s / max(1, len(todo) - failed), 3),
            "disk_files": st["disk"], "disk_gb": st["disk_gb"], "dir": cache.cache_dir}


def main(argv=None):
    p = argparse.ArgumentParser(description="Precompute SAM3 query-server caches")
    p.add_argument("--image-store", nargs="+", default=None,
                   help="image store dir(s); default: the paths recorded in Neo4j")
    p.add_argument("--neo4j", default=None, help="bolt URI of the map database")
    p.add_argument("--neo4j-user", default=os.getenv("HERACLES_NEO4J_USERNAME", "neo4j"))
    p.add_argument("--neo4j-password", default=os.getenv("HERACLES_NEO4J_PASSWORD", "testpass"))
    p.add_argument("--include-subkeyframes", action="store_true")
    p.add_argument("--live-store-root", default=None)
    p.add_argument("--embed-model", default="siglip2")
    p.add_argument("--device", default="cuda")
    p.add_argument("--compile", action="store_true", help="torch.compile the SAM3 trunk")
    p.add_argument("--feature-cache-dir", default=None,
                   help="feature cache root (default <first image store>/cache)")
    p.add_argument("--max-disk-cache-gb", type=float, default=64.0)
    p.add_argument("--no-sam3", action="store_true", help="skip the SAM3 features")
    p.add_argument("--no-siglip", action="store_true", help="skip the SigLIP2 embeddings")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    db = None
    if args.neo4j:
        from heracles.query_interface import Neo4jWrapper

        db = Neo4jWrapper(args.neo4j, (args.neo4j_user, args.neo4j_password),
                          atomic_queries=True)
        db.connect()
        if not args.image_store:
            from openset_query import keyframe_store as ks

            args.image_store = ks.image_store_paths_from_db(db) or None
    if not args.image_store:
        p.error("--image-store is required (the database records none)")
    try:
        pool = load_pool(args, db)
        print(f"pool: {len(pool)} keyframes")
        if not args.no_siglip:
            print("SigLIP2:", precompute_siglip(args, pool, db))
        if not args.no_sam3:
            print("SAM3 features:", precompute_sam3(args, pool))
    finally:
        if db is not None:
            db.close()


if __name__ == "__main__":
    main()
