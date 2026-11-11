"""Ingest hydra keyframes into openset's TrajectoryFrameIndex (step 5).

Reads the keyframe / sub-keyframe frames of the Hydra image store, composes each pose
from its anchor node (``--dsg``; odometry when absent) and runs
``TrajectoryFrameIndex.ingest_store`` so the frame embeddings + poses land in Neo4j
for `query_by_text`.

Run (SAM3 env):
    HF_HUB_OFFLINE=1 HERACLES_NEO4J_URI=bolt://localhost:7690 \
    HERACLES_NEO4J_USERNAME=neo4j HERACLES_NEO4J_PASSWORD=<password> \
    $ADT4_ENV/sam3/bin/python \
        -m openset_query.ingest_keyframes --image-store <run>/image_store [--dsg <run>/backend/dsg.json]
"""

from __future__ import annotations

import argparse
import os

import numpy as np


def _quat_from_R(R: np.ndarray) -> tuple:
    """Rotation matrix -> (qw, qx, qy, qz)."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        if i == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / s; qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s; qz = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / s; qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s; qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw = (R[1, 0] - R[0, 1]) / s; qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s; qz = 0.25 * s
    return float(qw), float(qx), float(qy), float(qz)


def main(argv=None):
    p = argparse.ArgumentParser(description="Ingest keyframes into TrajectoryFrameIndex")
    p.add_argument("--image-store", required=True, nargs="+",
                   help="image store directory (one per robot)")
    p.add_argument("--dsg", default=None,
                   help="dsg.json used to compose anchored poses (default: odometry)")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--embed-model", default="siglip2")
    p.add_argument("--no-place-attribution", action="store_true")
    args = p.parse_args(argv)

    from openset_query.perception.embedders import SigLip2Embedder
    from openset_query.perception.frame_index import TrajectoryFrameIndex
    from heracles.graph_interface import create_vector_indexes
    from heracles.query_interface import Neo4jWrapper

    from hydra_image_store import ImageStore, load_dsg

    store = ImageStore.open(*args.image_store)
    pose_lookup = load_dsg(args.dsg) if args.dsg else None

    uri = os.getenv("HERACLES_NEO4J_URI", "bolt://localhost:7690")
    auth = (os.getenv("HERACLES_NEO4J_USERNAME", "neo4j"),
            os.getenv("HERACLES_NEO4J_PASSWORD", "testpass"))
    db = Neo4jWrapper(uri, auth, atomic_queries=True); db.connect()
    embedder = SigLip2Embedder(device="cuda")
    idx = TrajectoryFrameIndex(db, embedder)
    n = idx.ingest_store(store, pose_lookup=pose_lookup, sample_stride=args.stride,
                         place_attribution=not args.no_place_attribution)
    print(f"ingested {n} TrajectoryFrame nodes")

    # Vector index for db.index.vector.queryNodes (idempotent; required by query_by_text).
    create_vector_indexes(db, dim=embedder.dim, model_name=args.embed_model)
    print(f"ensured vector indexes (dim={embedder.dim})")

    # Evidence provenance: CameraCalib nodes + HAS_CALIB from the store sensors and
    # Observation -> source-frame OBSERVED_IN edges (all idempotent).
    from heracles.graph_interface import (
        attach_camera_calibs,
        link_observations_to_frames,
    )

    n_calib = attach_camera_calibs(db, store)
    n_obs = link_observations_to_frames(db)
    print(f"provenance: {n_calib} HAS_CALIB edges, {n_obs} OBSERVED_IN edges")

    res = idx.query_by_text("car", k=5)
    print(f"query_by_text('car') -> {len(res)} frames; top:",
          [(r.get("nodeSymbol"), round(r.get("score", 0), 3)) for r in res[:3]])


if __name__ == "__main__":
    main()
