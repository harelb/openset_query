"""Tests for TrajectoryFrameIndex — ingestion, semantic, and spatial queries.

These tests require a running Neo4j instance.  They use testcontainers-neo4j
when available, or fall back to the NEO4J_URI / NEO4J_AUTH environment
variables.  Skip cleanly when neither is available.
"""

import os
import time

import pytest

# ---------------------------------------------------------------------------
# Availability guards — skip at collection time if nothing is usable
# ---------------------------------------------------------------------------

_TESTCONTAINERS_AVAILABLE = False
try:
    from testcontainers.neo4j import Neo4jContainer  # type: ignore

    _TESTCONTAINERS_AVAILABLE = True
except ImportError:
    pass

_NEO4J_ENV_URI = os.environ.get("NEO4J_URI", "")

if not _TESTCONTAINERS_AVAILABLE and not _NEO4J_ENV_URI:
    pytest.skip(
        "Skipping: testcontainers-neo4j is not installed and NEO4J_URI env var is not set.",
        allow_module_level=True,
    )

# ---------------------------------------------------------------------------
# Imports (only reached when a Neo4j instance is available)
# ---------------------------------------------------------------------------

import numpy as np
from PIL import Image

from heracles.graph_interface import (
    create_vector_indexes,
    insert_trajectory_frames_to_db,
)
from heracles.query_interface import Neo4jWrapper
from openset_query.perception.embedders import StubEmbedder
from openset_query.perception.frame_index import (
    TrajectoryFrameIndex,
    _frame_node_symbol,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def db():
    """Provide a connected Neo4jWrapper for the test module."""
    if _TESTCONTAINERS_AVAILABLE:
        container = Neo4jContainer("neo4j:5.18")
        container.start()
        uri = container.get_connection_url()
        auth = ("neo4j", container.NEO4J_ADMIN_PASSWORD)
        wrapper = Neo4jWrapper(uri, auth, atomic_queries=True)
        wrapper.connect()
        yield wrapper
        wrapper.close()
        container.stop()
    else:
        uri = _NEO4J_ENV_URI
        auth_str = os.environ.get("NEO4J_AUTH", "neo4j/neo4j")
        user, password = auth_str.split("/", 1)
        auth = (user, password)
        wrapper = Neo4jWrapper(uri, auth, atomic_queries=True)
        wrapper.connect()
        yield wrapper
        wrapper.close()


@pytest.fixture(autouse=True)
def clean_db(db):
    """Wipe the database before each test to ensure isolation."""
    db.execute("MATCH (n) DETACH DELETE n")
    for idx in ("object_embedding", "observation_embedding", "trajectoryframe_embedding"):
        try:
            db.execute(f"DROP INDEX {idx} IF EXISTS")
        except Exception:
            pass
    yield


def _await_index(db, index_name: str, timeout: float = 15.0) -> None:
    """Poll until the named index is ONLINE or timeout is reached."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            records, _, _ = db.execute("SHOW INDEXES")
            for row in records:
                if row.get("name") == index_name and row.get("state") == "ONLINE":
                    return
        except Exception:
            pass
        time.sleep(0.2)
    time.sleep(1.0)  # last-resort


def _make_synthetic_images(tmp_path, count: int, stride: int = 1) -> list:
    """Create *count* synthetic ``*_rgb.jpg`` files in *tmp_path*.

    Images are 32x32 pixels with a unique colour per file so StubEmbedder
    produces distinct (but deterministic) embeddings.  File mtimes are set
    sequentially so the sort order is well-defined.

    Returns a list of Path objects.
    """
    paths = []
    for i in range(count):
        # Each image has a unique solid colour so hash-based embedding differs
        colour = (i * 7 % 256, i * 13 % 256, i * 17 % 256)
        img = Image.new("RGB", (32, 32), colour)
        p = tmp_path / f"frame_{i:04d}_rgb.jpg"
        img.save(p)
        # Set deterministic mtime (seconds since epoch)
        os.utime(p, (1_700_000_000 + i, 1_700_000_000 + i))
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_ingest_writes_frames(tmp_path, db):
    """Ingest 20 synthetic frames; assert DB contains exactly 20 TrajectoryFrame nodes."""
    _make_synthetic_images(tmp_path, 20)
    embedder = StubEmbedder(dim=64)

    # Create vector index so TrajectoryFrame embeddings can be indexed
    create_vector_indexes(db, dim=64, model_name="stub")
    _await_index(db, "trajectoryframe_embedding")

    idx = TrajectoryFrameIndex(db, embedder)
    count = idx.ingest(tmp_path, sample_stride=1, place_attribution=False)

    assert count == 20, f"Expected 20 frames written, got {count}"

    records, _, _ = db.execute("MATCH (t:TrajectoryFrame) RETURN count(t) AS c")
    assert records[0]["c"] == 20, f"Expected 20 nodes in DB, got {records[0]['c']}"


def test_query_by_text_returns_top_k(tmp_path, db):
    """Plant a frame with a known embedding; query_by_text top-1 should be that frame."""
    embedder = StubEmbedder(dim=64)

    create_vector_indexes(db, dim=64, model_name="stub")
    _await_index(db, "trajectoryframe_embedding")

    # Plant 5 random-colour images
    paths = _make_synthetic_images(tmp_path, 5)

    # Determine the embedding of "red square" and find which image embedding
    # is closest to it (we'll plant that embedding manually for the target frame).
    target_text = "red square"
    text_vec = embedder.embed_text([target_text])[0]  # (64,) float32

    # We'll manually insert a frame whose embedding is exactly the text vector
    # so it is guaranteed to be the top-1 result.
    target_ns = "F(999999)"
    target_emb = text_vec.tolist()
    insert_trajectory_frames_to_db(
        db,
        [
            {
                "nodeSymbol": target_ns,
                "timestamp_ns": 1_700_000_999,
                "path": str(paths[0]),
                "pos_x": 0.0,
                "pos_y": 0.0,
                "pos_z": 0.0,
                "pose_qw": 1.0,
                "pose_qx": 0.0,
                "pose_qy": 0.0,
                "pose_qz": 0.0,
                "place_id": None,
                "embedding": target_emb,
                "embedding_dim": len(target_emb),
                "embedding_model": "stub",
            }
        ],
    )

    # Also ingest the other frames through the index so there is noise
    idx = TrajectoryFrameIndex(db, embedder)
    idx.ingest(tmp_path, sample_stride=1, place_attribution=False)

    results = idx.query_by_text(target_text, k=5)
    assert len(results) > 0, "Expected at least one result"
    top = results[0]
    assert top["nodeSymbol"] == target_ns, (
        f"Expected top-1 to be planted frame {target_ns}, got {top['nodeSymbol']}"
    )


def test_lookup_near_radius(tmp_path, db):
    """Plant 5 frames at z=0..4; lookup_near([0,0,0], 2.5m) returns exactly 3."""
    embedder = StubEmbedder(dim=64)

    create_vector_indexes(db, dim=64, model_name="stub")

    # Insert 5 frames at (0, 0, 0), (0, 0, 1), ..., (0, 0, 4)
    frames = []
    for i in range(5):
        frames.append(
            {
                "nodeSymbol": f"F({i})",
                "timestamp_ns": 1_700_000_000 + i,
                "path": f"/synthetic/frame_{i:04d}_rgb.jpg",
                "pos_x": 0.0,
                "pos_y": 0.0,
                "pos_z": float(i),
                "pose_qw": 1.0,
                "pose_qx": 0.0,
                "pose_qy": 0.0,
                "pose_qz": 0.0,
                "place_id": None,
                "embedding": None,
                "embedding_dim": None,
                "embedding_model": None,
            }
        )
    insert_trajectory_frames_to_db(db, frames)

    idx = TrajectoryFrameIndex(db, embedder)
    results = idx.lookup_near([0.0, 0.0, 0.0], radius_m=2.5, k=10)

    # Distances are 0, 1, 2, 3, 4 — only 0, 1, 2 are within 2.5 m
    assert len(results) == 3, (
        f"Expected 3 frames within 2.5 m, got {len(results)}: "
        f"{[r['nodeSymbol'] for r in results]}"
    )
    returned_symbols = {r["nodeSymbol"] for r in results}
    assert returned_symbols == {"F(0)", "F(1)", "F(2)"}, (
        f"Unexpected symbols: {returned_symbols}"
    )
    # Verify ascending distance order
    distances = [r["distance"] for r in results]
    assert distances == sorted(distances), f"Results not sorted by distance: {distances}"


def test_query_by_text_with_radius(tmp_path, db):
    """Spatial + text filter: only the in-radius planted frame should be top-1."""
    embedder = StubEmbedder(dim=64)

    create_vector_indexes(db, dim=64, model_name="stub")
    _await_index(db, "trajectoryframe_embedding")

    target_text = "blue sky"
    text_vec = embedder.embed_text([target_text])[0].tolist()

    # Two frames: one at (0,0,0), one at (0,0,100) — far away
    near_ns = "F(10)"
    far_ns = "F(11)"
    insert_trajectory_frames_to_db(
        db,
        [
            {
                "nodeSymbol": near_ns,
                "timestamp_ns": 1_700_001_000,
                "path": "/synthetic/near_rgb.jpg",
                "pos_x": 0.0,
                "pos_y": 0.0,
                "pos_z": 0.0,
                "pose_qw": 1.0,
                "pose_qx": 0.0,
                "pose_qy": 0.0,
                "pose_qz": 0.0,
                "place_id": None,
                "embedding": text_vec,
                "embedding_dim": len(text_vec),
                "embedding_model": "stub",
            },
            {
                "nodeSymbol": far_ns,
                "timestamp_ns": 1_700_001_001,
                "path": "/synthetic/far_rgb.jpg",
                "pos_x": 0.0,
                "pos_y": 0.0,
                "pos_z": 100.0,
                "pose_qw": 1.0,
                "pose_qx": 0.0,
                "pose_qy": 0.0,
                "pose_qz": 0.0,
                "place_id": None,
                "embedding": text_vec,  # same embedding, only spatial filter differs
                "embedding_dim": len(text_vec),
                "embedding_model": "stub",
            },
        ],
    )

    idx = TrajectoryFrameIndex(db, embedder)

    # With a 10-m radius around the origin, only the near frame should appear
    results = idx.query_by_text(
        target_text,
        k=10,
        radius_m=10.0,
        center_xyz=[0.0, 0.0, 0.0],
    )
    symbols = {r["nodeSymbol"] for r in results}
    assert near_ns in symbols, f"Near frame {near_ns} not in results: {symbols}"
    assert far_ns not in symbols, f"Far frame {far_ns} should not be in results: {symbols}"
