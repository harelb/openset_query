"""The scene-graph writes the server needs, on any heracles from master on.

The query + commit path needs only: create/update an ``Object`` node, write its
``Observation`` nodes and ``HAS_OBSERVATION`` edges, link observations to the
keyframe they were made in, and bump the map version. heracles master provides
the node/edge writers (``insert_nodes_to_db``, ``insert_edges``, ``node_to_dict``,
``link_observations_to_frames``); this module owns the rest so nothing depends
on a heracles fork:

- :func:`insert_observations`: the ``Observation`` MERGE. Same node shape as
  heracles' ``insert_observations_to_db`` (``nodeSymbol``, ``timestamp_ns``,
  ``obs_id``, ``frame_id``, ``rgb_path``, ``depth_path``, ``mask_path``,
  ``mask_rle``, ``bbox_2d``, ``score``, ``kept``) plus the detector provenance
  and admission evidence terms the admission policies read back.
- :func:`notify_map_change`: bump the map version through the newest heracles API
  that is importable (see the function), else with the same Cypher.
- :func:`link_observation_provenance`: ``OBSERVED_IN`` via heracles, and
  ``DEPICTS`` (TrajectoryFrame -> Object) only when heracles has
  ``insert_frame_edges``.

Optional heracles features (vector indexes, TrajectoryFrame ingest, epistemic
indexes) are probed with :func:`has` / :func:`optional_features`; code that
cannot work without one calls :func:`require`, which names what is missing.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import os
import time

logger = logging.getLogger("openset_query.heracles_compat")

OBSERVATIONS = "Observation"
MAP_STATE = "_MapState"
CONSOLE_STATE = "_State"

#: Optional heracles.graph_interface functions and what they turn on.
OPTIONAL = {
    "create_vector_indexes": "Neo4j vector indexes for keyframe embeddings "
    "(ingest_keyframes)",
    "insert_trajectory_frames_to_db": "TrajectoryFrame nodes (ingest_keyframes, "
    "TrajectoryFrameIndex.ingest)",
    "insert_frame_edges": "TrajectoryFrame edges and DEPICTS provenance edges",
    "query_similar_nodes": "TrajectoryFrame vector search "
    "(TrajectoryFrameIndex.query_by_*)",
    "create_epistemic_indexes": "indexes for the evidence store "
    "(EvidenceStore.ensure_schema)",
}

_logged = set()


def _attr(module: str, name: str):
    try:
        return getattr(importlib.import_module(module), name, None)
    except ImportError:
        return None


def has(name: str, module: str = "heracles.graph_interface") -> bool:
    """Whether the installed heracles provides ``module.name``."""
    return _attr(module, name) is not None


def require(name: str, feature: str, module: str = "heracles.graph_interface"):
    """``module.name``, or a RuntimeError naming the missing heracles feature."""
    fn = _attr(module, name)
    if fn is None:
        raise RuntimeError(
            f"{feature} needs {module}.{name}, which this heracles does not provide"
        )
    return fn


def optional_features() -> dict:
    """``{function: available}`` for the optional heracles functions."""
    return {name: has(name) for name in OPTIONAL}


def log_optional_features() -> None:
    """Log once which optional heracles features are off."""
    off = [n for n, ok in optional_features().items() if not ok]
    for name in off:
        if name not in _logged:
            _logged.add(name)
            logger.info(
                "heracles has no %s: %s off (not needed to query and commit)",
                name,
                OPTIONAL[name],
            )


DEFAULT_NEO4J_URI = "bolt://localhost:7687"


def neo4j_settings() -> tuple:
    """``(uri, (user, password))`` from ``HERACLES_NEO4J_URI`` (default
    bolt://localhost:7687), ``HERACLES_NEO4J_USERNAME`` (default neo4j) and
    ``HERACLES_NEO4J_PASSWORD`` (required: SystemExit when unset)."""
    password = os.getenv("HERACLES_NEO4J_PASSWORD")
    if not password:
        raise SystemExit(
            "HERACLES_NEO4J_PASSWORD is not set: export the Neo4j credentials "
            "(HERACLES_NEO4J_URI, HERACLES_NEO4J_USERNAME, HERACLES_NEO4J_PASSWORD)"
        )
    uri = os.getenv("HERACLES_NEO4J_URI", DEFAULT_NEO4J_URI)
    return uri, (os.getenv("HERACLES_NEO4J_USERNAME", "neo4j"), password)


# --- observations -------------------------------------------------------------

INSERT_OBSERVATIONS = f"""
WITH $observations AS observations
UNWIND observations AS obs
MERGE (o:{OBSERVATIONS} {{nodeSymbol: obs.nodeSymbol}})
SET o.timestamp_ns = obs.timestamp_ns,
    o.mask_file = obs.mask_file,
    o.obs_id = coalesce(obs.obs_id, o.obs_id),
    o.frame_id = coalesce(obs.frame_id, o.frame_id),
    o.rgb_path = coalesce(obs.rgb_path, o.rgb_path),
    o.depth_path = coalesce(obs.depth_path, o.depth_path),
    o.mask_path = coalesce(obs.mask_path, o.mask_path),
    o.mask_rle = coalesce(obs.mask_rle, o.mask_rle),
    o.bbox_2d = coalesce(obs.bbox_2d, o.bbox_2d),
    o.kept = coalesce(obs.kept, o.kept),
    o.bbox_2d_min_x = obs.bbox_2d_min_x,
    o.bbox_2d_min_y = obs.bbox_2d_min_y,
    o.bbox_2d_max_x = obs.bbox_2d_max_x,
    o.bbox_2d_max_y = obs.bbox_2d_max_y,
    o.score = coalesce(obs.score, o.score),
    o.detector = coalesce(obs.detector, o.detector),
    o.mechanism = coalesce(obs.mechanism, o.mechanism),
    o.target_prompt = coalesce(obs.target_prompt, o.target_prompt),
    o.target_score = coalesce(obs.target_score, o.target_score),
    o.control_max_score = coalesce(obs.control_max_score, o.control_max_score),
    o.margin = coalesce(obs.margin, o.margin),
    o.control_scores_json = coalesce(obs.control_scores_json, o.control_scores_json),
    o.depth_valid_px = coalesce(obs.depth_valid_px, o.depth_valid_px),
    o.view_x = coalesce(obs.view_x, o.view_x),
    o.view_y = coalesce(obs.view_y, o.view_y),
    o.view_z = coalesce(obs.view_z, o.view_z)
"""


def insert_observations(db, observations) -> None:
    """MERGE ``Observation`` nodes by ``nodeSymbol``. Re-observing the same
    (object, keyframe) updates the node in place; a value the new row lacks keeps
    the earlier one."""
    db.execute(INSERT_OBSERVATIONS, observations=list(observations))


def link_observation_provenance(db, depicts_pairs) -> None:
    """Best effort: ``(:Observation)-[:OBSERVED_IN]->(keyframe)`` with heracles'
    ``link_observations_to_frames``, and ``(:TrajectoryFrame)-[:DEPICTS]->(:Object)``
    when heracles has ``insert_frame_edges``."""
    link = _attr("heracles.graph_interface", "link_observations_to_frames")
    if link is not None:
        try:
            link(db)
        except Exception:
            logger.exception("OBSERVED_IN provenance linking failed (non-fatal)")
    pairs = [p for p in depicts_pairs if p is not None]
    insert_frame_edges = _attr("heracles.graph_interface", "insert_frame_edges")
    if pairs and insert_frame_edges is not None:
        try:
            insert_frame_edges(db, [], pairs)
        except Exception:
            logger.exception("DEPICTS provenance linking failed (non-fatal)")


# --- map version --------------------------------------------------------------


def _accepts_changes(fn) -> bool:
    try:
        return "changed_layers" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def _bump_console_state(db) -> None:
    # Only matched, never created: without a console there is nothing to bump.
    db.execute(
        f"MATCH (s:{CONSOLE_STATE}) "
        "SET s.version = coalesce(s.version, 0) + 1, s.updated = $t",
        t=time.time(),
    )


def read_state_version(db):
    """The console's ``_State {key: 'global'}.version``, or None without one."""
    records, _, _ = db.execute(
        f"MATCH (s:{CONSOLE_STATE} {{key: 'global'}}) RETURN s.version AS v"
    )
    if not records or records[0]["v"] is None:
        return None
    return int(records[0]["v"])


def read_map_version(db) -> int:
    """``_MapState.version`` (0 before the first bump)."""
    records, _, _ = db.execute(f"MATCH (m:{MAP_STATE}) RETURN max(m.version) AS v")
    if not records or records[0]["v"] is None:
        return 0
    return int(records[0]["v"])


def map_change_backend() -> str:
    """Which implementation :func:`notify_map_change` uses (for logs and tests)."""
    fn = _attr("heracles_ros.map_state", "notify_map_change")
    if fn is not None:
        return (
            "heracles_ros.map_state (changes)"
            if _accepts_changes(fn)
            else ("heracles_ros.map_state")
        )
    bump = _attr("heracles.map_reload", "bump_map_version")
    if bump is not None and _accepts_changes(bump):
        return "heracles.map_reload (changes)"
    if _attr("heracles.graph_interface", "notify_map_change") is not None:
        return "heracles.graph_interface"
    return "cypher"


def notify_map_change(db, changed_layers=None, changed_ids=None) -> tuple:
    """Tell map readers that the map changed. Returns ``(state_version, map_version)``;
    ``state_version`` is the console's ``_State`` counter, None when there is none.

    Uses, in order: ``heracles_ros.map_state.notify_map_change`` (with
    ``changed_layers`` / ``changed_ids`` when it takes them); heracles'
    ``map_reload.bump_map_version`` that records the change, plus the console
    counter; ``heracles.graph_interface.notify_map_change`` (older ADT4 heracles);
    else the same Cypher: bump ``_MapState.version`` and, if it exists, the
    console's ``_State.version``.
    """
    layers = None if changed_layers is None else list(changed_layers)
    ids = None if changed_ids is None else list(changed_ids)

    fn = _attr("heracles_ros.map_state", "notify_map_change")
    if fn is not None:
        if _accepts_changes(fn):
            map_version = fn(db, changed_layers=layers, changed_ids=ids)
        else:
            map_version = fn(db)
        return read_state_version(db), int(map_version)

    bump = _attr("heracles.map_reload", "bump_map_version")
    if bump is not None and _accepts_changes(bump):
        map_version = bump(db, changed_layers=layers, changed_ids=ids)
        _bump_console_state(db)
        return read_state_version(db), int(map_version)

    legacy = _attr("heracles.graph_interface", "notify_map_change")
    if legacy is not None:
        out = legacy(db, layers or [], ids or [])
        if isinstance(out, (tuple, list)) and len(out) == 2:
            return int(out[0]), int(out[1])
        return read_state_version(db), read_map_version(db)

    records, _, _ = db.execute(
        f"MERGE (m:{MAP_STATE}) "
        "SET m.version = coalesce(m.version, 0) + 1 "
        "RETURN m.version AS v"
    )
    _bump_console_state(db)
    return read_state_version(db), int(records[0]["v"])
