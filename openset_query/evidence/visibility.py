"""Visibility materializer: the missing negative-evidence instrument.

``detect_frustum_aware`` already decides per frame whether an object is in
view (``object_discovery._in_frustum``) — and throws that view-set away. This
module recomputes it from what the provenance layer persisted (frame poses +
``HAS_CALIB`` -> ``CameraCalib``) and materializes it as::

    (frame)-[:VIEWS {u, v, depth_m}]->(:Object)

so that "which covered frames could actually have seen X" becomes a Cypher
query, and a SearchEpisode's *visual* coverage (fraction of objects viewed by
at least one COVERED frame) is computable — grid coverage is not visual
coverage.

Projection semantics are gate-for-gate identical to ``_in_frustum``
(z > 0.1 m in front, optional ``max_depth_m``, ``margin``-padded image
bounds); a parity test in tests/test_evidence_visibility.py locks the two
together. Occlusion is NOT modelled here either — a wall between camera and
object still counts as a view, exactly as in the frustum early-stop.

Deliberately NOT re-exported from ``openset_query.evidence`` (this module
needs numpy at module scope; the package core stays stdlib-only). Import it
directly: ``from openset_query.evidence import visibility``.
"""

from __future__ import annotations

import logging

import numpy as np

from .store import OBJECTS, SEARCH_EPISODES

logger = logging.getLogger("evidence.visibility")

VIEWS = "VIEWS"  # in lockstep with heracles.constants (drift-guard test)
COVERED = "COVERED"

# Per-label pose properties, matching what the heracles writers persist:
# TrajectoryFrame: pos_* + pose_q* (insert_trajectory_frames_to_db);
# Agent: pos_* + rot_* (agent_to_dict); SubKeyframe: center point + rot_*
# (insert_subkeyframes_to_db).
_FRAME_POSE_SPECS = (
    (
        "TrajectoryFrame",
        ("n.pos_x", "n.pos_y", "n.pos_z"),
        ("n.pose_qx", "n.pose_qy", "n.pose_qz", "n.pose_qw"),
    ),
    (
        "Agent",
        ("n.pos_x", "n.pos_y", "n.pos_z"),
        ("n.rot_x", "n.rot_y", "n.rot_z", "n.rot_w"),
    ),
    (
        "SubKeyframe",
        ("n.center.x", "n.center.y", "n.center.z"),
        ("n.rot_x", "n.rot_y", "n.rot_z", "n.rot_w"),
    ),
)
FRAME_LABELS = tuple(label for label, _, _ in _FRAME_POSE_SPECS)

_WRITE_BATCH = 5000


def _pose_from_quat(x, y, z, qx, qy, qz, qw) -> np.ndarray:
    """4x4 world_T_body from position + quaternion, identical math to
    keyframe_store._pose_from_center_quat (normalized, xyzw)."""
    from openset_query.keyframe_store import _pose_from_center_quat

    return _pose_from_center_quat((x, y, z), (qx, qy, qz, qw))


def _load_frames(db, label: str, pos_exprs, quat_exprs) -> list:
    """[(nodeSymbol, world_T_body 4x4, calib dict)] for frames with pose+calib."""
    px, py, pz = pos_exprs
    qx, qy, qz, qw = quat_exprs
    records, _, _ = db.execute(
        f"""
    MATCH (n:{label})-[:HAS_CALIB]->(c:CameraCalib)
    WHERE {px} IS NOT NULL AND {qw} IS NOT NULL
    RETURN n.nodeSymbol AS ns,
           {px} AS x, {py} AS y, {pz} AS z,
           {qx} AS qx, {qy} AS qy, {qz} AS qz, {qw} AS qw,
           c.fx AS fx, c.fy AS fy, c.cx AS cx, c.cy AS cy,
           c.width AS width, c.height AS height, c.body_T_sensor AS bts
    ORDER BY n.nodeSymbol
    """
    )
    frames = []
    for r in records:
        bts = np.asarray(r["bts"], dtype=np.float64).reshape(4, 4)
        world_T_body = _pose_from_quat(
            r["x"], r["y"], r["z"], r["qx"], r["qy"], r["qz"], r["qw"]
        )
        calib = {
            "fx": float(r["fx"]),
            "fy": float(r["fy"]),
            "cx": float(r["cx"]),
            "cy": float(r["cy"]),
            "width": int(r["width"]),
            "height": int(r["height"]),
            "body_T_sensor": bts,
        }
        frames.append((r["ns"], world_T_body, calib))
    return frames


def _load_objects(db, object_symbols=None) -> list:
    """[(nodeSymbol, centroid (3,))] for objects with a stored position."""
    clause = "AND o.nodeSymbol IN $syms" if object_symbols is not None else ""
    records, _, _ = db.execute(
        f"""
    MATCH (o:{OBJECTS}) WHERE o.pos_x IS NOT NULL {clause}
    RETURN o.nodeSymbol AS ns, o.pos_x AS x, o.pos_y AS y, o.pos_z AS z
    ORDER BY o.nodeSymbol
    """,
        **({"syms": list(object_symbols)} if object_symbols is not None else {}),
    )
    return [
        (r["ns"], np.array([r["x"], r["y"], r["z"]], dtype=np.float64)) for r in records
    ]


def _project_batch(
    world_T_body, calib, centroids, margin: float = 0.0, max_depth_m=None
):
    """Project world points into one frame; returns (u, v, depth, in_view).

    Gate-for-gate the vectorized form of ``object_discovery._in_frustum``:
    z > 0.1 in front of the sensor, optional depth ceiling, margin-padded
    image bounds (parity-tested against the original).
    """
    centroids = np.asarray(centroids, dtype=np.float64).reshape(-1, 3)
    T = world_T_body @ calib["body_T_sensor"]  # world_T_sensor
    hom = np.hstack([centroids, np.ones((centroids.shape[0], 1))])
    p = np.linalg.inv(T) @ hom.T  # (4, N) sensor-frame points
    z = p[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = calib["fx"] * p[0] / z + calib["cx"]
        v = calib["fy"] * p[1] / z + calib["cy"]
    ok = z > 0.1
    if max_depth_m is not None:
        ok &= ~(z > max_depth_m)
    ok &= (
        (-margin <= u)
        & (u < calib["width"] + margin)
        & (-margin <= v)
        & (v < calib["height"] + margin)
    )
    return u, v, z, ok


def materialize_views(
    db,
    *,
    max_depth_m=None,
    margin: float = 0.0,
    object_symbols=None,
    frame_labels=FRAME_LABELS,
    refresh: bool = False,
) -> int:
    """Batch-write ``(frame)-[:VIEWS {u, v, depth_m}]->(:Object)`` edges.

    ``max_depth_m`` should be the same value the detection config used
    (``config.max_depth_m``) so the materialized view-set matches
    ``detect_frustum_aware``'s skip decisions. MERGE-based, so idempotent;
    ``refresh=True`` first drops existing VIEWS edges from the selected frame
    labels (use after objects moved). Returns the number of (frame, object)
    view pairs written.
    """
    objects = _load_objects(db, object_symbols)
    if not objects:
        return 0
    centroids = np.stack([c for _, c in objects])
    total = 0
    for label, pos_exprs, quat_exprs in _FRAME_POSE_SPECS:
        if label not in frame_labels:
            continue
        if refresh:
            db.execute(f"MATCH (f:{label})-[r:{VIEWS}]->(:{OBJECTS}) DELETE r")
        frames = _load_frames(db, label, pos_exprs, quat_exprs)
        rows = []
        for frame_symbol, world_T_body, calib in frames:
            u, v, z, ok = _project_batch(
                world_T_body, calib, centroids, margin=margin, max_depth_m=max_depth_m
            )
            for i in np.flatnonzero(ok):
                rows.append(
                    {
                        "f": frame_symbol,
                        "o": objects[i][0],
                        "u": float(u[i]),
                        "v": float(v[i]),
                        "d": float(z[i]),
                    }
                )
        for start in range(0, len(rows), _WRITE_BATCH):
            db.execute(
                f"""
            UNWIND $rows AS row
            MATCH (f:{label} {{nodeSymbol: row.f}})
            MATCH (o:{OBJECTS} {{nodeSymbol: row.o}})
            MERGE (f)-[v:{VIEWS}]->(o)
            SET v.u = row.u, v.v = row.v, v.depth_m = row.d
            """,
                rows=rows[start : start + _WRITE_BATCH],
            )
        total += len(rows)
        if frames:
            logger.info(
                "VIEWS materialized: %d %s frames -> %d view pairs",
                len(frames),
                label,
                len(rows),
            )
    return total


def episode_visual_coverage(
    db, episode_id: str, *, object_symbols=None, write: bool = True
):
    """Fraction of objects viewed by >= 1 frame COVERED by the episode.

    Reads the materialized VIEWS edges (run :func:`materialize_views` first).
    Denominator = objects with a stored position (optionally restricted to
    ``object_symbols``). Returns None when there is no such object. With
    ``write=True`` the fraction is persisted as ``visual_coverage`` on the
    SearchEpisode node — beside (never replacing) the episode's own search
    ``coverage``, which the producer reported on different semantics.
    """
    clause = "AND o.nodeSymbol IN $syms" if object_symbols is not None else ""
    params = {"syms": list(object_symbols)} if object_symbols is not None else {}
    records, _, _ = db.execute(
        f"MATCH (o:{OBJECTS}) WHERE o.pos_x IS NOT NULL {clause} RETURN count(o) AS n",
        **params,
    )
    n_total = records[0]["n"] if records else 0
    if not n_total:
        return None
    records, _, _ = db.execute(
        f"""
    MATCH (e:{SEARCH_EPISODES} {{episode_id: $eid}})-[:{COVERED}]->(f)
          -[:{VIEWS}]->(o:{OBJECTS})
    WHERE o.pos_x IS NOT NULL {clause}
    RETURN count(DISTINCT o) AS n
    """,
        eid=episode_id,
        **params,
    )
    n_viewed = records[0]["n"] if records else 0
    fraction = n_viewed / n_total
    if write:
        db.execute(
            f"MATCH (e:{SEARCH_EPISODES} {{episode_id: $eid}}) "
            "SET e.visual_coverage = $frac",
            eid=episode_id,
            frac=fraction,
        )
    return fraction
