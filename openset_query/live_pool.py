"""Which live images the query server may search, and where they are in the map.

The web console mirrors the robots' image stores during a run (fleetlink class C6)
under the *current run's* output directory and publishes a small state file
(``console_live.json``, default ``~/adt4_output/console_live.json``)::

    {"version": 1, "written_ns": ..., "console_started_ns": ...,
     "live_store_root": "/home/rrg/adt4_output/console_20261113_173536/live",
     "robots": {"hamilton": {"store_id": "15eb161a2880",
                             "path": ".../live/hamilton/15eb161a2880/image_store",
                             "relocalized": true, "reason": "...",
                             "frame_align": {"ok": true, "parent": "map",
                                             "child": "hamilton/map",
                                             "T": [x, y, z, qw, qx, qy, qz]}}}}

Only the stores listed there are searched: mirrors of earlier runs (other output
directories, other store ids) are never loaded. A mirror's poses are the robot's own
scene-graph frame (``frame_align.child``); a live keyframe joins a query only while
the console says the robot is relocalized, and is posed into the prior map's frame
with ``frame_align.T`` (map <- robot frame) at that time. A commit from a live
keyframe is refused when the robot is no longer relocalized, its store changed, the
state file went stale, or the alignment moved since the search.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_STATE_FILE = "~/adt4_output/console_live.json"
#: the console rewrites the file every few seconds; older than this = console gone
STATE_MAX_AGE_S = 30.0
#: alignment drift between search and commit that invalidates the search's positions
COMMIT_MAX_SHIFT_M = 0.5
COMMIT_MAX_TURN_DEG = 5.0


class LiveNotAligned(RuntimeError):
    """A commit used a live image that is not (or no longer) in the map's frame."""


def read_state(path, now_ns: Optional[int] = None) -> dict:
    """``{"ok", "reason", "state"}``: the console's live state, or why it is unusable
    (missing, unreadable, stale)."""
    if not path:
        return {
            "ok": False,
            "reason": "no console live state file configured",
            "state": None,
        }
    p = os.path.expanduser(str(path))
    try:
        with open(p) as f:
            state = json.load(f)
    except FileNotFoundError:
        return {
            "ok": False,
            "reason": f"no console live state at {p} (console not mirroring)",
            "state": None,
        }
    except (OSError, ValueError) as e:
        return {
            "ok": False,
            "reason": f"console live state {p} unreadable: {e}",
            "state": None,
        }
    now_ns = time.time_ns() if now_ns is None else now_ns
    age = (now_ns - int(state.get("written_ns") or 0)) / 1e9
    if age > STATE_MAX_AGE_S:
        return {
            "ok": False,
            "reason": f"console live state is {age:.0f} s old (console stopped?)",
            "state": state,
        }
    return {"ok": True, "reason": None, "state": state}


def current_stores(state: Optional[dict]) -> list:
    """``[(robot, store_id, store_dir)]`` of the current run's mirrors that exist on disk."""
    out = []
    for robot, r in sorted(((state or {}).get("robots") or {}).items()):
        sid, path = (r or {}).get("store_id"), (r or {}).get("path")
        if not sid or not path:
            continue
        if os.path.isfile(os.path.join(path, "store.sqlite")):
            out.append((robot, str(sid), path))
    return out


def align_matrix(T) -> Optional[np.ndarray]:
    """4x4 from ``[x, y, z, qw, qx, qy, qz]`` (None if malformed)."""
    if not T or len(T) != 7:
        return None
    x, y, z, qw, qx, qy, qz = (float(v) for v in T)
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n < 1e-9:
        return None
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    M = np.eye(4)
    M[:3, :3] = [
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ]
    M[:3, 3] = [x, y, z]
    return M


def robot_alignment(state_info: dict, robot: str, store_id: Optional[str] = None):
    """``(matrix, None)`` when ``robot``'s live images may be used now (state fresh,
    relocalized, frame alignment present, same store), else ``(None, reason)``."""
    if not state_info.get("ok"):
        return None, state_info.get("reason") or "no console live state"
    r = ((state_info.get("state") or {}).get("robots") or {}).get(robot)
    if not r:
        return None, f"{robot} is not in the console's current run"
    if store_id is not None and str(r.get("store_id")) != str(store_id):
        return (
            None,
            f"{robot}'s image store changed (new run or Hydra restart): search again",
        )
    if not r.get("relocalized"):
        why = r.get("reason") or "not relocalized"
        return None, f"{robot} is not relocalized against the prior map ({why})"
    fa = r.get("frame_align") or {}
    if not fa.get("ok"):
        return None, f"{robot} has no map <- odom alignment yet"
    M = align_matrix(fa.get("T"))
    if M is None:
        return None, f"{robot}'s frame alignment is malformed"
    return M, None


def robot_of(kf) -> Optional[str]:
    ns = getattr(kf, "cache_ns", None) or ""
    parts = ns.split("/")
    return parts[1] if len(parts) >= 3 and parts[0] == "live" else None


def store_id_of(kf) -> Optional[str]:
    ns = getattr(kf, "cache_ns", None) or ""
    parts = ns.split("/")
    return parts[2] if len(parts) >= 3 and parts[0] == "live" else None


def is_live(kf) -> bool:
    return robot_of(kf) is not None


def posed_pool(keyframes, state_info: dict):
    """``(pool, excluded)``: the prior keyframes as they are, plus each live keyframe
    of a usable robot as a copy posed into the map (``world_T_body = map_T_robot @
    robot_T_body``, the robot-frame pose kept in ``robot_T_body``); ``excluded`` =
    ``{robot: reason}`` for robots whose images were left out."""
    pool, excluded, cache = [], {}, {}
    for kf in keyframes:
        robot = robot_of(kf)
        if robot is None:
            pool.append(kf)
            continue
        key = (robot, store_id_of(kf))
        if key not in cache:
            cache[key] = robot_alignment(state_info, robot, key[1])
        M, why = cache[key]
        if M is None:
            excluded[robot] = why
            continue
        base = getattr(kf, "robot_T_body", None)
        if base is None:
            base = kf.world_T_body
        posed = dataclasses.replace(kf, world_T_body=M @ base)
        posed.robot_T_body = base
        posed.map_T_robot = M
        pool.append(posed)
    return pool, excluded


def _shift(A, B):
    d = float(np.linalg.norm(A[:3, 3] - B[:3, 3]))
    R = A[:3, :3].T @ B[:3, :3]
    ang = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2))))
    return d, ang


def check_commit(keyframes, state_info: dict) -> None:
    """Raise :class:`LiveNotAligned` unless every live keyframe of a candidate is
    still usable with (about) the alignment it was searched with."""
    for kf in keyframes:
        robot = robot_of(kf)
        if robot is None:
            continue
        M, why = robot_alignment(state_info, robot, store_id_of(kf))
        if M is None:
            raise LiveNotAligned(
                f"not added: this detection uses {robot}'s live images and {why}"
            )
        used = getattr(kf, "map_T_robot", None)
        if used is None:
            raise LiveNotAligned(
                f"not added: {robot}'s live image was searched without a map alignment"
            )
        d, ang = _shift(used, M)
        if d > COMMIT_MAX_SHIFT_M or ang > COMMIT_MAX_TURN_DEG:
            raise LiveNotAligned(
                f"not added: {robot}'s map alignment moved {d:.2f} m / {ang:.1f} deg since the "
                "search (relocalization updated): search again"
            )
