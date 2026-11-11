"""Locate and load the RGB-D keyframes of a run from the Hydra image store.

All images live in the SQLite image store (``hydra_image_store``): one ``frames`` row
per keyframe (agent), sub-keyframe or observation frame, with the pose composed from
its anchor node. Two sources feed :class:`Keyframe`:

* :func:`load_keyframes_from_store` / :func:`load_keyframes_from_graph`: the store
  itself, with poses looked up in a DSG (or the stored odometry when the anchor is
  unknown);
* :func:`load_keyframes_from_db`: Neo4j ``Agent`` / ``SubKeyframe`` nodes carrying
  ``rgb_path`` / ``depth_path`` (absolute) and a ``HAS_CALIB`` ``CameraCalib`` node.

Depth images are uint16 and scaled by the calibration's ``depth_scale``.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np

from openset_query.reprojection import CameraCalib

logger = logging.getLogger("openset_query.keyframe_store")

_TS_RE = re.compile(r"(\d{6,})_(?:rgb|depth)")


@dataclass
class Keyframe:
    timestamp_ns: int
    rgb_path: str
    depth_path: str
    world_T_body: np.ndarray  # (4, 4)
    calib: CameraCalib
    #: Image-store ids (None for keyframes that did not come from a store row).
    frame_id: Optional[int] = None
    sensor_id: Optional[int] = None
    #: Store the frame/sensor ids belong to, when it is not the store SAM3 writes
    #: into (e.g. a live mirror); None = the written (prior-map) store.
    store_dir: Optional[str] = None
    #: Namespace of the embedding-cache key (frame ids of different stores collide).
    cache_ns: Optional[str] = None

    @property
    def camera_position(self) -> np.ndarray:
        """Camera origin in the world frame."""
        world_T_sensor = self.world_T_body @ self.calib.body_T_sensor
        return world_T_sensor[:3, 3]


def calib_from_store(store, sensor_id: int) -> Optional[CameraCalib]:
    c = store.calib(sensor_id)
    if c is None:
        return None
    return CameraCalib(
        fx=c.fx, fy=c.fy, cx=c.cx, cy=c.cy, width=c.width, height=c.height,
        depth_scale=c.depth_scale, body_T_sensor=c.body_T_sensor,
    )


def _pose_from_center_quat(center, quat_xyzw) -> np.ndarray:
    """Build a 4x4 world_T_body from a position + quaternion (x, y, z, w)."""
    x, y, z, w = (float(v) for v in quat_xyzw)
    n = (x * x + y * y + z * z + w * w) ** 0.5
    if n > 0:
        x, y, z, w = x / n, y / n, z / n, w / n
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray([float(c) for c in center[:3]], dtype=np.float64)
    return T


def keyframe_from_frame(
    store,
    frame,
    pose_lookup=None,
    calib_cache: Optional[dict] = None,
) -> Optional[Keyframe]:
    """Build a Keyframe from an image-store ``Frame``.

    ``pose_lookup`` is anything ``ImageStore.frame_pose`` accepts (spark_dsg graph,
    ``load_dsg(...)``, mapping or callable); without one the stored odometry is used.
    Frames without an RGB or depth file, or without a calibration, are skipped.
    """
    if frame.rgb_path is None or frame.depth_path is None:
        return None
    calib_cache = calib_cache if calib_cache is not None else {}
    if frame.sensor_id not in calib_cache:
        calib_cache[frame.sensor_id] = calib_from_store(store, frame.sensor_id)
    calib = calib_cache[frame.sensor_id]
    if calib is None:
        logger.warning("No calibration for sensor %s (frame %s); skipping", frame.sensor_id, frame.frame_id)
        return None
    world_T_body, _ = store.body_pose(frame, pose_lookup)
    return Keyframe(
        timestamp_ns=int(frame.timestamp_ns),
        rgb_path=str(frame.rgb_path),
        depth_path=str(frame.depth_path),
        world_T_body=np.asarray(world_T_body, dtype=np.float64).reshape(4, 4),
        calib=calib,
        frame_id=int(frame.frame_id),
        sensor_id=int(frame.sensor_id),
    )


def load_keyframes_from_store(
    store,
    pose_lookup=None,
    near_position: Optional[np.ndarray] = None,
    radius_m: Optional[float] = None,
    include_subkeyframes: bool = False,
    masked_frame_ts: Optional[Iterable[int]] = None,
) -> List[Keyframe]:
    """The agent nodes' own keyframes (all keyframes with ``include_subkeyframes``) of
    an ``ImageStore``, time-ordered. A node's own keyframe is the one at its stamp when
    ``pose_lookup`` is a DSG, else the identity-anchored one (``agent_keyframes``).

    Optionally restrict to poses within ``radius_m`` of ``near_position``; hide the
    frames whose timestamp is in ``masked_frame_ts`` (Tier-2 observation masking).
    """
    from openset_query.perception.agent_keyframes import keyframes as store_keyframes
    from openset_query.perception.agent_keyframes import stamps_from_graph

    keyframes: List[Keyframe] = []
    calib_cache: dict = {}
    for frame in store_keyframes(store, include_subkeyframes, stamps_from_graph(pose_lookup)):
        kf = keyframe_from_frame(store, frame, pose_lookup, calib_cache)
        if kf is None:
            continue
        if near_position is not None and radius_m is not None:
            if np.linalg.norm(kf.world_T_body[:3, 3] - near_position) > radius_m:
                continue
        keyframes.append(kf)
    if masked_frame_ts:
        from openset_query.eval.masking import filter_keyframes

        keyframes = filter_keyframes(keyframes, masked_frame_ts)
    keyframes.sort(key=lambda k: k.timestamp_ns)
    return keyframes


def load_keyframes_from_graph(
    G,
    store,
    near_position: Optional[np.ndarray] = None,
    radius_m: Optional[float] = None,
    include_subkeyframes: bool = False,
) -> List[Keyframe]:
    """Keyframes of ``store`` posed from the (optimized) agent nodes of a DSG.

    Optionally restrict to poses within ``radius_m`` of ``near_position``
    (used to scope a query to a region/place).
    """
    return load_keyframes_from_store(
        store, pose_lookup=G, near_position=near_position, radius_m=radius_m,
        include_subkeyframes=include_subkeyframes,
    )


def _timestamp_of(row_ts, rgb_path: str) -> Optional[int]:
    if row_ts:
        return int(row_ts)
    m = _TS_RE.search(os.path.basename(rgb_path or ""))
    return int(m.group(1)) if m else None


def load_keyframes_from_db(
    db,
    near_position: Optional[np.ndarray] = None,
    radius_m: Optional[float] = None,
    include_subkeyframes: bool = False,
    masked_frame_ts=None,
) -> List[Keyframe]:
    """Load keyframes by querying Agent (and SubKeyframe) nodes from Neo4j.

    heracles' db_to_spark_dsg does NOT reconstruct the AGENTS layer, so keyframes are
    recovered from the nodes themselves: ``rgb_path`` / ``depth_path`` (absolute paths
    into the image store), the optimized ``center`` + ``rot_*`` pose and the
    ``(:CameraCalib)<-[:HAS_CALIB]`` intrinsics (``body_T_sensor`` is 16 row-major
    floats).

    ``masked_frame_ts`` (observation masking): an iterable of
    hidden frame timestamps — keyframes whose ``timestamp_ns`` is masked are
    excluded from the returned pool (``eval.masking.filter_keyframes``).
    """
    keyframes: List[Keyframe] = []

    label_match = "(a:Agent OR a:SubKeyframe)" if include_subkeyframes else "a:Agent"
    records, _, _ = db.execute(
        f"MATCH (a) WHERE {label_match} AND a.rgb_path IS NOT NULL "
        "AND a.depth_path IS NOT NULL "
        "OPTIONAL MATCH (a)-[:HAS_CALIB]->(c:CameraCalib) "
        "RETURN a.frame_id AS frame_id, a.timestamp_ns AS ts, "
        "a.rgb_path AS rgb, a.depth_path AS depth, a.center AS center, "
        "a.rot_x AS rot_x, a.rot_y AS rot_y, a.rot_z AS rot_z, a.rot_w AS rot_w, "
        "c.sensor_id AS sensor_id, c.fx AS fx, c.fy AS fy, c.cx AS cx, c.cy AS cy, "
        "c.width AS width, c.height AS height, c.depth_scale AS depth_scale, "
        "c.body_T_sensor AS body_T_sensor"
    )
    for r in records:
        if not r["rgb"] or not r["depth"]:
            continue
        if r["center"] is None or r["rot_w"] is None:
            continue
        if r["fx"] is None or r["body_T_sensor"] is None:
            logger.warning("No CameraCalib for keyframe %s; skipping", r["rgb"])
            continue
        c = r["center"]
        pos = np.array([c[0], c[1], c[2]], dtype=np.float64)
        if near_position is not None and radius_m is not None:
            if np.linalg.norm(pos - near_position) > radius_m:
                continue
        ts = _timestamp_of(r["ts"], r["rgb"])
        if ts is None:
            logger.warning("No timestamp for keyframe %s; skipping", r["rgb"])
            continue
        calib = CameraCalib(
            fx=float(r["fx"]), fy=float(r["fy"]), cx=float(r["cx"]), cy=float(r["cy"]),
            width=int(r["width"]), height=int(r["height"]),
            depth_scale=float(r["depth_scale"]),
            body_T_sensor=np.asarray(r["body_T_sensor"], dtype=np.float64).reshape(4, 4),
        )
        pose = _pose_from_center_quat(pos, (r["rot_x"], r["rot_y"], r["rot_z"], r["rot_w"]))
        keyframes.append(Keyframe(
            timestamp_ns=ts, rgb_path=r["rgb"], depth_path=r["depth"],
            world_T_body=pose, calib=calib,
            frame_id=None if r["frame_id"] is None else int(r["frame_id"]),
            sensor_id=None if r["sensor_id"] is None else int(r["sensor_id"]),
        ))

    if masked_frame_ts:
        from openset_query.eval.masking import filter_keyframes

        keyframes = filter_keyframes(keyframes, masked_frame_ts)
    keyframes.sort(key=lambda k: k.timestamp_ns)
    return keyframes


def read_rgb(keyframe: Keyframe) -> np.ndarray:
    """Read the RGB image as an (H, W, 3) uint8 RGB array."""
    img = _imread(keyframe.rgb_path)
    if img.ndim == 3 and img.shape[2] == 3:
        # OpenCV reads BGR; convert to RGB.
        return img[:, :, ::-1].copy()
    return img


def read_depth_meters(keyframe: Keyframe) -> np.ndarray:
    """Read the depth image and convert to float meters."""
    raw = _imread(keyframe.depth_path, unchanged=True)
    return raw.astype(np.float64) * keyframe.calib.depth_scale


def _imread(path: str, unchanged: bool = False) -> np.ndarray:
    try:
        import cv2

        flag = cv2.IMREAD_UNCHANGED if unchanged else cv2.IMREAD_COLOR
        img = cv2.imread(path, flag)
        if img is None:
            raise FileNotFoundError(path)
        return img
    except ImportError:
        from PIL import Image

        return np.asarray(Image.open(path))


def embed_cache_path(store_dir, model: str) -> str:
    """Embedding cache next to the image store (``<store>/cache/<model>_emb.npz``),
    keyed by frame_id (see ``EmbeddingFilter.frame_key``)."""
    return os.path.join(str(store_dir), "cache", f"{model}_emb.npz")


def image_store_paths_from_db(db) -> List[str]:
    """Image-store directories recorded by the heracles ingest
    (``(:_GraphMetadata {key: 'image_store'}).image_store_paths``); [] if none."""
    try:
        records, _, _ = db.execute(
            "MATCH (m:_GraphMetadata {key: 'image_store'}) "
            "RETURN m.image_store_paths AS paths")
    except Exception as e:
        logger.warning("could not read image_store_paths from the DB: %s", e)
        return []
    for r in records:
        if r["paths"]:
            return [str(p) for p in r["paths"]]
    return []


def resolve_store_dirs(root) -> List[str]:
    """Image-store directories under ``root``: ``root`` itself if it holds a
    ``store.sqlite``, else ``root/image_store``; [] when neither does."""
    for cand in (os.fspath(root), os.path.join(os.fspath(root), "image_store")):
        if os.path.exists(os.path.join(cand, "store.sqlite")):
            return [cand]
    return []


def rgb_by_timestamp(store_dirs, kinds=None) -> dict:
    """``{timestamp_ns: rgb_path}`` over the store frames that have an RGB file
    (all kinds by default; a keyframe wins over an observation frame on a tie)."""
    from hydra_image_store import ImageStore

    if not store_dirs:
        return {}
    store = ImageStore.open(*store_dirs)
    try:
        out: dict = {}
        for f in sorted(store.frames(kind=kinds), key=lambda f: (-int(f.kind), f.timestamp_ns)):
            if f.rgb_path is not None:
                out[int(f.timestamp_ns)] = str(f.rgb_path)
        return out
    finally:
        store.close()
