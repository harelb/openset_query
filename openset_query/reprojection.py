"""Reproject 2D masks into 3D and fuse detections across keyframes.

The geometry mirrors hydra's ``Camera::computeVertexMap``
(hydra/src/input/camera.cpp): for a pixel ``(u, v)`` with depth ``d`` in meters,
the camera-frame point is ``[d*(u-cx)/fx, d*(v-cy)/fy, d]``. We then map it to the
world frame via ``world_T_body @ body_T_sensor``.

This module is intentionally numpy-only (no torch / open3d) so it is cheap to unit
test and has no heavy dependencies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

logger = logging.getLogger("openset_query.reprojection")


@dataclass
class CameraCalib:
    """Static per-run calibration: the image store's ``sensors`` row / Neo4j
    ``CameraCalib`` node (intrinsics + extrinsics + depth storage convention)."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    # Meters per stored depth unit (e.g. 1e-3 for 16-bit millimeter PNGs).
    depth_scale: float
    # 4x4 transform: a point in the sensor frame -> body frame.
    body_T_sensor: np.ndarray

    def __post_init__(self):
        self.body_T_sensor = np.asarray(self.body_T_sensor, dtype=np.float64).reshape(4, 4)


@dataclass
class Object3D:
    """A reprojected detection in the world frame."""

    centroid: np.ndarray  # (3,)
    bbox_min: np.ndarray  # (3,)
    bbox_max: np.ndarray  # (3,)
    num_points: int
    label: str
    score: float
    color: tuple = (255, 0, 255)  # mean RGB (0-255) of the object's masked pixels
    # World-frame inlier points that produced this box, kept so fusion can pool
    # points across views and reduce once (pool-then-reduce) instead of unioning
    # per-view AABBs (which lets one bad view set the size). None for objects built
    # without a point cloud (e.g. synthetic/test detections).
    points: Optional[np.ndarray] = None

    @property
    def bbox_dimensions(self) -> np.ndarray:
        return self.bbox_max - self.bbox_min


def depth_to_meters(depth_raw: np.ndarray, calib: CameraCalib) -> np.ndarray:
    """Convert a stored depth image to float meters."""
    return depth_raw.astype(np.float64) * calib.depth_scale


def depth_mode_gate(
    depths: np.ndarray,
    bin_m: float = 0.15,
    peak_fraction: float = 0.05,
    min_points: int = 20,
) -> np.ndarray:
    """Keep only depths in the dominant *contiguous* depth cluster.

    A single 2D segmentation mask often laps onto the surface behind the object
    (a wall/floor a few meters further along the view ray). Those pixels form a
    separate depth population; backprojected, they stretch the object's box along
    the viewing axis. World-frame per-axis percentile trimming can't isolate them
    because in world coordinates they mix into the axis. Here we cut them in depth
    space, where they are cleanly separated.

    Returns a boolean mask over ``depths``. The object is assumed to be the densest
    depth surface (the mask is object-dominated by construction); we grow a
    contiguous interval out from the histogram peak, stopping at the first bin that
    falls below ``peak_fraction`` of the peak count — i.e. the gap to the
    background. Self-scaling: a thin chair yields a narrow interval, a deep couch
    keeps its contiguous bins. Too few points to cluster -> keep everything.
    """
    depths = np.asarray(depths, dtype=np.float64)
    n = depths.size
    if n == 0:
        return np.zeros(0, dtype=bool)
    if n < min_points or depths.max() - depths.min() <= bin_m:
        return np.ones(n, dtype=bool)

    # Explicit bin count so the last edge covers max without arange overshooting
    # into a spurious trailing bin that would strand the boundary point.
    lo_edge = float(depths.min())
    # -1e-9 so float error (e.g. 1.2/0.15 == 8.0000000000002) doesn't add a
    # spurious extra bin that strands the max in its own 1-point bin.
    n_bins = max(1, int(np.ceil((depths.max() - lo_edge) / bin_m - 1e-9)))
    edges = lo_edge + bin_m * np.arange(n_bins + 1)
    hist, edges = np.histogram(depths, bins=edges)
    peak = int(hist.argmax())
    thr = peak_fraction * hist[peak]

    lo = peak
    while lo > 0 and hist[lo - 1] > thr:
        lo -= 1
    hi = peak
    while hi < len(hist) - 1 and hist[hi + 1] > thr:
        hi += 1

    # Membership by bin index (robust to arange's float edges, where the top
    # edge can land a hair below the true max and drop boundary points).
    idx = np.clip(np.digitize(depths, edges) - 1, 0, len(hist) - 1)
    return (idx >= lo) & (idx <= hi)


def valid_depth_mask(depth_m: np.ndarray, min_depth_m: float = 0.2,
                     max_depth_m: float = 15.0) -> np.ndarray:
    """Pixels with a finite depth in ``[min_depth_m, max_depth_m]``."""
    with np.errstate(invalid="ignore"):
        return np.isfinite(depth_m) & (depth_m >= min_depth_m) & (depth_m <= max_depth_m)


def mask_to_points(
    mask: np.ndarray,
    depth_m: np.ndarray,
    calib: CameraCalib,
    world_T_body: np.ndarray,
    min_depth_m: float = 0.2,
    max_depth_m: float = 15.0,
    gate_background: bool = True,
    valid_depth: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Back-project the masked, valid-depth pixels into world-frame points.

    ``valid_depth`` (optional): the frame's precomputed ``isfinite & in-range``
    depth mask (:func:`valid_depth_mask`), shared by all its detections.

    Args:
        mask: (H, W) boolean array selecting the object's pixels.
        depth_m: (H, W) depth in meters (0 / NaN / inf treated as invalid).
        calib: intrinsics + extrinsics.
        world_T_body: (4, 4) body pose in the world frame for this keyframe.

    Returns:
        (N, 3) array of world-frame points (possibly empty).
    """
    world_T_body = np.asarray(world_T_body, dtype=np.float64).reshape(4, 4)

    # A mask whose resolution differs from the depth image (e.g. segmenter output
    # at a different size) would raise on the boolean-and; treat it as "no points"
    # rather than crashing the whole discovery over one frame.
    if mask.shape != depth_m.shape:
        logger.warning("mask shape %s != depth shape %s; skipping frame",
                       mask.shape, depth_m.shape)
        return np.empty((0, 3), dtype=np.float64)

    if valid_depth is None:
        valid_depth = valid_depth_mask(depth_m, min_depth_m, max_depth_m)
    valid = mask & valid_depth
    vs, us = np.nonzero(valid)
    if vs.size == 0:
        return np.empty((0, 3), dtype=np.float64)

    d = depth_m[vs, us]
    if gate_background:
        keep = depth_mode_gate(d)
        vs, us, d = vs[keep], us[keep], d[keep]
        if d.size == 0:
            return np.empty((0, 3), dtype=np.float64)
    x = d * (us - calib.cx) / calib.fx
    y = d * (vs - calib.cy) / calib.fy
    pts_sensor = np.stack([x, y, d], axis=1)  # (N, 3)

    # sensor -> body -> world
    world_T_sensor = world_T_body @ calib.body_T_sensor
    R = world_T_sensor[:3, :3]
    t = world_T_sensor[:3, 3]
    pts_world = pts_sensor @ R.T + t
    return pts_world


def points_to_object(
    points: np.ndarray,
    label: str,
    score: float,
    percentile_lo: float = 5.0,
    percentile_hi: float = 95.0,
) -> Optional[Object3D]:
    """Reduce a world-frame point cloud to a robust centroid + axis-aligned bbox.

    Trims the lowest/highest percentile along each axis before computing the bbox
    so a few mask-bleed outliers don't dominate.
    """
    if points.shape[0] == 0:
        return None

    lo = np.percentile(points, percentile_lo, axis=0)
    hi = np.percentile(points, percentile_hi, axis=0)
    inlier = np.all((points >= lo) & (points <= hi), axis=1)
    core = points[inlier] if inlier.any() else points

    centroid = core.mean(axis=0)
    bbox_min = core.min(axis=0)
    bbox_max = core.max(axis=0)
    return Object3D(
        centroid=centroid,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        num_points=int(core.shape[0]),
        label=label,
        score=float(score),
        points=core,
    )


def fuse_detections(
    detections: List[Object3D],
    centroid_dist_m: float = 0.75,
) -> List[Object3D]:
    """Greedily merge same-label detections whose centroids are close.

    Detections of the same label observed from multiple keyframes collapse into a
    single instance: centroids are score-weighted-averaged and bboxes unioned.
    Different labels never merge.
    """
    fused: List[Object3D] = []
    for det in sorted(detections, key=lambda d: d.score, reverse=True):
        match = None
        for f in fused:
            if f.label != det.label:
                continue
            if np.linalg.norm(f.centroid - det.centroid) <= centroid_dist_m:
                match = f
                break
        if match is None:
            # Copy so later merges don't mutate the caller's objects.
            m = Object3D(
                centroid=det.centroid.copy(),
                bbox_min=det.bbox_min.copy(),
                bbox_max=det.bbox_max.copy(),
                num_points=det.num_points,
                label=det.label,
                score=det.score,
            )
            m._fuse_w = max(float(det.score), 1e-6)  # accumulated weight
            fused.append(m)
        else:
            # Weight by accumulated score sum (not match.score, which becomes the
            # running max) so 3+ merges give a true score-weighted mean, order-free.
            w0 = getattr(match, "_fuse_w", max(float(match.score), 1e-6))
            w1 = max(float(det.score), 1e-6)
            match.centroid = (match.centroid * w0 + det.centroid * w1) / (w0 + w1)
            match._fuse_w = w0 + w1
            match.bbox_min = np.minimum(match.bbox_min, det.bbox_min)
            match.bbox_max = np.maximum(match.bbox_max, det.bbox_max)
            match.num_points += det.num_points
            match.score = max(match.score, det.score)
    return fused
