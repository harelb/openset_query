"""Query-time object discovery: run SAM3 over stored keyframes and upsert results.

Pipeline (see plan Part E):
  1. select keyframes (optionally scoped to a region),
  2. segment + reproject + collect per-keyframe observations,
  3. fuse same-label detections across keyframes,
  4. dedup against existing Object nodes,
  5. write DB-first (Neo4j source of truth), then mirror in-memory.
"""

from __future__ import annotations

import copy
import json
import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from openset_query.evidence.admission import (
    ADMISSION_OBSERVATIONS as _ADMISSION_OBSERVATIONS,
    ADMISSION_POLICY_VERSION as _ADMISSION_POLICY_VERSION,
    ADMISSION_REASON as _ADMISSION_REASON,
    ADMISSION_STATUS as _ADMISSION_STATUS,
)
from openset_query.config import Sam3FrontendConfig
from openset_query import keyframe_store as ks
from openset_query.reprojection import (
    Object3D,
    mask_to_points,
    points_to_object,
)

logger = logging.getLogger("openset_query.object_discovery")


@dataclass
class Observation:
    timestamp_ns: int
    mask_file: str
    bbox_2d: tuple  # (min_x, min_y, max_x, max_y)
    rgb_path: str = ""  # source keyframe RGB (absolute path in the image store)
    score: Optional[float] = None  # detector confidence for THIS view
    # -- admission evidence terms (raw, per-view; persisted by _obs_dict) ----------
    #: Depth-valid mask pixels that survived reprojection (mask/depth validity).
    depth_valid_px: Optional[int] = None
    #: THIS view's own reprojected centroid (world), pre-fusion — the
    #: geometric-consistency input for the admission policies.
    view_xyz: Optional[tuple] = None
    #: SigLIP2 cosine of the detection crop against the queried prompt, the
    #: prompt itself, and the per-control-prompt cosines (raw terms;
    #: forbids persisting only the derived margin). Filled by
    #: attach_crop_evidence; None when no policy/embedder is configured.
    target_prompt: Optional[str] = None
    target_score: Optional[float] = None
    control_scores: Optional[dict] = None
    #: Source keyframe (carries the image-store frame/sensor ids needed to record the
    #: observation in the store).
    keyframe: Optional[ks.Keyframe] = field(default=None, repr=False, compare=False)
    #: Image-store ids/paths, filled by :func:`record_observations` (absolute paths).
    frame_id: Optional[int] = None
    obs_id: Optional[int] = None
    depth_path: Optional[str] = None
    mask_path: Optional[str] = None
    kept: Optional[bool] = None
    #: The detection's boolean HxW mask, kept in memory so it reaches the image store
    #: (RLE + kept mask PNG) even without ``mask_output_dir``. A dry-run query
    #: compacts it into ``mask_rle`` (store RLE JSON) + ``mask_shape`` while pending.
    mask: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    mask_rle: Optional[str] = field(default=None, repr=False, compare=False)
    mask_shape: Optional[tuple] = None

    def mask_array(self) -> Optional[np.ndarray]:
        """The detection mask (decoded from ``mask_rle`` when compacted), or None."""
        if self.mask is not None:
            return self.mask
        if self.mask_rle and self.mask_shape:
            from hydra_python.image_store import rle_decode

            return rle_decode(self.mask_rle, tuple(self.mask_shape))
        return None

    def compact_mask(self) -> None:
        """Replace the in-memory mask by its RLE (a few hundred bytes)."""
        if self.mask is None:
            return
        from hydra_python.image_store import rle_encode

        self.mask_shape = tuple(int(v) for v in self.mask.shape[:2])
        self.mask_rle = json.dumps(rle_encode(self.mask), separators=(",", ":"))
        self.mask = None


@dataclass
class DiscoveredObject:
    obj: Object3D
    observations: List[Observation] = field(default_factory=list)
    #: distinct source frames of the observations (set by the association step)
    num_frames: Optional[int] = None


def _segment_kf(segmenter, kf, rgb, prompts):
    """Detections of ``prompts`` in keyframe ``kf`` (``rgb`` already read): through
    the segmenter's feature cache when it has one (``segment_keyframe``)."""
    fn = getattr(segmenter, "segment_keyframe", None)
    if fn is not None:
        return fn(kf, rgb, prompts)
    return segmenter.segment(rgb, prompts)


def associate(
    per_view: List["DiscoveredObject"], config: Sam3FrontendConfig, embeddings=None
) -> List["DiscoveredObject"]:
    """Group a query's per-view detections into objects: the 3D clustering of
    :mod:`association` (``config.association == "cluster"``, default) or the
    legacy greedy centroid fusion (``"centroid"``)."""
    if getattr(config, "association", "cluster") == "centroid":
        return _fuse(
            per_view,
            config.fuse_centroid_dist_m,
            config.depth_percentile_lo,
            config.depth_percentile_hi,
        )
    from openset_query.association import cluster_detections, params_from_config

    return cluster_detections(
        per_view,
        params_from_config(config),
        embeddings,
        config.depth_percentile_lo,
        config.depth_percentile_hi,
    )


def _mean_color(rgb: np.ndarray, mask: np.ndarray) -> tuple:
    """Mean RGB (0-255 ints) of the object's masked pixels — its 'primary' color."""
    try:
        px = rgb[mask]
        if px.size == 0:
            return (255, 0, 255)
        m = px.reshape(-1, px.shape[-1]).mean(axis=0)
        return (int(m[0]), int(m[1]), int(m[2]))
    except Exception:
        return (255, 0, 255)


def _bbox_from_mask(mask: np.ndarray) -> tuple:
    vs, us = np.nonzero(mask)
    if us.size == 0:
        return (0, 0, 0, 0)
    return (int(us.min()), int(vs.min()), int(us.max()), int(vs.max()))


def _views_from_detections(
    dets, rgb, depth_m, kf, config, mask_output_dir=None
) -> List["DiscoveredObject"]:
    """Reproject one frame's detections into per-view ``DiscoveredObject``s.

    The depth-validity mask is computed once per frame, and a detection whose
    mask has fewer valid-depth pixels than ``min_mask_depth_pixels`` is dropped
    before back-projection (the background gate only removes points, so this
    never drops a detection the full path would keep)."""
    from openset_query.reprojection import valid_depth_mask

    out: List[DiscoveredObject] = []
    if not dets:
        return out
    valid = None
    for det in dets:
        if det.mask.shape != depth_m.shape:
            pts = mask_to_points(
                det.mask,
                depth_m,
                kf.calib,
                kf.world_T_body,
                min_depth_m=config.min_depth_m,
                max_depth_m=config.max_depth_m,
            )
        else:
            if valid is None:
                valid = valid_depth_mask(
                    depth_m, config.min_depth_m, config.max_depth_m
                )
            if np.count_nonzero(det.mask & valid) < config.min_mask_depth_pixels:
                continue
            pts = mask_to_points(
                det.mask,
                depth_m,
                kf.calib,
                kf.world_T_body,
                min_depth_m=config.min_depth_m,
                max_depth_m=config.max_depth_m,
                valid_depth=valid,
            )
        if pts.shape[0] < config.min_mask_depth_pixels:
            continue
        o3d = points_to_object(
            pts,
            det.label,
            det.score,
            config.depth_percentile_lo,
            config.depth_percentile_hi,
        )
        if o3d is None:
            continue
        o3d.color = _mean_color(rgb, det.mask)  # primary color of the detected object
        mask_file = (
            _save_mask(det.mask, kf, det.label, mask_output_dir)
            if mask_output_dir
            else ""
        )
        out.append(
            DiscoveredObject(
                obj=o3d,
                observations=[
                    Observation(
                        kf.timestamp_ns,
                        mask_file,
                        _bbox_from_mask(det.mask),
                        kf.rgb_path,
                        score=float(det.score),
                        depth_valid_px=int(pts.shape[0]),
                        view_xyz=tuple(float(v) for v in o3d.centroid),
                        keyframe=kf,
                        mask=np.asarray(det.mask, dtype=bool),
                    )
                ],
            )
        )
    return out


def detect_in_keyframes(
    keyframes: Sequence[ks.Keyframe],
    segmenter,
    prompts: Sequence[str],
    config: Sam3FrontendConfig,
    mask_output_dir: Optional[str] = None,
) -> List[DiscoveredObject]:
    """Run SAM3 over keyframes, reproject, and fuse same-label detections."""
    per_view: List[DiscoveredObject] = []

    for kf in keyframes:
        try:
            rgb = ks.read_rgb(kf)
            depth_m = ks.read_depth_meters(kf)
        except Exception as e:
            logger.warning("Failed to read keyframe %s: %s", kf.rgb_path, e)
            continue

        detections = _segment_kf(segmenter, kf, rgb, prompts)
        per_view.extend(
            _views_from_detections(
                detections, rgb, depth_m, kf, config, mask_output_dir
            )
        )

    return associate(per_view, config)


def _fuse(
    items: List[DiscoveredObject],
    dist_m: float,
    percentile_lo: float = 5.0,
    percentile_hi: float = 95.0,
) -> List[DiscoveredObject]:
    """Greedy same-label fusion that preserves contributing observations.

    Operates on RAW per-view detections (each contributes its original score once),
    so the result is order-independent and the centroid is a true score-weighted
    mean. Inserts a copy of each object so repeated calls (e.g. the frustum loop's
    re-fusion over the accumulator) never mutate the caller's per-view objects.

    Box estimation is **pool-then-reduce**: the inlier points of every matched view
    are pooled and reduced once (percentile-trimmed) into a single box. This weights
    views by how many points they contribute — a close, dense view outvotes a far,
    grazing one — instead of unioning per-view AABBs, where a single elongated view
    would set the whole object's size. Views without a point cloud (synthetic/test
    detections) fall back to the old min/max union so their box is still defined.
    """
    fused: List[DiscoveredObject] = []
    for item in sorted(items, key=lambda d: d.obj.score, reverse=True):
        match = None
        for f in fused:
            if f.obj.label != item.obj.label:
                continue
            if np.linalg.norm(f.obj.centroid - item.obj.centroid) <= dist_m:
                match = f
                break
        if match is None:
            o = copy.copy(item.obj)
            o._fuse_w = max(float(item.obj.score), 1e-6)  # accumulated weight
            o._pool = [item.obj.points] if item.obj.points is not None else []
            fused.append(DiscoveredObject(obj=o, observations=list(item.observations)))
        else:
            a, b = match.obj, item.obj
            wa = getattr(a, "_fuse_w", max(float(a.score), 1e-6))
            wb = max(float(b.score), 1e-6)
            a.centroid = (a.centroid * wa + b.centroid * wb) / (wa + wb)
            a._fuse_w = wa + wb
            a.bbox_min = np.minimum(
                a.bbox_min, b.bbox_min
            )  # union fallback (no points)
            a.bbox_max = np.maximum(a.bbox_max, b.bbox_max)
            a.num_points += b.num_points
            a.score = max(a.score, b.score)
            if b.points is not None:
                a._pool.append(b.points)
            match.observations.extend(item.observations)

    # Pool-then-reduce: recompute geometry from the pooled points where available.
    for f in fused:
        pool = getattr(f.obj, "_pool", [])
        if pool:
            reduced = points_to_object(
                np.vstack(pool), f.obj.label, f.obj.score, percentile_lo, percentile_hi
            )
            if reduced is not None:
                f.obj.centroid = reduced.centroid
                f.obj.bbox_min = reduced.bbox_min
                f.obj.bbox_max = reduced.bbox_max
                f.obj.num_points = reduced.num_points
    return fused


def _save_mask(mask: np.ndarray, kf: ks.Keyframe, label: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    safe = label.replace(" ", "_").replace("/", "_")
    # Include the mask's 2D bbox so multiple same-label detections in one frame
    # (e.g. two chairs) don't collide on one filename and overwrite each other.
    x0, y0, x1, y1 = _bbox_from_mask(mask)
    path = os.path.join(
        out_dir, f"sam3_{kf.timestamp_ns}_{safe}_{x0}-{y0}-{x1}-{y1}_mask.png"
    )
    try:
        import cv2

        cv2.imwrite(path, mask.astype(np.uint8) * 255)
    except ImportError:
        from PIL import Image

        Image.fromarray(mask.astype(np.uint8) * 255).save(path)
    return path


def _in_frustum(
    centroid, kf, margin: float = 0.0, max_depth_m: Optional[float] = None
) -> bool:
    """True if a world-frame point projects inside the keyframe's image (in front),
    and within ``max_depth_m`` of the camera if given.

    The depth gate matters for the frustum early-stop: without it, an object far
    beyond the sensor's reliable range still counts as "viewed" by a frame and
    suppresses SAM3 there, missing nearer distinct instances. (Occlusion by closer
    geometry is NOT modelled — a wall between camera and object won't be detected.)
    """
    T = kf.world_T_body @ kf.calib.body_T_sensor  # world_T_sensor
    p = np.linalg.inv(T) @ np.array([centroid[0], centroid[1], centroid[2], 1.0])
    if p[2] <= 0.1:
        return False
    if max_depth_m is not None and p[2] > max_depth_m:
        return False
    u = kf.calib.fx * p[0] / p[2] + kf.calib.cx
    v = kf.calib.fy * p[1] / p[2] + kf.calib.cy
    return bool(
        -margin <= u < kf.calib.width + margin
        and -margin <= v < kf.calib.height + margin
    )


def _any_in_frustum(
    centroids: np.ndarray, kf, max_depth_m: Optional[float] = None
) -> bool:
    """Vectorised ``any(_in_frustum(c, kf, max_depth_m=...) for c in centroids)``
    (same gates: in front by > 0.1 m, optional depth ceiling, inside the image)."""
    if centroids.size == 0:
        return False
    T = kf.world_T_body @ kf.calib.body_T_sensor
    R, t = T[:3, :3], T[:3, 3]
    p = (centroids - t) @ R  # world -> sensor (R orthonormal: inv = R^T)
    z = p[:, 2]
    ok = z > 0.1
    if max_depth_m is not None:
        ok &= z <= max_depth_m
    if not ok.any():
        return False
    p, z = p[ok], z[ok]
    u = kf.calib.fx * p[:, 0] / z + kf.calib.cx
    v = kf.calib.fy * p[:, 1] / z + kf.calib.cy
    return bool(
        np.any((u >= 0) & (u < kf.calib.width) & (v >= 0) & (v < kf.calib.height))
    )


def _group_centroids(views: List[DiscoveredObject], config) -> np.ndarray:
    """Per-frame object centroids of the frustum loop: cluster membership (cheap,
    no point pooling) -> score-weighted member centroids, one row per object."""
    from openset_query.association import cluster_indices, params_from_config

    groups = cluster_indices(views, params_from_config(config))
    out = np.empty((len(groups), 3))
    for gi, g in enumerate(groups):
        w = np.array([max(float(views[i].obj.score), 1e-6) for i in g])
        c = np.stack([np.asarray(views[i].obj.centroid, dtype=np.float64) for i in g])
        out[gi] = (c * w[:, None]).sum(0) / w.sum()
    return out


def detect_frustum_aware(
    ranked,
    segmenter,
    prompts,
    config,
    max_frames: int = 20,
    patience: int = 3,
    mask_output_dir: Optional[str] = None,
    views_out: Optional[list] = None,
):
    """Frustum-coverage early-stop detection.

    ``ranked`` is [(keyframe, score)] sorted by descending text-image similarity.
    Frames whose frustum already views an object found for this query are skipped
    (redundant view), and the loop stops after ``patience`` processed frames yield
    no new object or after ``max_frames``. This runs SAM3 on far fewer frames than
    a fixed top-K while still recovering multiple instances.

    ``views_out`` (a list) receives the raw per-view detections, e.g. for a final
    re-association with appearance embeddings.
    """
    import time as _time

    t_read = t_seg = t_reproj = 0.0
    n_skipped = 0
    all_views: List[
        DiscoveredObject
    ] = []  # raw per-view detections (fused fresh each frame)
    fused: List[DiscoveredObject] = []
    # Found objects' centroids. "cluster": cheap per-frame grouping (the full
    # pooled-point merge runs once, after the loop); "centroid": the legacy fusion.
    clustering = getattr(config, "association", "cluster") != "centroid"
    centroids = np.empty((0, 3))
    processed = 0
    no_new = 0
    for kf, _score in ranked:
        if processed >= max_frames:
            break
        if len(centroids) and _any_in_frustum(centroids, kf, config.max_depth_m):
            n_skipped += 1
            continue  # already views a found object — skip the redundant SAM3 call
        try:
            _t = _time.perf_counter()
            rgb = ks.read_rgb(kf)
            depth_m = ks.read_depth_meters(kf)
            t_read += _time.perf_counter() - _t
        except Exception as e:
            logger.warning("Failed to read keyframe %s: %s", kf.rgb_path, e)
            continue
        _t = _time.perf_counter()
        dets = _segment_kf(segmenter, kf, rgb, prompts)
        t_seg += _time.perf_counter() - _t
        _t = _time.perf_counter()
        per_view = _views_from_detections(
            dets, rgb, depth_m, kf, config, mask_output_dir
        )
        before = len(centroids)
        # Fuse the RAW accumulated views (not the already-fused set) so each
        # detection contributes its original score exactly once (correct weighting).
        all_views.extend(per_view)
        if per_view:
            if clustering:
                centroids = _group_centroids(all_views, config)
            else:
                fused = associate(all_views, config)
                centroids = np.array(
                    [np.asarray(d.obj.centroid, dtype=np.float64) for d in fused]
                ).reshape(-1, 3)
        t_reproj += _time.perf_counter() - _t
        processed += 1
        no_new = no_new + 1 if len(centroids) == before else 0
        if no_new >= patience:
            break
    if clustering and all_views:
        _t = _time.perf_counter()
        fused = associate(all_views, config)
        t_reproj += _time.perf_counter() - _t
    logger.info(
        "frustum-aware: %d frames processed, %d skipped -> %d objects | "
        "timings: read=%.2fs seg(SAM3)=%.2fs reproject=%.2fs",
        processed,
        n_skipped,
        len(fused),
        t_read,
        t_seg,
        t_reproj,
    )
    if views_out is not None:
        views_out.extend(all_views)
    return fused


def detect_batched(
    keyframes, segmenter, prompts, config, mask_output_dir=None, fixed_batch=None
):
    """Run SAM3 over all keyframes in a single batched forward, then reproject+fuse.

    Best for multi-instance queries: trades the frustum early-stop's adaptivity for
    one batched GPU forward over a fixed frame set.
    """
    import time as _time

    rgbs, depths, kfs = [], [], []
    t = _time.perf_counter()
    for kf in keyframes:
        try:
            rgbs.append(ks.read_rgb(kf))
            depths.append(ks.read_depth_meters(kf))
            kfs.append(kf)
        except Exception as e:
            logger.warning("Failed to read keyframe %s: %s", kf.rgb_path, e)
    t_read = _time.perf_counter() - t
    if not kfs:
        return []

    t = _time.perf_counter()
    per_image = segmenter.segment_batch(rgbs, prompts, fixed_batch=fixed_batch)
    t_seg = _time.perf_counter() - t

    t = _time.perf_counter()
    per_view = []
    for kf, depth_m, rgb, dets in zip(kfs, depths, rgbs, per_image):
        per_view.extend(
            _views_from_detections(dets, rgb, depth_m, kf, config, mask_output_dir)
        )
    fused = associate(per_view, config)
    t_reproj = _time.perf_counter() - t
    logger.info(
        "batched: %d frames (1 forward) -> %d objects | timings: "
        "read=%.2fs seg(SAM3)=%.2fs reproject=%.2fs",
        len(kfs),
        len(fused),
        t_read,
        t_seg,
        t_reproj,
    )
    return fused


# ---------------------------------------------------------------------------
# Admission evidence: SigLIP2 target/control scores for each detection crop
# ---------------------------------------------------------------------------


def crop_embeddings(views: List[DiscoveredObject], ef, pad_frac: float = 0.08) -> list:
    """L2-normalised SigLIP2 embedding of each per-view detection's crop (its first
    observation's padded 2D box), or None when the crop cannot be read: the optional
    appearance gate of :func:`associate`. One batched forward for the query."""
    from PIL import Image

    crops, idx, images = [], [], {}
    for i, d in enumerate(views):
        o = d.observations[0] if d.observations else None
        if o is None or not o.rgb_path:
            continue
        x0, y0, x1, y1 = o.bbox_2d
        if x1 <= x0 or y1 <= y0:
            continue
        try:
            img = images.get(o.rgb_path)
            if img is None:
                img = images[o.rgb_path] = Image.open(o.rgb_path).convert("RGB")
        except Exception:
            continue
        px, py = int((x1 - x0) * pad_frac), int((y1 - y0) * pad_frac)
        crops.append(
            img.crop(
                (
                    max(0, x0 - px),
                    max(0, y0 - py),
                    min(img.width, x1 + px),
                    min(img.height, y1 + py),
                )
            )
        )
        idx.append(i)
    out = [None] * len(views)
    if crops:
        emb = ef.embed_pil(crops)
        for k, i in enumerate(idx):
            out[i] = emb[k]
    return out


def attach_crop_evidence(
    discovered: List[DiscoveredObject],
    ef,
    prompt: str,
    control_prompts: Sequence[str],
    pad_frac: float = 0.08,
) -> int:
    """Fill each observation's SigLIP2 target/control scores in place.

    For every observation, the detection's 2D bbox (padded by ``pad_frac`` of
    its size) is cropped out of its source RGB and scored against
    ``[prompt] + control_prompts`` with ``ef.score_crops`` — one batched warm
    forward for the whole query. Observations whose source image cannot be
    read (or whose bbox is degenerate) are left unscored: a missing margin is
    "ungradable", never a pass or a fail. Returns the number of observations
    scored.
    """
    from PIL import Image

    texts = [prompt] + [c for c in control_prompts if c]
    targets: List[Observation] = []
    crops = []
    images: dict = {}
    for disc in discovered:
        for o in disc.observations:
            if not o.rgb_path:
                continue
            x0, y0, x1, y1 = o.bbox_2d
            if x1 <= x0 or y1 <= y0:
                continue
            try:
                img = images.get(o.rgb_path)
                if img is None:
                    img = Image.open(o.rgb_path).convert("RGB")
                    images[o.rgb_path] = img
            except Exception as e:
                logger.warning("crop evidence: cannot read %s: %s", o.rgb_path, e)
                continue
            pad_x = int((x1 - x0) * pad_frac)
            pad_y = int((y1 - y0) * pad_frac)
            crops.append(
                img.crop(
                    (
                        max(0, x0 - pad_x),
                        max(0, y0 - pad_y),
                        min(img.width, x1 + pad_x),
                        min(img.height, y1 + pad_y),
                    )
                )
            )
            targets.append(o)

    if not crops:
        return 0
    sims = ef.score_crops(crops, texts)  # (n_crops, 1 + n_controls)
    for i, o in enumerate(targets):
        o.target_prompt = prompt
        o.target_score = float(sims[i, 0])
        o.control_scores = {c: float(sims[i, 1 + j]) for j, c in enumerate(texts[1:])}
    return len(targets)


# ---------------------------------------------------------------------------
# Graph write (DB-first, then in-memory mirror)
# ---------------------------------------------------------------------------


def _existing_objects(db, G):
    """Return list of (nodeSymbol, centroid np.array, class/name) for dedup."""
    out = []
    if db is not None:
        from heracles import constants

        records, _, _ = db.execute(
            f"MATCH (o:{constants.OBJECTS}) RETURN o.nodeSymbol AS sym, "
            "o.center AS c, o.class AS cls, o.name AS name"
        )
        for r in records:
            c = r["c"]
            if c is None:
                continue
            out.append(
                (
                    r["sym"],
                    np.array([c[0], c[1], c[2]], dtype=np.float64),
                    r.get("cls") or r.get("name"),
                )
            )
    elif G is not None:
        import spark_dsg

        for node in G.get_layer(spark_dsg.DsgLayers.OBJECTS).nodes:
            a = node.attributes
            out.append(
                (
                    node.id.str(True),
                    np.asarray(a.position, dtype=np.float64),
                    getattr(a, "name", None),
                )
            )
    return out


def _next_object_index(db, G) -> int:
    import spark_dsg
    from heracles.graph_interface import str_to_ns_value

    max_idx = -1
    for sym, _, _ in _existing_objects(db, G):
        try:
            # sym is a string like "O5"; convert to the index it encodes.
            idx = spark_dsg.NodeSymbol(str_to_ns_value(sym)).category_id
            max_idx = max(max_idx, idx)
        except Exception as e:
            # Don't silently underestimate the max — that would risk reusing an
            # index and colliding with an existing object.
            logger.warning("could not parse object symbol %r for indexing: %s", sym, e)
    return max_idx + 1


def open_sam3_writer(image_store, robot_id: int = 0):
    """Writer onto the image store that records SAM3 observations (None if no store)."""
    if not image_store:
        return None
    from hydra_python.image_store import ImageStore

    path = image_store[0] if isinstance(image_store, (list, tuple)) else image_store
    return ImageStore.open_writer(path, robot_id=robot_id)


def _node_value(sym: str) -> Optional[int]:
    try:
        from heracles.graph_interface import str_to_ns_value

        return int(str_to_ns_value(sym))
    except Exception as e:
        logger.warning("cannot map object symbol %r to a node id: %s", sym, e)
        return None


def _ensure_frame(writer, reader, kf: ks.Keyframe) -> int:
    """frame_id of ``kf`` in the writer's store (reusing the existing frame row)."""
    from hydra_python.image_store import FrameKind

    # A keyframe from another store (e.g. a live mirror) carries ids of THAT store:
    # never reuse them here, add the frame by reference instead.
    same_store = _same_store(getattr(kf, "store_dir", None), writer.root)
    if same_store and kf.frame_id is not None and reader.frame(kf.frame_id) is not None:
        return kf.frame_id
    sensor = (
        kf.sensor_id
        if same_store and kf.sensor_id is not None and reader.calib(kf.sensor_id)
        else None
    )
    if sensor is None:
        c = kf.calib
        sensor = writer.add_sensor(
            "sam3",
            c.width,
            c.height,
            c.fx,
            c.fy,
            c.cx,
            c.cy,
            depth_scale=c.depth_scale,
            body_T_sensor=c.body_T_sensor,
        )
    return writer.add_frame_ref(
        sensor,
        kf.timestamp_ns,
        kind=FrameKind.OBSERVATION,
        world_T_body_odom=kf.world_T_body,
        rgb=kf.rgb_path or None,
        depth=kf.depth_path or None,
    )


def _same_store(store_dir, root) -> bool:
    if not store_dir:
        return True  # keyframes of the written store (prior map / Neo4j pool)
    try:
        return os.path.samefile(str(store_dir), str(root))
    except OSError:
        return False


def _add_sam3_observation(writer, node, fid, mask, mask_path, **kw) -> int:
    """SAM3 observation of ``node`` in frame ``fid`` with ``mask`` stored as RLE and a
    mask PNG: ``mask_path`` when given, else ``mask`` written to a PNG the store copies
    to ``masks/r<robot>/<obs_id>.png``."""
    from openset_query.store_writeback import add_sam3_observation

    if mask_path is not None or mask is None:
        return add_sam3_observation(
            writer, node, fid, mask=mask, mask_path=mask_path, **kw
        )
    import tempfile

    import cv2

    with tempfile.TemporaryDirectory(prefix="sam3_mask_") as tmp:
        png = os.path.join(tmp, "mask.png")
        cv2.imwrite(png, np.asarray(mask, dtype=np.uint8) * 255)
        return add_sam3_observation(writer, node, fid, mask=mask, mask_path=png, **kw)


def record_observations(
    writer, sym: str, observations, new_object: bool = False
) -> None:
    """Record ``observations`` of object ``sym`` in the image store.

    Each observation reuses its keyframe's existing frame row (by frame_id, else
    (sensor, timestamp)) and becomes a kept SAM3 observation of the node
    (``add_sam3_observation``). The store ids and absolute paths are written back onto
    the :class:`Observation` so the Neo4j ``Observation`` node can carry them.
    ``new_object`` drops stale SAM3 rows left by an earlier, cleared object that used
    the same node id. Re-recording an identical (frame, bbox) is a no-op.
    """
    if writer is None:
        return
    from openset_query.store_writeback import sam3_track_id

    node = _node_value(sym)
    if node is None:
        return
    track = sam3_track_id(node)
    reader = writer.reader()
    with writer.batch():
        if new_object:
            writer.conn.execute(
                "DELETE FROM observations WHERE robot_id = ? AND track_id = ?",
                (writer.robot_id, track),
            )
        existing = {
            (ob.frame_id, ob.bbox2d): ob.obs_id
            for ob in reader.track_observations(writer.robot_id, track)
        }
        recorded = []  # (Observation, frame_id, bbox) of this call
        for o in observations:
            if o.keyframe is None:
                continue
            fid = _ensure_frame(writer, reader, o.keyframe)
            bbox = tuple(int(v) for v in o.bbox_2d)
            if (fid, bbox) not in existing:
                mask_path = (
                    o.mask_file if o.mask_file and os.path.exists(o.mask_file) else None
                )
                # the in-memory mask: RLE always, and the kept mask PNG unless a
                # mask file from mask_output_dir is linked instead
                mask = o.mask_array()
                existing[(fid, bbox)] = _add_sam3_observation(
                    writer,
                    node,
                    fid,
                    mask,
                    mask_path,
                    timestamp_ns=o.timestamp_ns,
                    bbox2d=bbox,
                    score=float(o.score or 0.0),
                    kept=True,
                )
            recorded.append((o, fid, bbox))
        by_id = {
            ob.obs_id: ob for ob in reader.track_observations(writer.robot_id, track)
        }
        for o, fid, bbox in recorded:
            ob = by_id[existing[(fid, bbox)]]
            o.frame_id, o.obs_id, o.kept = ob.frame_id, ob.obs_id, ob.kept
            o.rgb_path = str(ob.frame.rgb_path) if ob.frame.rgb_path else o.rgb_path
            o.depth_path = str(ob.frame.depth_path) if ob.frame.depth_path else None
            o.mask_path = str(ob.mask_path) if ob.mask_path else None


def _make_object_attrs(disc: DiscoveredObject):
    import spark_dsg

    obj = disc.obj
    # KhronosObjectAttributes (not plain ObjectNodeAttributes) so the object
    # round-trips as a Khronos object in the viewer. Images are not an attribute:
    # they live in the image store (Observation rows).
    attrs = spark_dsg.KhronosObjectAttributes()
    attrs.position = obj.centroid
    attrs.name = obj.label
    try:
        c = spark_dsg.Color()
        c.r, c.g, c.b = int(obj.color[0]), int(obj.color[1]), int(obj.color[2])
        attrs.color = c  # object's primary color -> stored as color_r/g/b in Neo4j
    except Exception as e:
        logger.debug("could not set object color: %s", e)
    try:
        dims = (obj.bbox_max - obj.bbox_min).astype(np.float32)
        center = ((obj.bbox_min + obj.bbox_max) / 2.0).astype(np.float32)
        # spark_dsg.BoundingBox(dimensions, center) builds a valid AABB.
        attrs.bounding_box = spark_dsg.BoundingBox(dims, center)
    except Exception as e:
        logger.debug("Could not set bounding_box: %s", e)
    return attrs


# Base-object properties the dedup-match path overwrites on an *existing* object.
# On the first dedup we snapshot each as ``o._sam3_orig_<p>`` so a later
# ``POST /api/sam3/clear`` can restore the base-map object verbatim and strip the
# SAM3 mutation — i.e. clear is non-destructive to the base map (it only deletes
# objects SAM3 *created*, flagged ``sam3_created=true``). server.clear_sam3 reads
# this list to build the restore query — keep the two in lockstep.
SAM3_PRESERVED_PROPS = (
    "center",
    "pos_x",
    "pos_y",
    "pos_z",
    "name",
    "class",
    "attr_type",
    "color_r",
    "color_g",
    "color_b",
    # Provenance: the dedup SET overwrites `source` with 'sam3_query'. Snapshot
    # it so clear RESTORES the base object's original provenance instead of
    # dropping the property (a base object ingested with source='hydra' used to
    # come back source-less). For SAM3-*created* nodes this changes nothing --
    # clear deletes them outright.
    "source",
)

#: The admission decision a query stamps on a base object it deduped onto
#: (``stamp_dedup_admission``) is part of that query's mutation, so clear must
#: put back whatever the object carried BEFORE the query -- not blank-remove it.
#: Blank-removal was harmless while the only pre-query state was "no admission",
#: but once an :class:`AdmissionPolicy` (Milestone 2) can ``reject_object`` a
#: prior, erasing that verdict would silently re-trust a refused object.
SAM3_PRESERVED_ADMISSION_PROPS = (
    _ADMISSION_STATUS,
    _ADMISSION_POLICY_VERSION,
    _ADMISSION_OBSERVATIONS,
    _ADMISSION_REASON,
)

#: Everything ``_preserve_originals_cypher`` snapshots and ``query_backends.reset``
#: restores. Keep the two in lockstep.
SAM3_SNAPSHOT_PROPS = SAM3_PRESERVED_PROPS + SAM3_PRESERVED_ADMISSION_PROPS


def _preserve_originals_cypher(label: str) -> str:
    """Cypher that snapshots a base object's original props ONCE (skips objects
    SAM3 created, and skips if already snapshotted by an earlier query)."""
    sets = ", ".join(f"o._sam3_orig_{p} = o.{p}" for p in SAM3_SNAPSHOT_PROPS)
    return (
        f"MATCH (o:{label} {{nodeSymbol: $sym}}) "
        "WHERE o.sam3_created IS NULL AND o._sam3_orig_saved IS NULL "
        f"SET o._sam3_orig_saved = true, {sets}"
    )


def stamp_dedup_admission(db, sym: str, label: str = None) -> None:
    """Stamp the admission of an object a SAM3 query DEDUPED onto (admission gate).

    The decision is read off ``o.sam3_created`` -- did the detector CREATE this
    node? -- NOT off ``o.source``, which this very dedup path overwrites with
    ``'sam3_query'`` and which a pre-gate database already carries on genuine
    prior-map objects. So a prior-map object re-detected by SAM3 is stamped
    ``trusted_prior`` and stays plannable, while a re-detection of an earlier
    SAM3-created candidate stays a candidate. ``coalesce`` means an existing
    decision is never downgraded (an already-admitted object survives
    re-detection admitted).

    Because ``sam3_created`` is immutable once written, this is order-independent
    with respect to the caller's ``SET o.source='sam3_query'``; it is still
    called first, next to the ``_preserve_originals_cypher`` snapshot, so the
    whole "what this query did to a pre-existing object" block stays together.

    Deliberately NOT an admission *writer* in the
    :mod:`~openset_query.evidence.admission` sense -- it grants no new
    trust, it only prevents the source overwrite from destroying trust the
    object already had.
    """
    from heracles import constants
    from openset_query.evidence.admission import (
        ADMISSION_POLICY_VERSION,
        ADMISSION_STATUS,
        PRIOR_MAP_POLICY_VERSION,
        dedup_admission_case,
    )

    db.execute(
        f"MATCH (o:{constants.OBJECTS} {{nodeSymbol: $sym}}) "
        f"SET o.{ADMISSION_STATUS} = {dedup_admission_case('o')}, "
        f"o.{ADMISSION_POLICY_VERSION} = "
        f"coalesce(o.{ADMISSION_POLICY_VERSION}, $prior_policy)",
        sym=sym,
        prior_policy=PRIOR_MAP_POLICY_VERSION,
    )


def _candidate_admission_props() -> dict:
    """Node-dict properties marking a NEWLY created detector object as a
    candidate: written, but not yet a planning fact (admission gate). Nothing auto-admits
    at write time; the warm server's configured admission policy
    (``evidence.admission_policy``) grades candidates right after the upsert."""
    from openset_query.evidence.admission import ADMISSION_STATUS, CANDIDATE

    return {ADMISSION_STATUS: CANDIDATE}


# Provenance markers persisted onto Observation nodes: which detector produced
# the view and through which verification mechanism it entered the graph.
_SAM3_DETECTOR = "sam3"
_SAM3_MECHANISM = "sam3_query"


def _store_obs_props(o: Observation) -> dict:
    """Image-store provenance of an observation (Neo4j contract: ``obs_id``,
    ``frame_id``, absolute ``rgb_path`` / ``depth_path`` / ``mask_path``, ``bbox_2d``,
    ``kept``). Keys with no value are absent so the heracles MERGE coalesces."""
    props: dict = {"bbox_2d": [int(v) for v in o.bbox_2d]}
    if o.rgb_path:
        props["rgb_path"] = o.rgb_path
    for k, v in (
        ("obs_id", o.obs_id),
        ("frame_id", o.frame_id),
        ("depth_path", o.depth_path),
        ("mask_path", o.mask_path),
        ("kept", o.kept),
    ):
        if v is not None:
            props[k] = v
    return props


def _obs_dict(sym: str, o: Observation) -> dict:
    return {
        "nodeSymbol": f"{sym}_{o.timestamp_ns}",
        "timestamp_ns": o.timestamp_ns,
        "mask_file": o.mask_file,
        **_store_obs_props(o),
        "bbox_2d_min_x": o.bbox_2d[0],
        "bbox_2d_min_y": o.bbox_2d[1],
        "bbox_2d_max_x": o.bbox_2d[2],
        "bbox_2d_max_y": o.bbox_2d[3],
        "score": o.score,
        "detector": _SAM3_DETECTOR,
        "mechanism": _SAM3_MECHANISM,
        # Raw evidence terms (raw scores AND the derived margin —
        # never only the margin). Keys the view has no value for are simply
        # absent; the heracles MERGE coalesces onto any earlier value.
        **_obs_evidence_props(o),
    }


def _obs_evidence_props(o: Observation) -> dict:
    """The persistable admission evidence terms of one observation (raw + derived).

    Kept in lockstep with ``evidence.admission_policy._EVIDENCE_QUERY`` and
    ``heracles.graph_interface.insert_observations_to_db`` (drift-guard in
    tests/test_admission_policy.py)."""
    props: dict = {}
    if o.depth_valid_px is not None:
        props["depth_valid_px"] = int(o.depth_valid_px)
    if o.view_xyz is not None:
        props["view_x"], props["view_y"], props["view_z"] = (
            float(v) for v in o.view_xyz
        )
    if o.target_prompt is not None:
        props["target_prompt"] = o.target_prompt
    if o.target_score is not None:
        props["target_score"] = float(o.target_score)
    if o.control_scores:
        control_max = max(o.control_scores.values())
        props["control_scores_json"] = json.dumps(o.control_scores, sort_keys=True)
        props["control_max_score"] = float(control_max)
        if o.target_score is not None:
            props["margin"] = float(o.target_score) - float(control_max)
    return props


def _depicts_pair(sym: str, o: Observation):
    """(TrajectoryFrame symbol, object symbol) for a DEPICTS edge, or None.

    The frame symbol is content-derived exactly like TrajectoryFrameIndex.ingest
    does, so the MATCH-based edge merge silently no-ops when the frame was never
    ingested (e.g. maps set up with --skip-frame-index)."""
    if not o.rgb_path:
        return None
    from openset_query.perception.frame_index import _frame_node_symbol

    return (_frame_node_symbol(o.timestamp_ns, o.rgb_path), sym)


def _link_observation_provenance(db, depicts_pairs) -> None:
    """Best-effort: OBSERVED_IN (obs -> source frame, timestamp join) and
    DEPICTS (TrajectoryFrame -> Object) edges after an upsert batch."""
    from heracles.graph_interface import (
        insert_frame_edges,
        link_observations_to_frames,
    )

    try:
        link_observations_to_frames(db)
    except Exception:
        logger.exception("OBSERVED_IN provenance linking failed (non-fatal)")
    try:
        pairs = [p for p in depicts_pairs if p is not None]
        if pairs:
            insert_frame_edges(db, [], pairs)
    except Exception:
        logger.exception("DEPICTS provenance linking failed (non-fatal)")


def upsert_objects(
    discovered: List[DiscoveredObject],
    G,
    db=None,
    config: Optional[Sam3FrontendConfig] = None,
    graph_api=None,
    image_store=None,
    result: Optional[dict] = None,
) -> List[str]:
    """Write discovered objects to the graph, deduping against existing ones.

    DB-first: when ``db`` is provided every mutation is written to Neo4j before
    the in-memory ``G`` is touched (Neo4j is the source of truth). Returns the
    nodeSymbols that were created or updated.

    ``image_store`` (store directory, or a list of them: the first is written): when
    given, every observation is recorded there as a kept SAM3 observation of its object
    and the Neo4j ``Observation`` nodes carry the store paths.

    New object indices are claimed in Neo4j (:func:`claim_object_index`: one
    serialized statement per object, retried on a symbol that already exists), so
    concurrent writers never land on the same ``O<n>``. ``result`` (a dict), when
    given, receives ``{"created": [syms], "merged": [syms]}``.
    """
    config = config or Sam3FrontendConfig()
    existing = _existing_objects(db, G)
    next_idx = _next_object_index(db, G)
    touched: List[str] = []

    new_dicts = []
    observations = []
    obs_edges = []

    writer = open_sam3_writer(image_store)
    alloc = _IndexAllocator(db)
    if result is None:
        result = {}
    result.setdefault("created", [])
    result.setdefault("merged", [])
    try:
        touched = _upsert_objects_write(
            discovered,
            G,
            db,
            config,
            writer,
            existing,
            next_idx,
            touched,
            new_dicts,
            observations,
            obs_edges,
            alloc=alloc,
            result=result,
        )
        alloc.finish()
    except Exception as exc:
        alloc.abort()
        # touched is a local list this call alone appends to as it goes --
        # by the time any DB write below can raise, every nodeSymbol THIS
        # call is responsible for (a dedup match already SET, or a new
        # object about to be/just batch-inserted) is already in `touched`
        # (the per-item loop below finishes appending to it before the
        # batch insert_nodes_to_db/insert_observations_to_db/insert_edges
        # calls run). Attach that exact, purely LOCAL list to the exception
        # so a caller (e.g. CatalogOracleBackend.query's except block) can
        # salvage it without any DB round-trip/diff -- a global before/after
        # existing-objects diff would also catch a concurrent writer's own
        # new node created in that same window and misattribute it into
        # THIS call's touched set (the exact cross-writer corruption class
        # this whole fix targets).
        exc.partial_touched = list(touched)
        raise
    finally:
        if writer is not None:
            writer.close()
    return touched


def _upsert_objects_write(
    discovered,
    G,
    db,
    config,
    writer,
    existing,
    next_idx,
    touched,
    new_dicts,
    observations,
    obs_edges,
    alloc=None,
    result=None,
) -> List[str]:
    import spark_dsg
    from heracles import constants
    from heracles.graph_interface import (
        insert_nodes_to_db,
        insert_observations_to_db,
        insert_edges,
        node_to_dict,
    )

    depicts_pairs = []
    for disc in discovered:
        centroid = disc.obj.centroid
        # Dedup: closest existing object of the same label within threshold.
        match_sym = None
        best = config.dedup_centroid_dist_m
        for sym, pos, name in existing:
            d = float(np.linalg.norm(pos - centroid))
            if d <= best and (name is None or name == disc.obj.label):
                best = d
                match_sym = sym

        if match_sym is not None:
            sym = match_sym
            record_observations(writer, sym, disc.observations)
            if db is not None:
                col = disc.obj.color
                # Snapshot the base object's originals once so /api/sam3/clear can
                # restore it (non-destructive to the base map).
                db.execute(_preserve_originals_cypher(constants.OBJECTS), sym=sym)
                # Admission gate: a re-observed object keeps the trust it already had;
                # only a detector-CREATED node (sam3_created) is a candidate.
                stamp_dedup_admission(db, sym)
                db.execute(
                    f"MATCH (o:{constants.OBJECTS} {{nodeSymbol: $sym}}) "
                    "SET o.center = point({x:$x, y:$y, z:$z}), "
                    "o.pos_x=$x, o.pos_y=$y, o.pos_z=$z, o.name=$name, o.class=$name, "
                    "o.source='sam3_query', o.attr_type='KhronosObjectAttributes', "
                    "o.color_r=$cr, o.color_g=$cg, o.color_b=$cb",
                    sym=sym,
                    x=float(centroid[0]),
                    y=float(centroid[1]),
                    z=float(centroid[2]),
                    name=disc.obj.label,
                    cr=int(col[0]),
                    cg=int(col[1]),
                    cb=int(col[2]),
                )
            logger.info("Updated existing object %s (%s)", sym, disc.obj.label)
            if result is not None:
                result["merged"].append(sym)
        else:
            idx = alloc.claim(next_idx) if alloc is not None else next_idx
            ns = spark_dsg.NodeSymbol("O", idx)
            sym = ns.str(True)
            next_idx = idx + 1
            if result is not None:
                result["created"].append(sym)
            record_observations(writer, sym, disc.observations, new_object=True)
            attrs = _make_object_attrs(disc)
            d = node_to_dict(_DummyNode(ns, attrs))
            d["class"] = disc.obj.label
            d["name"] = disc.obj.label
            d["source"] = "sam3_query"  # provenance marker for viz color-coding
            d["sam3_created"] = True  # SAM3 created this node → safe to delete on clear
            d.update(
                _candidate_admission_props()
            )  # Admission gate: written, not yet a fact
            new_dicts.append(d)
            existing.append((sym, centroid, disc.obj.label))
            # Mirror in memory (skipped for DB-only build-time service).
            if G is not None:
                try:
                    G.add_node(spark_dsg.DsgLayers.OBJECTS, ns.value, attrs)
                except Exception as e:
                    logger.warning("In-memory add_node failed for %s: %s", sym, e)
            logger.info("Created new object %s (%s)", sym, disc.obj.label)

        touched.append(sym)
        for o in disc.observations:
            d = _obs_dict(sym, o)
            observations.append(d)
            obs_edges.append({"from": sym, "to": d["nodeSymbol"]})
            depicts_pairs.append(_depicts_pair(sym, o))

    if db is not None:
        if new_dicts:
            insert_nodes_to_db(db, constants.OBJECTS, new_dicts)
        if observations:
            insert_observations_to_db(db, observations)
            insert_edges(
                db,
                constants.HAS_OBSERVATION,
                constants.OBJECTS,
                constants.OBSERVATIONS,
                obs_edges,
            )
            _link_observation_provenance(db, depicts_pairs)
    else:
        logger.warning(
            "No db provided: objects added only in-memory. Neo4j is the source of "
            "truth; pass a Neo4jWrapper to persist."
        )

    return touched


_CLAIM_INDEX_CYPHER = (
    # One statement: the write lock on the allocator node serializes concurrent
    # claims; the index is above every existing O<n> and every earlier claim.
    "MERGE (c:_IdAlloc {key: 'Object'}) "
    "SET c._lock = timestamp() "
    "WITH c "
    "OPTIONAL MATCH (o:Object) WHERE o.nodeSymbol =~ 'O[0-9]+' "
    "WITH c, coalesce(max(toInteger(substring(o.nodeSymbol, 1))), -1) + 1 AS scanned "
    "WITH c, reduce(m = $floor, v IN [scanned, coalesce(c.next, 0)] | "
    "CASE WHEN v > m THEN v ELSE m END) AS idx "
    "SET c.next = idx + 1 "
    "WITH c, idx "
    "MERGE (n:Object {nodeSymbol: 'O' + toString(idx)}) "
    "ON CREATE SET n._alloc_token = $token "
    "RETURN idx, n._alloc_token = $token AS mine"
)


def claim_object_index(db, floor: int, token: str, max_tries: int = 64) -> int:
    """Claim a fresh object index ``>= floor`` in Neo4j and return it.

    The claim creates a bare ``Object {nodeSymbol, _alloc_token}`` placeholder that
    the following ``insert_nodes_to_db`` MERGE fills in. When the symbol already
    exists (a writer that does not use the allocator got there first) the next index
    is tried. A database stub that returns no rows falls back to ``floor``.
    """
    for _ in range(max_tries):
        records, _, _ = db.execute(_CLAIM_INDEX_CYPHER, floor=int(floor), token=token)
        if not records:
            return int(floor)
        idx = int(records[0]["idx"])
        if records[0]["mine"]:
            return idx
        floor = idx + 1
    raise RuntimeError(f"could not claim an object index after {max_tries} tries")


class _IndexAllocator:
    """Per-upsert index claims; placeholders are cleaned up when the write fails."""

    def __init__(self, db):
        import uuid

        self.db = db
        self.token = uuid.uuid4().hex
        self.claimed = 0

    def claim(self, floor: int) -> int:
        if self.db is None:
            return floor
        self.claimed += 1
        return claim_object_index(self.db, floor, self.token)

    def finish(self) -> None:
        if self.db is not None and self.claimed:
            self.db.execute(
                "MATCH (o:Object {_alloc_token: $t}) REMOVE o._alloc_token",
                t=self.token,
            )

    def abort(self) -> None:
        if self.db is None or not self.claimed:
            return
        try:  # placeholders that never got their properties
            self.db.execute(
                "MATCH (o:Object {_alloc_token: $t}) WHERE o.class IS NULL "
                "DETACH DELETE o",
                t=self.token,
            )
            self.finish()
        except Exception:
            logger.exception("could not clean up claimed object placeholders")


class _DummyNode:
    """Minimal node shim so node_to_dict() can read id + attributes."""

    def __init__(self, node_symbol, attributes):
        self.id = node_symbol
        self.attributes = attributes


# ---------------------------------------------------------------------------
# Step 4: openset-framework path — TrajectoryFrameIndex selection + GraphWriter
# writes. This is the reconciled path; the EmbeddingFilter+upsert_objects path
# above is kept for the warm query_server until this path is validated in prod.
# ---------------------------------------------------------------------------


def frame_index_select(
    frame_index,
    prompt: str,
    k: int = 20,
    center_xyz: Optional[Sequence[float]] = None,
    radius_m: Optional[float] = None,
    store=None,
    pose_lookup=None,
) -> List[tuple]:
    """Rank keyframes via openset's ``TrajectoryFrameIndex.query_by_text`` and
    load each as a reprojection-ready ``Keyframe`` from the image ``store``.

    Returns ``[(Keyframe, score)]`` descending by text-image cosine similarity —
    the same shape ``detect_frustum_aware`` expects. Frame rows are resolved from the
    TrajectoryFrame ``path`` (the store's absolute RGB path); calibrations are cached
    per sensor.
    """
    if store is None:
        raise ValueError("frame_index_select needs the image store (store=...)")
    rows = frame_index.query_by_text(
        prompt, k=k, radius_m=radius_m, center_xyz=center_xyz
    )
    out: List[tuple] = []
    by_path = {str(f.rgb_path): f for f in store.frames() if f.rgb_path is not None}
    calib_cache: dict = {}
    for r in rows:
        path = r.get("path")
        frame = by_path.get(str(path)) if path else None
        if frame is None:
            continue
        kf = ks.keyframe_from_frame(store, frame, pose_lookup, calib_cache)
        if kf is None:
            continue
        # carry the frame's place_id so upsert can link CONTAINS provenance
        kf.place_id = r.get("place_id")
        out.append((kf, float(r.get("score", 0.0))))
    return out


async def upsert_via_graph_writer(
    discovered: List[DiscoveredObject],
    writer,
    config: Optional[Sam3FrontendConfig] = None,
    image_store=None,
) -> List[str]:
    sw = open_sam3_writer(image_store)
    try:
        return await _upsert_via_graph_writer(discovered, writer, config, sw)
    finally:
        if sw is not None:
            sw.close()


async def _upsert_via_graph_writer(
    discovered: List[DiscoveredObject],
    writer,
    config: Optional[Sam3FrontendConfig],
    sw,
) -> List[str]:
    """Persist discovered objects through openset's async ``GraphWriter``.

    Mirrors :func:`upsert_objects` (dedup by label+proximity, per-object primary
    color, image-store observations, ``HAS_OBSERVATION`` provenance) but routes new-object
    creation through ``GraphWriter.add_object`` so openset's in-memory mirror and
    fanout/versioning fire. Color / attr_type and observations —
    which ``add_object`` does not model — are applied as DB follow-ups on the same
    connection (``writer._db``). DB remains the source of truth.
    """
    from heracles import constants
    from heracles.graph_interface import insert_observations_to_db, insert_edges

    config = config or Sam3FrontendConfig()
    db = writer._db
    existing = _existing_objects(db, None)
    touched: List[str] = []
    observations = []
    obs_edges = []
    depicts_pairs = []

    for disc in discovered:
        centroid = disc.obj.centroid
        col = disc.obj.color
        match_sym = None
        best = config.dedup_centroid_dist_m
        for sym, pos, name in existing:
            d = float(np.linalg.norm(pos - centroid))
            if d <= best and (name is None or name == disc.obj.label):
                best = d
                match_sym = sym

        dims = disc.obj.bbox_max - disc.obj.bbox_min
        bcen = (disc.obj.bbox_min + disc.obj.bbox_max) / 2.0

        if match_sym is not None:
            sym = match_sym
            # Snapshot the base object's originals once so /api/sam3/clear can
            # restore it (non-destructive to the base map).
            db.execute(_preserve_originals_cypher(constants.OBJECTS), sym=sym)
            # Admission gate: a re-observed object keeps the trust it already had; only a
            # detector-CREATED node (sam3_created) is a candidate.
            stamp_dedup_admission(db, sym)
            db.execute(
                f"MATCH (o:{constants.OBJECTS} {{nodeSymbol: $sym}}) "
                "SET o.center = point({x:$x, y:$y, z:$z}), "
                "o.pos_x=$x, o.pos_y=$y, o.pos_z=$z, o.name=$name, o.class=$name, "
                "o.source='sam3_query'",
                sym=sym,
                x=float(centroid[0]),
                y=float(centroid[1]),
                z=float(centroid[2]),
                name=disc.obj.label,
            )
            logger.info("Updated existing object %s (%s)", sym, disc.obj.label)
        else:
            place_id = getattr(_obs_kf(disc), "place_id", None)
            # bbox=None: we write the canonical bbox_center/bbox_dim (the schema
            # db_to_spark_dsg reads) in the follow-up SET, not add_object's bbox_cx/dx.
            sym = await writer.add_object(
                class_label=disc.obj.label,
                position=[float(centroid[0]), float(centroid[1]), float(centroid[2])],
                name=disc.obj.label,
                bbox=None,
                source="sam3_query",
                confidence=float(disc.obj.score),
                parent_place=place_id,
            )
            existing.append((sym, centroid, disc.obj.label))
            # Admission gate: a fresh detector object is a candidate, never a planning fact.
            from openset_query.evidence.admission import ADMISSION_STATUS, CANDIDATE

            db.execute(
                f"MATCH (o:{constants.OBJECTS} {{nodeSymbol: $sym}}) "
                f"SET o.sam3_created = true, o.{ADMISSION_STATUS} = $adm",
                sym=sym,
                adm=CANDIDATE,
            )
            logger.info(
                "Created new object %s (%s) via GraphWriter", sym, disc.obj.label
            )

        # Follow-up: color + attr_type + the canonical bbox the viewer
        # reads (bbox_center/bbox_dim points). add_object models none of these.
        record_observations(sw, sym, disc.observations, new_object=match_sym is None)
        db.execute(
            f"MATCH (o:{constants.OBJECTS} {{nodeSymbol: $sym}}) "
            "SET o.color_r=$cr, o.color_g=$cg, o.color_b=$cb, "
            "o.attr_type='KhronosObjectAttributes', "
            "o.bbox_center=point({x:$bx, y:$by, z:$bz}), "
            "o.bbox_dim=point({x:$dx, y:$dy, z:$dz})",
            sym=sym,
            cr=int(col[0]),
            cg=int(col[1]),
            cb=int(col[2]),
            bx=float(bcen[0]),
            by=float(bcen[1]),
            bz=float(bcen[2]),
            dx=float(dims[0]),
            dy=float(dims[1]),
            dz=float(dims[2]),
        )

        touched.append(sym)
        for o in disc.observations:
            d = _obs_dict(sym, o)
            observations.append(d)
            obs_edges.append({"from": sym, "to": d["nodeSymbol"]})
            depicts_pairs.append(_depicts_pair(sym, o))

    if observations:
        insert_observations_to_db(db, observations)
        insert_edges(
            db,
            constants.HAS_OBSERVATION,
            constants.OBJECTS,
            constants.OBSERVATIONS,
            obs_edges,
        )
        _link_observation_provenance(db, depicts_pairs)
    return touched


def _obs_kf(disc: DiscoveredObject):
    """Best-effort handle to a contributing keyframe (carries place_id)."""
    return getattr(disc, "_src_kf", None)


async def discover_and_upsert_indexed(
    frame_index,
    segmenter,
    prompt: str,
    writer,
    config: Optional[Sam3FrontendConfig] = None,
    k: int = 20,
    mode: str = "frustum",
    max_frames: int = 12,
    patience: int = 1,
    center_xyz: Optional[Sequence[float]] = None,
    radius_m: Optional[float] = None,
    mask_output_dir: Optional[str] = None,
    image_store=None,
    pose_lookup=None,
) -> List[str]:
    """Reconciled end-to-end discovery: ``TrajectoryFrameIndex`` selects frames,
    ``segmenter`` (a ``Sam3Detector``/``Sam3Segmenter``) detects, reprojection +
    fusion build 3D objects, and ``GraphWriter`` persists them. Returns touched
    nodeSymbols.
    """
    config = config or Sam3FrontendConfig()
    from hydra_python.image_store import ImageStore

    path = image_store[0] if isinstance(image_store, (list, tuple)) else image_store
    store = ImageStore.open(
        *(image_store if isinstance(image_store, (list, tuple)) else [path])
    )
    try:
        ranked = frame_index_select(
            frame_index,
            prompt,
            k=k,
            center_xyz=center_xyz,
            radius_m=radius_m,
            store=store,
            pose_lookup=pose_lookup,
        )
    finally:
        store.close()
    logger.info("frame-index selected %d keyframes for %r", len(ranked), prompt)
    if not ranked:
        return []

    if mode == "batched":
        discovered = detect_batched(
            [kf for kf, _ in ranked],
            segmenter,
            [prompt],
            config,
            mask_output_dir=mask_output_dir,
        )
    elif mode == "topk":
        discovered = detect_in_keyframes(
            [kf for kf, _ in ranked],
            segmenter,
            [prompt],
            config,
            mask_output_dir=mask_output_dir,
        )
    else:  # frustum-aware early stop (default)
        discovered = detect_frustum_aware(
            ranked,
            segmenter,
            [prompt],
            config,
            max_frames=max_frames,
            patience=patience,
            mask_output_dir=mask_output_dir,
        )
    # tag each discovered object with a contributing keyframe for place provenance
    by_ts = {kf.timestamp_ns: kf for kf, _ in ranked}
    for disc in discovered:
        if disc.observations:
            disc._src_kf = by_ts.get(disc.observations[0].timestamp_ns)
    logger.info("Fused into %d candidate objects", len(discovered))
    return await upsert_via_graph_writer(
        discovered, writer, config=config, image_store=image_store
    )


def discover_and_upsert(
    G,
    segmenter,
    prompts: Sequence[str],
    config: Optional[Sam3FrontendConfig] = None,
    db=None,
    near_position: Optional[np.ndarray] = None,
    radius_m: Optional[float] = None,
    mask_output_dir: Optional[str] = None,
    max_keyframes: Optional[int] = None,
    image_store=None,
    embed_filter: bool = False,
    embed_model: str = "siglip2",
    embed_top_k: int = 30,
    embed_threshold: Optional[float] = None,
) -> List[str]:
    """End-to-end query-time discovery + upsert. Returns touched nodeSymbols.

    If ``embed_filter`` is set, a CLIP/SigLIP2 model ranks all keyframes by
    text-image cosine similarity to ``prompts`` and only the top-``embed_top_k``
    are passed to SAM3 (much faster + more precise than running SAM3 on every
    frame). Otherwise ``max_keyframes`` evenly samples the frames.
    """
    config = config or Sam3FrontendConfig()
    # Neo4j is the source of truth: when a db is provided, get keyframes by querying
    # Agent rgb_path/depth_path + CameraCalib from the DB (db_to_spark_dsg does not
    # reconstruct AGENTS). Without a db, read the image store and pose from the DSG.
    store_dirs = (
        [image_store]
        if isinstance(image_store, (str, os.PathLike))
        else list(image_store or [])
    )
    if db is not None:
        keyframes = ks.load_keyframes_from_db(db, near_position, radius_m)
    else:
        if not store_dirs:
            raise ValueError("discover_and_upsert without a db needs image_store=...")
        from hydra_python.image_store import ImageStore

        store = ImageStore.open(*store_dirs)
        try:
            keyframes = ks.load_keyframes_from_graph(G, store, near_position, radius_m)
        finally:
            store.close()

    if embed_filter and prompts and keyframes:
        from openset_query.embedding_filter import EmbeddingFilter

        cache = ks.embed_cache_path(store_dirs[0], embed_model) if store_dirs else None
        ef = EmbeddingFilter(model=embed_model, device=config.device, cache_path=cache)
        keyframes = ef.select(
            keyframes, list(prompts), top_k=embed_top_k, threshold=embed_threshold
        )
        logger.info(
            "embed-filter (%s) selected %d keyframes", embed_model, len(keyframes)
        )
    elif max_keyframes and len(keyframes) > max_keyframes:
        import numpy as _np

        idx = _np.linspace(0, len(keyframes) - 1, max_keyframes).astype(int)
        keyframes = [keyframes[i] for i in idx]
        logger.info("Sampled %d of the available keyframes", len(keyframes))
    logger.info("Loaded %d keyframes for prompts %s", len(keyframes), list(prompts))
    discovered = detect_in_keyframes(
        keyframes, segmenter, prompts, config, mask_output_dir=mask_output_dir
    )
    logger.info("Fused into %d candidate objects", len(discovered))
    return upsert_objects(discovered, G, db=db, config=config, image_store=image_store)
