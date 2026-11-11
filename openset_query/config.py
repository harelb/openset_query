"""Configuration for the SAM3 frontend.

Centralizes the tunable knobs for open-vocabulary object discovery and region
labeling so the CLI, query-time, and build-time entry points share one source of
truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


# A small default object vocabulary used for "fixed-list" sweeps during mapping
# or when no explicit query is given. Override via config / CLI for a deployment.
DEFAULT_OBJECT_CLASSES: List[str] = [
    "chair",
    "table",
    "door",
    "fire extinguisher",
    "backpack",
    "laptop",
    "monitor",
    "trash can",
    "box",
    "person",
]

# Region/scene prompts used to label places/rooms.
DEFAULT_REGION_CLASSES: List[str] = [
    "office",
    "hallway",
    "kitchen",
    "conference room",
    "stairwell",
    "lobby",
    "bathroom",
    "storage room",
]


@dataclass
class Sam3FrontendConfig:
    # --- Model / inference ---
    device: str = "cuda"
    sam3_version: str = "sam3.1"  # "sam3" or "sam3.1"
    conf_threshold: float = 0.5
    # torch.compile the SAM3 image model (compiled at load, warmed at startup).
    # Big per-query speedup once warm; adds a one-time compile cost at startup.
    compile: bool = False

    # --- Vocabulary (hybrid: fixed list + arbitrary query strings) ---
    object_classes: List[str] = field(
        default_factory=lambda: list(DEFAULT_OBJECT_CLASSES)
    )
    region_classes: List[str] = field(
        default_factory=lambda: list(DEFAULT_REGION_CLASSES)
    )

    # --- Reprojection ---
    # Reject masks whose number of valid (finite, in-range) depth pixels is below
    # this, since their 3D centroid would be unreliable.
    min_mask_depth_pixels: int = 50
    # Depth values outside [min_depth_m, max_depth_m] are treated as invalid.
    min_depth_m: float = 0.2
    max_depth_m: float = 15.0
    # Drop the farthest/nearest depth percentiles before computing the centroid to
    # be robust to mask bleed onto background.
    depth_percentile_lo: float = 5.0
    depth_percentile_hi: float = 95.0

    # --- Multi-view fusion (same prompt, across keyframes) ---
    # "cluster": 3D clustering of all a query's detections (association.py: spatial
    # hash + union-find on AABB IoU / containment / extent-scaled centroid distance,
    # label-gated, optional appearance gate); "centroid": the legacy greedy fusion.
    association: str = "cluster"
    # Caps the centroid rule of the clustering; the sole rule of "centroid".
    fuse_centroid_dist_m: float = 0.75
    # Overrides of association.AssociationParams (None = its default).
    assoc_cell_m: Optional[float] = None
    assoc_iou_min: Optional[float] = None
    assoc_contain_min: Optional[float] = None
    assoc_max_volume_ratio: Optional[float] = None
    assoc_k_extent: Optional[float] = None
    assoc_min_dist_m: Optional[float] = None
    assoc_max_dist_m: Optional[float] = None
    assoc_emb_min: Optional[float] = None
    # Gate the final clustering by the SigLIP2 cosine of the detection crops
    # (one extra embedding forward per detection; off by default).
    cluster_crop_embeddings: bool = False

    # --- Dedup vs existing objects in the graph ---
    dedup_centroid_dist_m: float = 0.75

    # --- Region labeling ---
    # Max distance from a keyframe's camera position to a place node for the
    # keyframe to be considered an observation of that place.
    region_assoc_dist_m: float = 5.0
