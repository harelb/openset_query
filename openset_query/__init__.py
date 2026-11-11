"""SAM3 / SAM3.1 open-vocabulary frontend for Hydra scene graphs.

Runs SAM3 over stored RGB-D keyframes (query time) or a live keyframe stream
(build time), reprojects masks into 3D, and writes discovered objects / region
labels back into the scene graph (Neo4j is the source of truth).

Heavy imports (torch / sam3) are intentionally kept inside the modules that need
them so importing this package is cheap.
"""

from openset_query.config import Sam3FrontendConfig  # noqa: F401
from openset_query.reprojection import (  # noqa: F401
    CameraCalib,
    Object3D,
    fuse_detections,
    mask_to_points,
    points_to_object,
)
