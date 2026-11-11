"""SAM3 object-discovery write-back into a Hydra image store.

Each SAM3-discovered scene graph node gets one track in the store's external
track-id range (``EXTERNAL_TRACK_BASE`` and up), which Hydra's indexer never
rebinds or deletes. Only the public ``hydra_python.image_store`` writer API is
used (``batch``, ``bind_track``, ``add_observation``).
"""

from __future__ import annotations

from hydra_python.image_store import EXTERNAL_TRACK_BASE

#: SAM3 tracks: ``SAM3_TRACK_BASE | (node_id & SAM3_NODE_MASK)``, one per node
SAM3_TRACK_BASE = EXTERNAL_TRACK_BASE
SAM3_NODE_MASK = (1 << 40) - 1


def sam3_track_id(node_id: int) -> int:
    """Track id of the SAM3 observations bound to ``node_id`` (one track per node)."""
    return SAM3_TRACK_BASE | (int(node_id) & SAM3_NODE_MASK)


def is_sam3_track(track_id: int) -> bool:
    return SAM3_TRACK_BASE <= int(track_id) <= SAM3_TRACK_BASE | SAM3_NODE_MASK


def add_sam3_observation(writer, node_id: int, frame_id: int, **kwargs) -> int:
    """Observation of a SAM3-discovered object in ``frame_id``; binds the node's SAM3
    track to ``node_id``. ``kwargs`` are those of ``ImageStoreWriter.add_observation``
    (``mask``, ``mask_rle``, ``mask_path``, ``bbox2d``, ``score``, ``kept``, ...)."""
    track = sam3_track_id(node_id)
    with writer.batch():
        writer.bind_track(track, node_id)
        return writer.add_observation(track, frame_id, **kwargs)
