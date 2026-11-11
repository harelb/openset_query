"""Relative-margin evidence terms for detection admission.

The contract this module serves ("source-admissibility gate"):

    a query-time candidate may satisfy a symbolic literal only if its positive
    evidence separates from archive-specific negative controls by a held-out
    calibrated margin and passes geometric/multi-view checks.

This module holds the *terms* of that contract as data, plus the pure math
over them. It deliberately knows nothing about SigLIP2, SAM3, Neo4j, or
thresholds: producers (the warm query server's crop scorer) fill in raw
scores; the :mod:`~openset_query.evidence.admission_policy` policies own
the thresholds and the verdicts. Keeping the raw terms first-class is deliberate
("Persist raw score terms and the admission result. Do not persist only the
final margin"): a margin alone cannot be re-audited when the control set or
the calibration changes.

**What a margin is here** — per-observation, target-vs-control:

    margin = cos(crop, target_prompt) - max_c cos(crop, control_c)

i.e. how much better the detection's image crop matches the queried concept
than it matches any negative-control concept. It is *never* a comparison
between two frames of the same object (a top-1-vs-top-2-frame "margin" would
punish objects for being visible in many frames — pinned by
tests/test_margins.py).

Import contract: stdlib only at module scope (same as the rest of
``evidence/``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ObservationEvidence:
    """The raw, per-observation evidence terms an admission policy consumes.

    One instance per (detection view) — i.e. per ``:Observation`` node. Every
    field a producer cannot supply is ``None``; nothing here invents values
    (the ``evidence/types.py`` rule). ``control_scores`` maps each negative
    control prompt to the crop's cosine against it — the raw terms, kept so a
    re-calibration can recompute margins without re-embedding.
    """

    obs_id: str
    timestamp_ns: Optional[int] = None
    #: SAM3 detector confidence for THIS view (``Observation.score``).
    sam3_score: Optional[float] = None
    #: SigLIP2 cosine of the detection crop against the queried prompt.
    target_score: Optional[float] = None
    #: SigLIP2 cosine of the crop against each negative-control prompt.
    control_scores: Dict[str, float] = field(default_factory=dict)
    #: Depth-valid mask pixels that survived reprojection (mask/depth validity).
    depth_valid_px: Optional[int] = None
    #: This view's OWN reprojected centroid (world frame), pre-fusion — the
    #: geometric-consistency input (do the views agree on where the object is?).
    view_xyz: Optional[Tuple[float, float, float]] = None
    #: The prompt the target score was computed against (margins from
    #: different prompts are not comparable; graders must not mix them).
    target_prompt: Optional[str] = None

    @property
    def control_max(self) -> Optional[float]:
        """Best negative-control cosine (``None`` when no controls scored)."""
        return max(self.control_scores.values()) if self.control_scores else None

    @property
    def margin(self) -> Optional[float]:
        """Target-vs-control relative margin; ``None`` when either side is
        missing (an unscored view can never pass or fail a margin gate)."""
        c = self.control_max
        if self.target_score is None or c is None:
            return None
        return float(self.target_score) - float(c)

    def to_json(self) -> dict:
        return {
            "obs_id": self.obs_id,
            "timestamp_ns": self.timestamp_ns,
            "sam3_score": self.sam3_score,
            "target_score": self.target_score,
            "control_scores": dict(self.control_scores),
            "depth_valid_px": self.depth_valid_px,
            "view_xyz": list(self.view_xyz) if self.view_xyz is not None else None,
            "target_prompt": self.target_prompt,
            # Derived, for readers that only see the JSON (raw terms above
            # stay the source of truth).
            "control_max": self.control_max,
            "margin": self.margin,
        }

    @classmethod
    def from_json(cls, d: dict) -> "ObservationEvidence":
        xyz = d.get("view_xyz")
        return cls(
            obs_id=str(d["obs_id"]),
            timestamp_ns=d.get("timestamp_ns"),
            sam3_score=d.get("sam3_score"),
            target_score=d.get("target_score"),
            control_scores=dict(d.get("control_scores") or {}),
            depth_valid_px=d.get("depth_valid_px"),
            view_xyz=tuple(float(v) for v in xyz) if xyz is not None else None,
            target_prompt=d.get("target_prompt"),
        )


def view_spread_m(evidences: Iterable[ObservationEvidence]) -> Optional[float]:
    """Geometric consistency: max distance of any view's own centroid from the
    mean of all view centroids (metres).

    ``None`` when fewer than two views carry a centroid — a single view has no
    cross-view consistency to measure, and a policy must treat that as
    "unmeasured", never as "consistent".
    """
    pts = [e.view_xyz for e in evidences if e.view_xyz is not None]
    if len(pts) < 2:
        return None
    n = len(pts)
    mean = tuple(sum(p[i] for p in pts) / n for i in range(3))
    return max(math.dist(p, mean) for p in pts)


def distinct_view_count(evidences: Iterable[ObservationEvidence]) -> int:
    """Multi-view support: number of distinct capture timestamps.

    Two detections in the SAME frame (co-timestamped) are one view — they share
    every failure mode of that vantage, so they must not double-count."""
    ts = {e.timestamp_ns for e in evidences if e.timestamp_ns is not None}
    if ts:
        return len(ts)
    # Synthetic/test evidence without timestamps: fall back to record count.
    return sum(1 for _ in evidences)


def best_margin(evidences: Sequence[ObservationEvidence]) -> Optional[float]:
    margins = [e.margin for e in evidences if e.margin is not None]
    return max(margins) if margins else None
