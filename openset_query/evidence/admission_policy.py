"""The admission policies that grade detector candidates.

The admission gate (``evidence/admission.py``) made "the detector wrote it" distinct
from "the system will plan on it" and left one seam open: *nothing auto-admits*.
This module fills that seam with the AdmissionPolicy interface and TWO
implementations, deliberately shipped side by side so they can be compared:

``RelativeMarginPolicy`` (kind ``"margin"``)
    The source-admissibility contract: a candidate is admitted only when enough
    distinct views separate from archive negative controls by a held-out
    calibrated SigLIP2 margin AND pass mask/depth validity AND agree
    geometrically. It can also *reject*: when every valid view fails the
    control separation decisively, the candidate is a phantom.

``GeometricPolicy`` (kind ``"geometric"``)
    Plain "segmentation and geometric validation": mask/depth validity + multi-view geometric
    consistency, no embedding margins. It never rejects — geometric
    agreement cannot prove a consistent phantom wrong, so failed candidates
    stay ``candidate`` (UNKNOWN), and the margin-vs-geometric delta on
    phantom-prone queries is what the comparison measures.

Both policies read their thresholds from ONE calibration artifact (a JSON file
produced on held-out data by ``openset_query.eval.calibrate_admission``), so
the two policies differ only in the margin term, never in tuning. Decisions are
stamped with ``"<artifact version>:<kind>"`` so every admission in a trace
names the exact gate that made it.

Verdicts are three-way — ``admitted`` / ``rejected`` / ``unknown`` — and
``unknown`` writes nothing: the candidate stays a candidate, acquirable by a
later tier. Only :func:`grade_object` touches the database, and it goes
through the sanctioned writers in :mod:`~openset_query.evidence.admission`.

Import contract: stdlib only at module scope; ``db`` handles are passed in.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Protocol, Sequence, Tuple, runtime_checkable

from .admission import (
    ADMITTED,
    CANDIDATE,
    REJECTED,
    admission_status,
    admit_object,
    reject_object,
)
from .margins import (
    ObservationEvidence,
    distinct_view_count,
    view_spread_m,
)

#: The non-writing verdict: the policy could not settle the candidate either
#: way. NOT an admission status (nothing is written for it).
UNKNOWN = "unknown"

#: Config values for "which admission policy runs" (the policy switch).
POLICY_KINDS = ("margin", "geometric", "off")


@dataclass(frozen=True)
class AdmissionDecision:
    """One policy verdict over one candidate object's evidence.

    ``belief`` is a *measured* number — the best SAM3 detector confidence among
    the views that carried the admission — never an invented calibration of the
    margin. ``observation_ids`` are the views the policy actually weighed
    (persisted by ``admit_object`` for auditability).
    """

    status: str  # ADMITTED | REJECTED | UNKNOWN
    reason: str
    policy_version: str
    belief: Optional[float] = None
    observation_ids: Tuple[str, ...] = ()

    def to_json(self) -> dict:
        return {
            "status": self.status,
            "reason": self.reason,
            "policy_version": self.policy_version,
            "belief": self.belief,
            "observation_ids": list(self.observation_ids),
        }


# ---------------------------------------------------------------------------
# Calibration artifact: thresholds come from held-out data, never
# from constants in this file.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdmissionCalibration:
    """The negative-control configuration, loaded from a calibration artifact.

    ``version`` names the calibration (it prefixes every policy version this
    process stamps); ``calibrated_on`` is free-form provenance (archives,
    classes, date, script commit) recorded by the calibration run so an
    admission can be traced back to the data that set its thresholds.
    """

    version: str
    control_prompts: Tuple[str, ...]
    margin_admit: float
    margin_reject: float
    min_views: int
    min_sam3_score: float
    min_depth_valid_px: int
    max_view_spread_m: float
    calibrated_on: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.version:
            raise ValueError("calibration artifact needs a version")
        if not self.control_prompts:
            raise ValueError("calibration artifact needs control_prompts")
        if self.margin_reject > self.margin_admit:
            raise ValueError(
                f"margin_reject ({self.margin_reject}) must not exceed "
                f"margin_admit ({self.margin_admit})")
        if self.min_views < 1:
            raise ValueError("min_views must be >= 1")

    def to_json(self) -> dict:
        return {
            "version": self.version,
            "control_prompts": list(self.control_prompts),
            "margin_admit": self.margin_admit,
            "margin_reject": self.margin_reject,
            "min_views": self.min_views,
            "min_sam3_score": self.min_sam3_score,
            "min_depth_valid_px": self.min_depth_valid_px,
            "max_view_spread_m": self.max_view_spread_m,
            "calibrated_on": dict(self.calibrated_on),
        }

    @classmethod
    def from_json(cls, d: dict) -> "AdmissionCalibration":
        missing = [k for k in ("version", "control_prompts", "margin_admit",
                               "margin_reject", "min_views", "min_sam3_score",
                               "min_depth_valid_px", "max_view_spread_m")
                   if k not in d]
        if missing:
            raise ValueError(f"calibration artifact missing keys: {missing}")
        return cls(
            version=str(d["version"]),
            control_prompts=tuple(str(p) for p in d["control_prompts"]),
            margin_admit=float(d["margin_admit"]),
            margin_reject=float(d["margin_reject"]),
            min_views=int(d["min_views"]),
            min_sam3_score=float(d["min_sam3_score"]),
            min_depth_valid_px=int(d["min_depth_valid_px"]),
            max_view_spread_m=float(d["max_view_spread_m"]),
            calibrated_on=dict(d.get("calibrated_on") or {}),
        )


def load_calibration(path: str) -> AdmissionCalibration:
    """Load + validate a calibration artifact. Fails loud: a policy running on
    a missing/malformed artifact would be a silent threshold change."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"admission calibration artifact not found: {path}")
    with open(path) as f:
        return AdmissionCalibration.from_json(json.load(f))


# ---------------------------------------------------------------------------
# The policies.
# ---------------------------------------------------------------------------


@runtime_checkable
class AdmissionPolicy(Protocol):
    """Consume raw evidence terms, return a graded decision."""

    policy_version: str

    def grade(self, evidences: Sequence[ObservationEvidence]) -> AdmissionDecision:
        ...


def _valid(e: ObservationEvidence, cal: AdmissionCalibration) -> bool:
    """Mask/depth validity + detector floor. Fail-closed: a view missing either
    measurement cannot vouch for anything."""
    return (e.sam3_score is not None and e.sam3_score >= cal.min_sam3_score
            and e.depth_valid_px is not None
            and e.depth_valid_px >= cal.min_depth_valid_px)


def _spread_ok(views: Sequence[ObservationEvidence],
               cal: AdmissionCalibration) -> bool:
    """Geometric consistency over the vouching views. ``None`` spread (fewer
    than two located views) is acceptable only when one view suffices —
    otherwise consistency was required and could not be measured."""
    spread = view_spread_m(views)
    if spread is None:
        return cal.min_views <= 1
    return spread <= cal.max_view_spread_m


class RelativeMarginPolicy:
    """Source-admissibility contract: held-out-calibrated target-vs-control margins + geometric
    and multi-view checks. The only policy that can REJECT."""

    kind = "margin"

    def __init__(self, calibration: AdmissionCalibration):
        self.calibration = calibration
        self.policy_version = f"{calibration.version}:{self.kind}"

    def grade(self, evidences: Sequence[ObservationEvidence]) -> AdmissionDecision:
        cal = self.calibration
        gradable = [e for e in evidences if e.margin is not None and _valid(e, cal)]
        if not gradable:
            return AdmissionDecision(
                status=UNKNOWN, policy_version=self.policy_version,
                reason="no valid margin-scored views to grade")

        passing = [e for e in gradable if e.margin >= cal.margin_admit]
        if (distinct_view_count(passing) >= cal.min_views
                and _spread_ok(passing, cal)):
            return AdmissionDecision(
                status=ADMITTED, policy_version=self.policy_version,
                reason=(f"{distinct_view_count(passing)} view(s) with margin >= "
                        f"{cal.margin_admit} (best "
                        f"{max(e.margin for e in passing):.4f}), geometrically "
                        f"consistent"),
                belief=max(e.sam3_score for e in passing),
                observation_ids=tuple(e.obs_id for e in passing))

        if all(e.margin < cal.margin_reject for e in gradable):
            return AdmissionDecision(
                status=REJECTED, policy_version=self.policy_version,
                reason=(f"all {len(gradable)} valid view(s) fail the "
                        f"negative-control margin (best "
                        f"{max(e.margin for e in gradable):.4f} < reject "
                        f"threshold {cal.margin_reject})"),
                observation_ids=tuple(e.obs_id for e in gradable))

        if passing and distinct_view_count(passing) < cal.min_views:
            why = (f"only {distinct_view_count(passing)} passing view(s); "
                   f"{cal.min_views} required")
        elif passing:
            why = "margin-passing views geometrically inconsistent"
        else:
            why = "margins between reject and admit thresholds"
        return AdmissionDecision(status=UNKNOWN,
                                 policy_version=self.policy_version, reason=why)


class GeometricPolicy:
    """The plain geometric rule: segmentation (detector score +
    mask/depth validity) and geometric (multi-view consistency) validation —
    no embedding margins. Never rejects: geometric agreement cannot disprove a
    consistent phantom, and measuring that blind spot is the point of the comparison."""

    kind = "geometric"

    def __init__(self, calibration: AdmissionCalibration):
        self.calibration = calibration
        self.policy_version = f"{calibration.version}:{self.kind}"

    def grade(self, evidences: Sequence[ObservationEvidence]) -> AdmissionDecision:
        cal = self.calibration
        valid = [e for e in evidences if _valid(e, cal)]
        if not valid:
            return AdmissionDecision(
                status=UNKNOWN, policy_version=self.policy_version,
                reason="no views pass segmentation validity")
        if distinct_view_count(valid) >= cal.min_views and _spread_ok(valid, cal):
            return AdmissionDecision(
                status=ADMITTED, policy_version=self.policy_version,
                reason=(f"{distinct_view_count(valid)} valid view(s), "
                        f"geometrically consistent"),
                belief=max(e.sam3_score for e in valid),
                observation_ids=tuple(e.obs_id for e in valid))
        if not _spread_ok(valid, cal):
            why = "valid views geometrically inconsistent"
        else:
            why = (f"only {distinct_view_count(valid)} valid view(s); "
                   f"{cal.min_views} required")
        return AdmissionDecision(status=UNKNOWN,
                                 policy_version=self.policy_version, reason=why)


def make_admission_policy(kind: str,
                          calibration: Optional[AdmissionCalibration]):
    """The policy switch. ``"off"`` (or a missing calibration) returns None —
    candidates then stay candidates, the default behavior."""
    if kind not in POLICY_KINDS:
        raise ValueError(f"unknown admission policy kind {kind!r}; "
                         f"expected one of {POLICY_KINDS}")
    if kind == "off":
        return None
    if calibration is None:
        raise ValueError(f"admission policy {kind!r} requires a calibration "
                         "artifact (see eval/calibrate_admission.py)")
    return (RelativeMarginPolicy(calibration) if kind == "margin"
            else GeometricPolicy(calibration))


# ---------------------------------------------------------------------------
# DB glue: read accumulated evidence, write graded decisions.
# ---------------------------------------------------------------------------

# Keep in lockstep with object_discovery._obs_dict (drift-guard in
# tests/test_admission_policy.py).
_EVIDENCE_QUERY = """
MATCH (o:Object {nodeSymbol: $sym})-[:HAS_OBSERVATION]->(obs:Observation)
RETURN obs.nodeSymbol AS ns, obs.timestamp_ns AS ts, obs.score AS sam3,
       obs.target_score AS target, obs.control_scores_json AS controls,
       obs.depth_valid_px AS depth_px, obs.target_prompt AS prompt,
       obs.view_x AS vx, obs.view_y AS vy, obs.view_z AS vz
ORDER BY obs.timestamp_ns, obs.nodeSymbol
"""


def load_observation_evidence(db, node_symbol: str,
                              prompt: Optional[str] = None) -> list:
    """Rebuild :class:`ObservationEvidence` for every observation of an object.

    When ``prompt`` is given, views scored against a DIFFERENT prompt keep
    their geometric/detector terms but drop their target/control scores —
    margins from different prompts are not comparable, and mixing them would
    let an object admitted-for-"mug" vouch for "coffee maker".
    """
    records, _, _ = db.execute(_EVIDENCE_QUERY, sym=node_symbol)
    out = []
    for r in records:
        obs_prompt = r.get("prompt")
        same_prompt = (prompt is None or (obs_prompt is not None
                       and obs_prompt.strip().lower() == prompt.strip().lower()))
        controls = {}
        if same_prompt and r.get("controls"):
            try:
                controls = {str(k): float(v)
                            for k, v in json.loads(r["controls"]).items()}
            except (ValueError, TypeError):
                controls = {}
        xyz = None
        if r.get("vx") is not None:
            xyz = (float(r["vx"]), float(r["vy"]), float(r["vz"]))
        out.append(ObservationEvidence(
            obs_id=r["ns"],
            timestamp_ns=r.get("ts"),
            sam3_score=r.get("sam3"),
            target_score=(r.get("target") if same_prompt else None),
            control_scores=controls,
            depth_valid_px=r.get("depth_px"),
            view_xyz=xyz,
            target_prompt=obs_prompt,
        ))
    return out


def grade_object(db, node_symbol: str, policy,
                 evidences: Optional[Sequence[ObservationEvidence]] = None,
                 prompt: Optional[str] = None) -> AdmissionDecision:
    """Grade ONE candidate and write the verdict through the sanctioned writers.

    Only a ``candidate`` is graded: trusted priors, human/or-earlier decisions
    (admitted, rejected) are never re-graded here — re-grading admitted
    objects on every query would let a later bad view revoke a sound
    admission, and rejections are sticky by design. ``unknown`` writes
    nothing (the candidate stays acquirable).

    ``evidences=None`` loads the object's accumulated persisted evidence (so
    two one-view queries can jointly reach ``min_views``).
    """
    current = admission_status(db, node_symbol)
    if current != CANDIDATE:
        return AdmissionDecision(
            status=UNKNOWN, policy_version=policy.policy_version,
            reason=f"not graded: admission_status is {current!r}, not candidate")

    if evidences is None:
        evidences = load_observation_evidence(db, node_symbol, prompt=prompt)
    decision = policy.grade(list(evidences))

    if decision.status == ADMITTED:
        admit_object(db, node_symbol, policy_version=decision.policy_version,
                     observation_ids=decision.observation_ids,
                     belief=decision.belief)
    elif decision.status == REJECTED:
        reject_object(db, node_symbol, policy_version=decision.policy_version,
                      reason=decision.reason)
    return decision
