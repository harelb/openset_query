"""Observation masking + Tier-1 evidence replay.

A :class:`MaskArtifact` names a reproducible set of hidden frame timestamps
``{map, generator, seed, masked_frame_ts}``. Tier 1 replays the RAW SAM3
detection corpus (``~/adt4_output/motion_bench/grounding_raw_v1``: one JSON
cell per (map, prompt_class), each detection carrying its supporting
``frame_ts`` list) with the mask applied — zero GPU, exactly the offline
filter the dcist_sim currency horizon already proved sound
(``grounding/adapters.py`` ``CurrentObservedOperator._pre_admissibility``):
a detection's ``frame_ts`` is filtered, ``n_frames`` recomputed from the
SURVIVING distinct timestamps so the multi-view gate judges surviving
support, never pooled hidden support; a detection with no surviving frame
provenance fails closed (dropped, not passed).

``n_pixels`` caveat: per-frame pixel counts were never recorded in the
corpus, so the masked ``n_pixels`` is the proportional estimate
``round(n_pixels * kept/total)`` — an assumption, stated here rather than
hidden, and irrelevant whenever the multi-view arm of the gate decides.

The replay then re-runs the SHIPPED epistemic chain: ``accept_detections``
-> ``merge_detections`` (vendored verbatim from dcist_sim ``requery`` @
7e8bf5a — keep in lockstep) -> the vendored ``compute_status``. Everything
is deterministic (no clocks, no uuids): same mask -> byte-identical
assertions, which is the A8 acceptance.

A completed archive scan that found nothing admissible is NEGATIVE evidence
with ``coverage=None`` — an archive scan asserts nothing about spatial
coverage of the world (same stance as ``ArchiveQueryOperator``), so masked
absence never manufactures a CONTRADICTED; certified absence needs the
visibility instrument or a coverage-bearing search episode.

Tier 2 (masking a real map's ingest / keyframe pool, hidden frames revealed
as ``reprocess_memory`` acquisitions, ``--mask`` joining the run_openset
resume key) is the follow-up — this module owns the artifact + the replay
semantics both tiers share.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

from openset_query.evidence import (
    Assertion,
    EvidenceRecord,
    EvidenceSource,
    GroundingQuery,
    Polarity,
    compute_status,
)
from openset_query.evidence.store import assertion_id_for

DEFAULT_CORPUS_ROOT = Path.home() / "adt4_output/motion_bench/grounding_raw_v1"

# The shipped gate defaults (requery.accept_detections / LegacyGate).
DEFAULT_MIN_FRAMES = 2
DEFAULT_MIN_PIXELS_SINGLE = 800
_MERGE_RADIUS_M = 1.0
_DETECTION_KEYS = ("label", "score", "x", "y", "z", "n_pixels", "n_frames")


# ---------------------------------------------------------------------------
# Mask artifact
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MaskArtifact:
    """A reproducible hidden-frame set. ``generator`` + ``seed`` + ``params``
    say how it was made; ``masked_frame_ts`` is the materialized set (sorted
    tuple) so a consumer never needs the generator to apply it."""

    map: str
    generator: str
    seed: int
    masked_frame_ts: tuple
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(
            self, "masked_frame_ts", tuple(sorted(int(t) for t in self.masked_frame_ts))
        )

    @property
    def frames(self) -> frozenset:
        return frozenset(self.masked_frame_ts)

    @property
    def mask_id(self) -> str:
        """Stable content identity (generator:seed:hash-of-frames) — joins the
        run_openset resume key so a masked run never resume-skips an unmasked
        row or a different mask's row."""
        import hashlib

        digest = hashlib.sha256(
            ",".join(str(t) for t in self.masked_frame_ts).encode()
        ).hexdigest()[:12]
        return f"{self.generator}:{self.seed}:{digest}"

    def to_json(self) -> dict:
        return {
            "map": self.map,
            "generator": self.generator,
            "seed": self.seed,
            "masked_frame_ts": list(self.masked_frame_ts),
            "params": self.params,
        }

    @classmethod
    def from_json(cls, d: dict) -> "MaskArtifact":
        return cls(
            map=str(d["map"]),
            generator=str(d["generator"]),
            seed=int(d["seed"]),
            masked_frame_ts=tuple(d.get("masked_frame_ts", ())),
            params=dict(d.get("params", {})),
        )

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), sort_keys=True) + "\n")

    @classmethod
    def load(cls, path) -> "MaskArtifact":
        return cls.from_json(json.loads(Path(path).read_text()))


def mask_random_fraction(
    frame_universe, fraction: float, *, seed: int, map_name: str
) -> MaskArtifact:
    """Hide a seeded random ``fraction`` of the (sorted) frame universe."""
    universe = sorted(int(t) for t in set(frame_universe))
    k = int(round(len(universe) * float(fraction)))
    rng = random.Random(seed)
    masked = rng.sample(universe, k) if k else []
    return MaskArtifact(
        map=map_name,
        generator="random_fraction",
        seed=seed,
        masked_frame_ts=tuple(masked),
        params={"fraction": float(fraction), "universe_size": len(universe)},
    )


def mask_class_support(cell: dict, *, seed: int = 0) -> MaskArtifact:
    """Hide EVERY frame supporting the cell's class — the exclusivity
    instrument: after this mask, no assertion for the class may survive."""
    frames = sorted(
        {
            int(t)
            for det in cell.get("detections", ())
            for t in (det.get("frame_ts") or ())
        }
    )
    return MaskArtifact(
        map=str(cell.get("env", "")),
        generator="class_support",
        seed=seed,
        masked_frame_ts=tuple(frames),
        params={"prompt_class": cell.get("prompt_class")},
    )


# ---------------------------------------------------------------------------
# The mask filter (mirror of CurrentObservedOperator._pre_admissibility)
# ---------------------------------------------------------------------------


def apply_mask_to_detections(dets, masked_frames) -> list:
    """Drop hidden support from raw detections, fail-closed.

    A detection without ``frame_ts`` provenance cannot prove any of its
    support survives the mask -> dropped. ``n_frames`` is recomputed from
    surviving distinct timestamps; ``n_pixels`` proportionally estimated
    (module docstring). Inputs are not mutated.
    """
    masked = frozenset(int(t) for t in masked_frames)
    kept = []
    for det in dets:
        frame_ts = det.get("frame_ts")
        if not isinstance(frame_ts, (list, tuple)) or not frame_ts:
            continue  # survival unprovable -> fail closed
        try:
            all_ts = [int(t) for t in frame_ts]
        except (TypeError, ValueError):
            continue
        surviving = [t for t in all_ts if t not in masked]
        if not surviving:
            continue
        adjusted = dict(det)
        adjusted["frame_ts"] = surviving
        adjusted["n_frames"] = len(set(surviving))
        try:
            n_pixels = int(det.get("n_pixels", 0) or 0)
        except (TypeError, ValueError):
            n_pixels = 0
        adjusted["n_pixels"] = int(round(n_pixels * len(surviving) / len(all_ts)))
        kept.append(adjusted)
    return kept


# ---------------------------------------------------------------------------
# Accept + merge, vendored verbatim from dcist_sim requery @ 7e8bf5a
# (keep semantics in lockstep; docstrings condensed, logic unchanged)
# ---------------------------------------------------------------------------


def accept_detections(
    dets,
    *,
    min_score,
    min_frames=DEFAULT_MIN_FRAMES,
    min_pixels_single=DEFAULT_MIN_PIXELS_SINGLE,
) -> list:
    """The shipped anti-hallucination gate: ``score >= min_score`` AND
    (multi-view ``n_frames >= min_frames`` OR big-mask ``n_pixels >=
    min_pixels_single``). Missing/malformed keys fail closed."""
    kept = []
    for det in dets:
        try:
            score = float(det.get("score", 0.0))
            n_frames = int(det.get("n_frames", 0) or 0)
            n_pixels = int(det.get("n_pixels", 0) or 0)
        except (TypeError, ValueError):
            continue
        if score < float(min_score):
            continue
        if n_frames >= int(min_frames) or n_pixels >= int(min_pixels_single):
            kept.append(det)
    return kept


def merge_detections(dets, radius_m: float = _MERGE_RADIUS_M) -> list:
    """Greedy same-label 3D fusion, highest-score-first: score-weighted mean
    position, max score, summed n_pixels; ``n_frames`` = size of the UNIONed
    ``frame_ts`` when every contributor has provenance, else max (sound lower
    bound, ``frame_ts`` omitted)."""
    fused = []
    for det in sorted(
        dets, key=lambda d: float(d.get("score", 0.0) or 0.0), reverse=True
    ):
        label = str(det.get("label", ""))
        key = label.strip().lower()
        x = float(det.get("x", 0.0) or 0.0)
        y = float(det.get("y", 0.0) or 0.0)
        z = float(det.get("z", 0.0) or 0.0)
        score = float(det.get("score", 0.0) or 0.0)
        n_pixels = int(det.get("n_pixels", 0) or 0)
        n_frames = int(det.get("n_frames", 0) or 0)
        weight = max(score, 1e-6)
        frame_ts = det.get("frame_ts") or None
        try:
            frame_ts = {int(t) for t in frame_ts} if frame_ts is not None else None
        except (TypeError, ValueError):
            frame_ts = None

        match = None
        for cand in fused:
            if cand["_key"] != key:
                continue
            if (
                math.sqrt(
                    (cand["x"] - x) ** 2 + (cand["y"] - y) ** 2 + (cand["z"] - z) ** 2
                )
                <= radius_m
            ):
                match = cand
                break

        if match is None:
            fused.append(
                {
                    "label": label,
                    "score": score,
                    "x": x,
                    "y": y,
                    "z": z,
                    "n_pixels": n_pixels,
                    "n_frames": n_frames,
                    "_key": key,
                    "_w": weight,
                    "_ts": frame_ts,
                }
            )
            continue
        w0 = match["_w"]
        total = w0 + weight
        match["x"] = (match["x"] * w0 + x * weight) / total
        match["y"] = (match["y"] * w0 + y * weight) / total
        match["z"] = (match["z"] * w0 + z * weight) / total
        match["_w"] = total
        match["score"] = max(match["score"], score)
        match["n_pixels"] += n_pixels
        if match["_ts"] is not None and frame_ts is not None:
            match["_ts"] |= frame_ts
        else:
            match["_ts"] = None  # provenance broken: fall back to the bound
        match["n_frames"] = max(match["n_frames"], n_frames)

    out = []
    for d in fused:
        merged = {k: d[k] for k in _DETECTION_KEYS}
        if d["_ts"]:
            merged["frame_ts"] = sorted(d["_ts"])
            merged["n_frames"] = len(d["_ts"])
        out.append(merged)
    return out


# ---------------------------------------------------------------------------
# Tier-1 replay
# ---------------------------------------------------------------------------


def load_cell(path) -> dict:
    return json.loads(Path(path).read_text())


def iter_cells(corpus_root=DEFAULT_CORPUS_ROOT, env: str | None = None):
    """Yield (env, cache12, prompt_class, path) for every corpus cell."""
    root = Path(corpus_root)
    env_dirs = [root / env] if env else sorted(p for p in root.iterdir() if p.is_dir())
    for env_dir in env_dirs:
        for cache_dir in sorted(p for p in env_dir.iterdir() if p.is_dir()):
            for cell_path in sorted(cache_dir.glob("*.json")):
                yield env_dir.name, cache_dir.name, cell_path.stem, cell_path


def cell_frame_universe(cell: dict) -> list:
    """Every frame timestamp with recorded support in the cell (the corpus
    does not record the full SCANNED frame list, only detection-bearing
    frames; Tier-2 masks over a map's actual keyframe list instead)."""
    return sorted(
        {
            int(t)
            for det in cell.get("detections", ())
            for t in (det.get("frame_ts") or ())
        }
    )


def replay_cell(
    cell: dict,
    *,
    min_score: float,
    mask: MaskArtifact | None = None,
    min_frames: int = DEFAULT_MIN_FRAMES,
    min_pixels_single: int = DEFAULT_MIN_PIXELS_SINGLE,
    required_confidence: float = 0.0,
) -> tuple:
    """Mask -> accept -> merge -> compute_status for one corpus cell.

    Returns ``(GroundingResult, provenance dict)``. Fully deterministic:
    evidence ids derive from the cell identity + cluster index; timestamps
    from the data (max supporting frame_ts; the cell's own capture time for
    the negative record).
    """
    prompt_class = str(cell["prompt_class"])
    query = GroundingQuery(
        predicate="exists",
        args=(prompt_class,),
        required_confidence=required_confidence,
    )
    dets = list(cell.get("detections", ()))
    n_raw = len(dets)
    if mask is not None:
        dets = apply_mask_to_detections(dets, mask.frames)
    accepted = accept_detections(
        dets,
        min_score=min_score,
        min_frames=min_frames,
        min_pixels_single=min_pixels_single,
    )
    merged = merge_detections(accepted)

    cell_id = f"{cell.get('env', '')}_{cell.get('cache12', '')}_{prompt_class}"
    provenance_base = {
        "cell": cell_id,
        "n_raw": n_raw,
        "n_after_mask": len(dets),
        "n_accepted": len(accepted),
        "n_merged": len(merged),
        "mask": (
            {
                "generator": mask.generator,
                "seed": mask.seed,
                "n_masked": len(mask.masked_frame_ts),
            }
            if mask is not None
            else None
        ),
        "admissibility": {
            "min_score": min_score,
            "min_frames": min_frames,
            "min_pixels_single": min_pixels_single,
        },
    }

    if merged:
        records = []
        for i, det in enumerate(merged):
            prov = dict(provenance_base)
            prov.update(
                {k: det.get(k) for k in _DETECTION_KEYS if det.get(k) is not None}
            )
            ts = det.get("frame_ts") or ()
            if ts:
                prov["frame_ts_range"] = [min(ts), max(ts)]
            records.append(
                EvidenceRecord(
                    evidence_id=f"{cell_id}_hit_{i}",
                    source=EvidenceSource.PRIOR_ARCHIVE,
                    polarity=Polarity.POSITIVE,
                    predicate=query.predicate,
                    args=query.args,
                    t_recorded_ns=max(ts) if ts else 0,
                    confidence=det.get("score"),
                    t_collected_ns=max(ts) if ts else None,
                    session_id=cell.get("cache12"),
                    provenance=prov,
                )
            )
        return compute_status(query, records), provenance_base

    # Completed scan, nothing admissible: an archive scan asserts nothing
    # about spatial coverage -> coverage None (never CONTRADICTED from here).
    t_cell_ns = int(float(cell.get("t_unix", 0)) * 1e9)
    negative = EvidenceRecord(
        evidence_id=f"{cell_id}_miss",
        source=EvidenceSource.PRIOR_ARCHIVE,
        polarity=Polarity.NEGATIVE,
        predicate=query.predicate,
        args=query.args,
        t_recorded_ns=t_cell_ns,
        session_id=cell.get("cache12"),
        coverage=None,
        provenance=provenance_base,
    )
    return compute_status(query, [negative]), provenance_base


# ---------------------------------------------------------------------------
# Tier-2: masking real maps (ingest-time graph surgery + query-time visibility)
# ---------------------------------------------------------------------------


def entry_visible(observations, masked_frames) -> bool:
    """Is a catalog entry visible under the mask?

    Visible iff at least one of its observations' ``frame_ts`` is unmasked.
    An entry with NO observation provenance fails closed under a non-empty
    mask (same discipline as :func:`apply_mask_to_detections`): it cannot
    prove any unmasked frame ever saw it.
    """
    masked = frozenset(int(t) for t in masked_frames)
    if not masked:
        return True
    if not observations:
        return False
    for obs in observations:
        try:
            if int(obs.get("frame_ts")) not in masked:
                return True
        except (TypeError, ValueError):
            continue
    return False


def apply_mask_to_graph(db, mask: MaskArtifact) -> dict:
    """Hide the mask's frames from an INGESTED map (Tier 2, ingest side).

    Three deletions, in order:

    1. ``:Object`` nodes whose EVERY Observation timestamp is masked — the
       map was built without those frames, so the object was never seen.
       An object with NO Observation nodes is structural (khronos base
       geometry without per-view provenance) and is deliberately KEPT — the
       mask can only speak about frames, and hiding provenance-free objects
       would erase most of a map that never ingested observations. This is
       the documented asymmetry with the detection-level fail-closed rule.
    2. the masked ``:Observation`` nodes themselves (including those on
       surviving mixed-support objects — hidden evidence must not remain
       queryable by the provenance chain).
    3. masked ``:TrajectoryFrame`` nodes (the text-query frame pool).

    MUTATES the shared database: re-ingest with ``--force`` before running
    unmasked experiments again. Idempotent. Returns per-step counts.
    """
    masked = list(mask.masked_frame_ts)
    counts = {}
    records, _, _ = db.execute(
        """
    MATCH (o:Object)-[:HAS_OBSERVATION]->(obs:Observation)
    WITH o, collect(obs.timestamp_ns) AS ts
    WHERE all(t IN ts WHERE t IN $masked)
    DETACH DELETE o
    RETURN count(o) AS n
    """,
        masked=masked,
    )
    counts["objects_removed"] = records[0]["n"] if records else 0
    records, _, _ = db.execute(
        "MATCH (obs:Observation) WHERE obs.timestamp_ns IN $masked "
        "DETACH DELETE obs RETURN count(obs) AS n",
        masked=masked,
    )
    counts["observations_removed"] = records[0]["n"] if records else 0
    records, _, _ = db.execute(
        "MATCH (f:TrajectoryFrame) WHERE f.timestamp_ns IN $masked "
        "DETACH DELETE f RETURN count(f) AS n",
        masked=masked,
    )
    counts["trajectory_frames_removed"] = records[0]["n"] if records else 0
    return counts


def filter_keyframes(keyframes, masked_frames) -> list:
    """Drop keyframes whose ``timestamp_ns`` is masked (the live SAM3 query
    pool side of Tier 2; ``keyframe_store.load_keyframes_from_db`` applies it)."""
    masked = frozenset(int(t) for t in masked_frames)
    if not masked:
        return list(keyframes)
    return [kf for kf in keyframes if int(kf.timestamp_ns) not in masked]


def replay_corpus(
    corpus_root=DEFAULT_CORPUS_ROOT,
    *,
    env: str | None = None,
    min_score: float,
    mask: MaskArtifact | None = None,
    min_frames: int = DEFAULT_MIN_FRAMES,
    min_pixels_single: int = DEFAULT_MIN_PIXELS_SINGLE,
) -> dict:
    """Replay every cell -> ``{(env, cache12, prompt_class): Assertion}``.

    Assertion ids are the deterministic query hash (store discipline);
    ``computed_at_ns`` is pinned to 0 so two replays of the same corpus +
    mask are byte-identical (the determinism acceptance).
    """
    out = {}
    for env_name, cache12, prompt_class, path in iter_cells(corpus_root, env):
        result, _prov = replay_cell(
            load_cell(path),
            min_score=min_score,
            mask=mask,
            min_frames=min_frames,
            min_pixels_single=min_pixels_single,
        )
        out[(env_name, cache12, prompt_class)] = Assertion.from_result(
            result, computed_at_ns=0, assertion_id=assertion_id_for(result.query)
        )
    return out
