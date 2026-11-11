"""Produce the admission-calibration artifact from a HELD-OUT archive.

The negative-control thresholds must come from a
calibration artifact produced on held-out data — a threshold tuned and
evaluated on the same archive would not generalize. This script runs the
EXACT evidence pipeline the warm query server runs (EmbeddingFilter ranking →
SAM3 frustum-aware detection → per-crop SigLIP2 target/control scoring via
``object_discovery.attach_crop_evidence``) over a labelled archive, with NO
database writes, and derives:

- ``margin_admit``   — above the phantom population (FP-quantile + epsilon);
- ``margin_reject``  — the decisive-phantom line (an FP quantile);
- ``max_view_spread_m`` — from real objects' multi-view geometry;
- the negative-control prompt set actually measured.

Labels are declared by the operator: ``--present`` prompts name classes truly
in the archive (their detections are treated as true positives), ``--absent``
prompts name classes truly NOT in it (any detection is a phantom). The report
carries the raw per-observation samples and a risk-coverage table so the
threshold trade-off is inspectable and re-derivable (selective grounding, not
a perfect existence test).

Runs in the SAM3 venv (GPU):

    $ADT4_ENV/sam3/bin/python \\
        -m openset_query.eval.calibrate_admission \\
        --archive-dir ~/adt4_output/<archive> \\
        --present "a chair" --present "a door" \\
        --absent "a wheelbarrow" --absent "a fire hydrant" \\
        --out ~/adt4_output/calib/sam3_admission_v1.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time

logger = logging.getLogger("eval.calibrate_admission")

#: Archive-generic negative controls: BACKGROUND/nuisance concepts every indoor
#: archive contains, so a detection crop matching one of them better than the
#: target prompt is evidence the "object" is really scenery. Deliberately NOT
#: object categories, superclasses, or image-quality captions: the first
#: calibration run on an indoor Spot archive showed "a piece of furniture" out-scoring
#: "a table" on true tables, "a door frame" out-scoring "a door" on true
#: doors, and "a blurry photo" out-scoring everything on flat sim renders --
#: such controls turn true positives into negative margins wholesale. A control
#: must be a caption a PHANTOM crop satisfies and a real object crop does not.
DEFAULT_CONTROL_PROMPTS = (
    "a blank wall",
    "a floor surface",
    "a ceiling",
    "an empty corridor",
    "a window",
    "a shadow on the ground",
    "a reflection on a shiny floor",
)


def _quantile(sorted_vals, q: float) -> float:
    """Inclusive linear-interpolation quantile over a sorted list (stdlib-only,
    matches numpy's default 'linear' method)."""
    if not sorted_vals:
        raise ValueError("quantile of empty population")
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return float(sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac)


def load_archive_keyframes(archive_dir: str):
    """Keyframe pool of an archive run from its image store
    (``<archive>/image_store`` or the store directory itself); poses are composed from
    ``<archive>/hydra/backend/dsg.json`` (or ``<archive>/dsg.json``) when present,
    else the stored odometry."""
    from openset_query import keyframe_store as ks

    store_dirs = ks.resolve_store_dirs(archive_dir)
    if not store_dirs:
        raise SystemExit(f"no image store under {archive_dir}")
    pose_lookup = None
    for cand in (os.path.join(archive_dir, "hydra", "backend", "dsg.json"),
                 os.path.join(archive_dir, "backend", "dsg.json"),
                 os.path.join(archive_dir, "dsg.json")):
        if os.path.exists(cand):
            from hydra_python.image_store import load_dsg

            pose_lookup = load_dsg(cand)
            break
    from hydra_python.image_store import ImageStore

    store = ImageStore.open(*store_dirs)
    try:
        pool = ks.load_keyframes_from_store(store, pose_lookup=pose_lookup)
    finally:
        store.close()
    if not pool:
        raise SystemExit(f"no loadable keyframes in {store_dirs}")
    logger.info("Loaded %d keyframes from %s", len(pool), store_dirs)
    return pool


def collect_prompt_evidence(prompt, pool, ef, segmenter, config,
                            control_prompts, max_frames, patience):
    """Run the server's detection+scoring pipeline for one prompt; return the
    per-observation evidence rows and per-object view stats. NO db writes."""
    from openset_query.evidence.margins import (
        ObservationEvidence, view_spread_m)
    from openset_query import object_discovery as od

    ranked = ef.rank(pool, [prompt])
    discovered = od.detect_frustum_aware(
        ranked, segmenter, [prompt], config,
        max_frames=max_frames, patience=patience)
    od.attach_crop_evidence(discovered, ef, prompt, control_prompts)

    rows, objects = [], []
    for i, disc in enumerate(discovered):
        evs = [ObservationEvidence(
                   obs_id=f"{prompt}#{i}@{o.timestamp_ns}",
                   timestamp_ns=o.timestamp_ns, sam3_score=o.score,
                   target_score=o.target_score,
                   control_scores=dict(o.control_scores or {}),
                   depth_valid_px=o.depth_valid_px, view_xyz=o.view_xyz,
                   target_prompt=prompt)
               for o in disc.observations]
        rows.extend(e.to_json() for e in evs)
        spread = view_spread_m(evs)
        objects.append({"prompt": prompt, "n_views": len(evs),
                        "view_spread_m": spread,
                        "best_margin": max((e.margin for e in evs
                                            if e.margin is not None),
                                           default=None)})
    return rows, objects


def derive_thresholds(tp_margins, fp_margins, *, fp_quantile, reject_quantile,
                      epsilon):
    """The calibration math, pure and unit-testable.

    ``margin_admit`` sits above the phantom population's ``fp_quantile``;
    ``margin_reject`` at its ``reject_quantile``. With NO phantom detections at
    all (the absent prompts triggered nothing) the archive gave us no FP
    population — fall back to the TP population's low tail and say so."""
    fp = sorted(fp_margins)
    tp = sorted(tp_margins)
    if fp:
        margin_admit = _quantile(fp, fp_quantile) + epsilon
        margin_reject = _quantile(fp, reject_quantile)
        basis = f"fp_population(n={len(fp)})"
    elif tp:
        margin_admit = _quantile(tp, 0.05)
        margin_reject = margin_admit - 10 * epsilon
        basis = f"tp_low_tail_fallback(n={len(tp)}; no phantom detections)"
    else:
        raise SystemExit("no detections at all — cannot calibrate")
    # An admit line below the reject line is degenerate; keep the contract.
    margin_reject = min(margin_reject, margin_admit)
    coverage = (sum(1 for m in tp if m >= margin_admit) / len(tp)) if tp else None
    fp_admitted = (sum(1 for m in fp if m >= margin_admit) / len(fp)) if fp else 0.0
    return {
        "margin_admit": round(margin_admit, 6),
        "margin_reject": round(margin_reject, 6),
        "basis": basis,
        "tp_kept_fraction": coverage,
        "fp_admitted_fraction": fp_admitted,
    }


def risk_coverage_table(tp_margins, fp_margins, n_points: int = 12):
    """Threshold sweep: (threshold, TP kept, FP admitted) — the selective-
    grounding curve."""
    allm = sorted(set(tp_margins) | set(fp_margins))
    if not allm:
        return []
    lo, hi = allm[0], allm[-1]
    out = []
    for i in range(n_points + 1):
        t = lo + (hi - lo) * i / n_points
        out.append({
            "threshold": round(t, 6),
            "tp_kept": (sum(1 for m in tp_margins if m >= t) / len(tp_margins)
                        if tp_margins else None),
            "fp_admitted": (sum(1 for m in fp_margins if m >= t) / len(fp_margins)
                            if fp_margins else None),
        })
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--archive-dir", required=True,
                   help="Held-out map dir (holding image_store/), or the store dir")
    p.add_argument("--present", action="append", default=[],
                   help="Prompt for a class truly PRESENT in the archive "
                        "(repeatable); its detections are the TP population")
    p.add_argument("--absent", action="append", default=[],
                   help="Prompt for a class truly ABSENT from the archive "
                        "(repeatable); its detections are the phantom (FP) "
                        "population")
    p.add_argument("--control-prompt", action="append", default=[],
                   help="Negative-control prompt (repeatable); default = the "
                        "built-in archive-generic set")
    p.add_argument("--out", required=True, help="Artifact JSON path")
    p.add_argument("--report", default=None,
                   help="Report JSON path (default: <out>.report.json)")
    p.add_argument("--version", default="sam3_admission_v1")
    p.add_argument("--device", default="cuda")
    p.add_argument("--conf", type=float, default=0.25,
                   help="SAM3 confidence threshold (match the serving config)")
    p.add_argument("--max-frames", type=int, default=40)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--min-views", type=int, default=2)
    p.add_argument("--fp-quantile", type=float, default=1.0,
                   help="FP-margin quantile margin_admit sits above (1.0 = max)")
    p.add_argument("--reject-quantile", type=float, default=0.5)
    p.add_argument("--epsilon", type=float, default=0.01)
    p.add_argument("--spread-quantile", type=float, default=0.95,
                   help="TP view-spread quantile for max_view_spread_m")
    args = p.parse_args(argv)
    if not args.present or not args.absent:
        raise SystemExit("need at least one --present and one --absent prompt")

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    from openset_query.config import Sam3FrontendConfig
    from openset_query.embedding_filter import EmbeddingFilter
    from openset_query.sam3_segmenter import Sam3Segmenter

    controls = tuple(args.control_prompt) or DEFAULT_CONTROL_PROMPTS
    config = Sam3FrontendConfig(device=args.device, conf_threshold=args.conf)
    pool = load_archive_keyframes(args.archive_dir)
    from openset_query import keyframe_store as ks
    cache = ks.embed_cache_path(ks.resolve_store_dirs(args.archive_dir)[0], "siglip2")
    ef = EmbeddingFilter(model="siglip2", device=args.device, cache_path=cache)
    ef.embed_images([kf.rgb_path for kf in pool], keys=[ef.frame_key(kf) for kf in pool])
    segmenter = Sam3Segmenter(config)
    segmenter._ensure_loaded()

    all_rows, all_objects = [], []
    t0 = time.time()
    for kind, prompts in (("present", args.present), ("absent", args.absent)):
        for prompt in prompts:
            rows, objects = collect_prompt_evidence(
                prompt, pool, ef, segmenter, config, controls,
                args.max_frames, args.patience)
            for r in rows:
                r["population"] = kind
            for o in objects:
                o["population"] = kind
            logger.info("%s %r: %d objects / %d observations",
                        kind, prompt, len(objects), len(rows))
            all_rows.extend(rows)
            all_objects.extend(objects)

    tp = [r["margin"] for r in all_rows
          if r["population"] == "present" and r.get("margin") is not None]
    fp = [r["margin"] for r in all_rows
          if r["population"] == "absent" and r.get("margin") is not None]
    thresholds = derive_thresholds(
        tp, fp, fp_quantile=args.fp_quantile,
        reject_quantile=args.reject_quantile, epsilon=args.epsilon)

    tp_spreads = sorted(o["view_spread_m"] for o in all_objects
                        if o["population"] == "present"
                        and o["view_spread_m"] is not None)
    max_spread = (_quantile(tp_spreads, args.spread_quantile)
                  if tp_spreads else 1.5)

    artifact = {
        "version": args.version,
        "control_prompts": list(controls),
        "margin_admit": thresholds["margin_admit"],
        "margin_reject": thresholds["margin_reject"],
        "min_views": args.min_views,
        "min_sam3_score": args.conf,
        "min_depth_valid_px": config.min_mask_depth_pixels,
        "max_view_spread_m": round(max_spread, 4),
        "calibrated_on": {
            "archive": os.path.abspath(args.archive_dir),
            "n_keyframes": len(pool),
            "present_prompts": args.present,
            "absent_prompts": args.absent,
            "date": time.strftime("%Y-%m-%d"),
            "threshold_basis": thresholds["basis"],
            "tp_kept_fraction": thresholds["tp_kept_fraction"],
            "fp_admitted_fraction": thresholds["fp_admitted_fraction"],
        },
    }
    # Validate through the loader contract before writing anything.
    from openset_query.evidence.admission_policy import AdmissionCalibration
    AdmissionCalibration.from_json(artifact)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    report_path = args.report or args.out + ".report.json"
    with open(report_path, "w") as f:
        json.dump({
            "artifact": artifact,
            "risk_coverage": risk_coverage_table(tp, fp),
            "tp_margins_n": len(tp), "fp_margins_n": len(fp),
            "objects": all_objects,
            "observations": all_rows,
            "elapsed_s": round(time.time() - t0, 1),
        }, f, indent=2, sort_keys=True)
    logger.info("Artifact -> %s\nReport -> %s\n%s", args.out, report_path,
                json.dumps(thresholds, indent=2))


if __name__ == "__main__":
    main()
