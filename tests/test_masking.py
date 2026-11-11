"""Tests for eval.masking (Tier 1).

Synthetic cells prove the semantics portably; the real-corpus tests
(determinism + exclusivity over grounding_raw_v1) skip when the corpus is
not on this machine.
"""
import json
from pathlib import Path

import pytest

from openset_query.eval.masking import (
    DEFAULT_CORPUS_ROOT,
    MaskArtifact,
    accept_detections,
    apply_mask_to_detections,
    cell_frame_universe,
    iter_cells,
    load_cell,
    mask_class_support,
    mask_random_fraction,
    merge_detections,
    replay_cell,
    replay_corpus,
)
from openset_query.evidence import EpistemicStatus, ReasonCode

_CORPUS = Path(DEFAULT_CORPUS_ROOT)
_needs_corpus = pytest.mark.skipif(
    not _CORPUS.is_dir(), reason="grounding_raw_v1 corpus not on this machine")


def _det(score=0.5, ts=(1, 2, 3), n_pixels=3000, label="cup", xyz=(0, 0, 0)):
    return {"label": label, "score": score, "x": xyz[0], "y": xyz[1],
            "z": xyz[2], "n_pixels": n_pixels, "n_frames": len(set(ts)),
            "frame_ts": list(ts)}


def _cell(dets, prompt_class="cup", env="synth", cache12="cafe12"):
    return {"schema": "grounding_raw_v1", "env": env, "cache12": cache12,
            "prompt_class": prompt_class, "role": "present", "t_unix": 1000.0,
            "n_detections": len(dets), "detections": dets}


# ---------------------------------------------------------------------------
# Mask artifact + filter
# ---------------------------------------------------------------------------


def test_mask_artifact_roundtrip(tmp_path):
    m = mask_random_fraction([5, 1, 3, 2, 4], 0.4, seed=7, map_name="floor3")
    assert m.params == {"fraction": 0.4, "universe_size": 5}
    assert len(m.masked_frame_ts) == 2
    p = tmp_path / "mask.json"
    m.save(p)
    again = MaskArtifact.load(p)
    assert again == m
    # Deterministic: same seed -> same mask; different seed -> (almost surely)
    # a different sample of this tiny universe is not guaranteed, so assert
    # only same-seed equality (the determinism that matters).
    assert mask_random_fraction([5, 1, 3, 2, 4], 0.4, seed=7,
                                map_name="floor3") == m


def test_apply_mask_filters_recomputes_and_fails_closed():
    dets = [
        _det(ts=(1, 2, 3), n_pixels=3000),
        _det(ts=(1, 1, 2), n_pixels=900),         # duplicate ts
        {"label": "cup", "score": 0.9, "x": 0, "y": 0, "z": 0,
         "n_pixels": 5000, "n_frames": 4},          # no frame_ts -> fail closed
    ]
    out = apply_mask_to_detections(dets, {1})
    assert len(out) == 2
    assert out[0]["frame_ts"] == [2, 3]
    assert out[0]["n_frames"] == 2
    assert out[0]["n_pixels"] == 2000  # 3000 * 2/3
    assert out[1]["frame_ts"] == [2]
    assert out[1]["n_frames"] == 1
    assert out[1]["n_pixels"] == 300   # 900 * 1/3 (list length, not distinct)
    # fully-masked detection disappears
    assert apply_mask_to_detections([_det(ts=(1,))], {1}) == []
    # inputs not mutated
    assert dets[0]["frame_ts"] == [1, 2, 3]


def test_empty_mask_is_identity_through_replay():
    cell = _cell([_det(score=0.6, ts=(1, 2), n_pixels=100)])
    r_none, _ = replay_cell(cell, min_score=0.5)
    r_empty, _ = replay_cell(
        cell, min_score=0.5,
        mask=MaskArtifact(map="synth", generator="none", seed=0,
                          masked_frame_ts=()))
    assert r_none.status == r_empty.status
    assert r_none.supporting == r_empty.supporting
    assert r_none.belief == r_empty.belief


# ---------------------------------------------------------------------------
# Replay semantics (synthetic)
# ---------------------------------------------------------------------------


def test_replay_supported_then_removed_by_full_support_mask():
    cell = _cell([_det(score=0.7, ts=(1, 2, 3)),
                  _det(score=0.4, ts=(3, 4), xyz=(5, 5, 0))])
    r, prov = replay_cell(cell, min_score=0.3)
    assert r.status is EpistemicStatus.SUPPORTED
    assert r.belief == 0.7
    assert prov["n_merged"] == 2  # 5 m apart: not fused

    mask = mask_class_support(cell)
    assert set(mask.masked_frame_ts) == {1, 2, 3, 4}
    r2, prov2 = replay_cell(cell, min_score=0.3, mask=mask)
    assert r2.status is EpistemicStatus.UNKNOWN
    assert r2.reason is ReasonCode.INSUFFICIENT_COVERAGE  # coverage-None negative
    assert prov2["n_after_mask"] == 0


def test_partial_mask_keeps_surviving_support():
    # det A survives on frame 3; det B is fully hidden.
    cell = _cell([_det(score=0.7, ts=(1, 2, 3)),
                  _det(score=0.9, ts=(1, 2), xyz=(9, 9, 0), n_pixels=5000)])
    r, _ = replay_cell(cell, min_score=0.3, mask=MaskArtifact(
        map="synth", generator="manual", seed=0, masked_frame_ts=(1, 2)))
    assert r.status is EpistemicStatus.SUPPORTED
    # Only the surviving cluster supports; the hidden 0.9 det contributes
    # nothing (belief comes from the survivor).
    assert r.belief == 0.7
    assert len(r.supporting) == 1


def test_mask_can_defeat_multiview_gate():
    # 2 frames + small mask: passes min_frames=2; hiding one frame drops it
    # to single-view + small mask -> gate rejects -> UNKNOWN.
    cell = _cell([_det(score=0.9, ts=(1, 2), n_pixels=10)])
    r, _ = replay_cell(cell, min_score=0.5)
    assert r.status is EpistemicStatus.SUPPORTED
    r2, _ = replay_cell(cell, min_score=0.5, mask=MaskArtifact(
        map="synth", generator="manual", seed=0, masked_frame_ts=(2,)))
    assert r2.status is EpistemicStatus.UNKNOWN


def test_negative_replay_never_contradicts():
    # Nothing admissible even unmasked: archive scan asserts no spatial
    # coverage -> UNKNOWN / INSUFFICIENT_COVERAGE, never CONTRADICTED.
    cell = _cell([_det(score=0.1)])
    r, _ = replay_cell(cell, min_score=0.5)
    assert r.status is EpistemicStatus.UNKNOWN
    assert r.reason is ReasonCode.INSUFFICIENT_COVERAGE


def test_vendored_accept_and_merge_semantics():
    # accept: score gate inclusive; multi-view OR big-mask.
    dets = [_det(score=0.5, ts=(1,), n_pixels=800),   # single view, big mask
            _det(score=0.5, ts=(1, 2), n_pixels=1),   # multi-view, tiny mask
            _det(score=0.49, ts=(1, 2), n_pixels=9999),  # under score gate
            _det(score=0.5, ts=(1,), n_pixels=1)]     # neither arm
    assert len(accept_detections(dets, min_score=0.5)) == 2
    # merge: same label within 1 m fuses; frame_ts unioned -> n_frames exact.
    merged = merge_detections([
        _det(score=0.8, ts=(1, 2), xyz=(0, 0, 0)),
        _det(score=0.4, ts=(2, 3), xyz=(0.5, 0, 0)),
        _det(score=0.4, ts=(9,), xyz=(0.5, 0, 0), label="mop"),
    ])
    assert len(merged) == 2
    cup = next(m for m in merged if m["label"] == "cup")
    assert cup["frame_ts"] == [1, 2, 3]
    assert cup["n_frames"] == 3
    assert cup["score"] == 0.8


# ---------------------------------------------------------------------------
# Real-corpus acceptance (skip when the corpus is absent)
# ---------------------------------------------------------------------------


@_needs_corpus
def test_corpus_determinism_same_mask_identical_assertions():
    cells = list(iter_cells(env="floor3"))
    assert cells
    universe = sorted({t for _, _, _, p in cells[:6]
                       for t in cell_frame_universe(load_cell(p))})
    mask = mask_random_fraction(universe, 0.5, seed=42, map_name="floor3")
    a = replay_corpus(env="floor3", min_score=0.4, mask=mask)
    b = replay_corpus(env="floor3", min_score=0.4, mask=mask)
    assert {k: v.to_json() for k, v in a.items()} == \
           {k: v.to_json() for k, v in b.items()}
    assert len(a) == len(cells)


@_needs_corpus
def test_corpus_exclusivity_property():
    """The A8 exclusivity acceptance, in its provable form (min_frames=1,
    no pixel arm, so the gate is per-frame monotone): after masking, a class
    is SUPPORTED iff some raw detection clears the score gate with >= 1
    unmasked frame — i.e. masking removes exactly the assertions exclusively
    supported by masked frames."""
    checked = 0
    for env, cache12, prompt_class, path in iter_cells(env="floor3"):
        cell = load_cell(path)
        universe = cell_frame_universe(cell)
        if not universe:
            continue
        mask = mask_random_fraction(universe, 0.6, seed=13, map_name=env)
        masked = mask.frames
        r, _ = replay_cell(cell, min_score=0.4, mask=mask,
                           min_frames=1, min_pixels_single=10**9)
        expect_supported = any(
            float(d.get("score", 0) or 0) >= 0.4
            and any(int(t) not in masked for t in (d.get("frame_ts") or ()))
            for d in cell["detections"])
        assert (r.status is EpistemicStatus.SUPPORTED) == expect_supported, \
            (env, cache12, prompt_class)
        checked += 1

        # Full-support mask removes the assertion, unconditionally.
        r_all, _ = replay_cell(cell, min_score=0.4,
                               mask=mask_class_support(cell),
                               min_frames=1, min_pixels_single=10**9)
        assert r_all.status is not EpistemicStatus.SUPPORTED
    assert checked >= 5  # the floor3 corpus is non-trivial


@_needs_corpus
def test_corpus_unmasked_replay_sanity():
    """Present-role cells should generally certify; the replay agrees with
    the corpus's own role labels on the easy direction (present -> some
    admissible support at a low bar) for at least most cells."""
    agree = total = 0
    for env, cache12, prompt_class, path in iter_cells(env="floor3"):
        cell = load_cell(path)
        if cell.get("role") != "present":
            continue
        r, _ = replay_cell(cell, min_score=0.3)
        total += 1
        agree += r.status is EpistemicStatus.SUPPORTED
    assert total > 0
    assert agree / total >= 0.7
