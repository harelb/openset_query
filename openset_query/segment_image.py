"""``POST /segment``: SAM3 on one uploaded image (not the keyframe pool).

Used by the ADT4 console for images that are not in the map, e.g. Penn's
quadrotor's nadir camera::

    request  {"image_b64": str (JPEG/PNG, plain base64 or a data: URI),
              "prompt": str, "conf": float?, "max_detections": int? (10),
              "overlay": bool? (true)}
    response {"ok": true, "prompt", "width", "height", "elapsed_s",
              "detections": [{"label", "score", "box_xyxy": [x0, y0, x1, y1],
                              "center_px": [u, v], "area_px",
                              "mask_rle": {"size": [H, W], "counts": str}}],
              "overlay_jpeg_b64": str | null}

Pixel coordinates are in the uploaded image. ``mask_rle`` is COCO RLE
(pycocotools ``encode``, column-major, counts as a string); without pycocotools
it is uncompressed column-major run lengths (``counts`` a list of ints,
starting with a run of zeros).
"""

from __future__ import annotations

import base64
import binascii
import io
import time

import numpy as np

MAX_IMAGE_BYTES = 20 * 1024 * 1024
DEFAULT_MAX_DETECTIONS = 10
PALETTE = [(255, 0, 255), (0, 200, 255), (255, 160, 0), (0, 230, 120), (255, 60, 60)]


class SegmentRequestError(ValueError):
    """A bad request (HTTP 400)."""


def decode_image(b64: str) -> np.ndarray:
    """base64 (or a data URI) of a JPEG/PNG -> HxWx3 uint8 RGB."""
    from PIL import Image

    if not isinstance(b64, str) or not b64:
        raise SegmentRequestError("image_b64 is required")
    if b64.startswith("data:"):
        b64 = b64.partition(",")[2]
    try:
        raw = base64.b64decode(b64, validate=False)
    except (binascii.Error, ValueError) as e:
        raise SegmentRequestError(f"image_b64 is not base64: {e}") from None
    if len(raw) > MAX_IMAGE_BYTES:
        raise SegmentRequestError(
            f"image larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MB"
        )
    try:
        with Image.open(io.BytesIO(raw)) as im:
            return np.asarray(im.convert("RGB"))
    except Exception as e:  # noqa: BLE001 — PIL raises many types
        raise SegmentRequestError(f"cannot decode the image: {e}") from None


def mask_rle(mask: np.ndarray) -> dict:
    m = np.asarray(mask, dtype=bool)
    try:
        from pycocotools import mask as cocomask

        rle = cocomask.encode(np.asfortranarray(m.astype(np.uint8)))
        counts = rle["counts"]
        return {"size": [int(m.shape[0]), int(m.shape[1])],
                "counts": counts.decode() if isinstance(counts, bytes) else counts}  # fmt: skip
    except ImportError:
        flat = m.flatten(order="F").astype(np.uint8)
        change = np.flatnonzero(np.diff(flat)) + 1
        bounds = np.concatenate([[0], change, [flat.size]])
        runs = np.diff(bounds).tolist()
        if flat.size and flat[0]:
            runs = [0] + runs
        return {"size": [int(m.shape[0]), int(m.shape[1])], "counts": runs}


def overlay_jpeg_b64(rgb: np.ndarray, dets: list, quality: int = 85) -> str:
    """The image with each detection's mask tinted, its box and score drawn."""
    from PIL import Image, ImageDraw

    from openset_query.overlay import annotate_rgb

    out = np.asarray(rgb).astype(np.uint8)
    for i, d in enumerate(dets):
        out = annotate_rgb(
            out,
            mask=d.mask,
            bbox=list(d.box),
            color=PALETTE[i % len(PALETTE)],
            thickness=3,
        )
    img = Image.fromarray(out)
    draw = ImageDraw.Draw(img)
    for i, d in enumerate(dets):
        x0, y0 = float(d.box[0]), float(d.box[1])
        text = f"{d.label} {float(d.score):.2f}"
        ty = max(0.0, y0 - 14)
        draw.rectangle(
            [x0, ty, x0 + 7 * len(text) + 4, ty + 13], fill=PALETTE[i % len(PALETTE)]
        )
        draw.text((x0 + 2, ty), text, fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def parse_request(req: dict) -> dict:
    if not isinstance(req, dict):
        raise SegmentRequestError("the body must be a JSON object")
    prompt = str(req.get("prompt") or "").strip()
    if not prompt:
        raise SegmentRequestError("empty prompt")
    conf = req.get("conf")
    try:
        conf = None if conf is None else float(conf)
        max_det = int(req.get("max_detections") or DEFAULT_MAX_DETECTIONS)
    except (TypeError, ValueError) as e:
        raise SegmentRequestError(f"bad conf / max_detections: {e}") from None
    if conf is not None and not 0.0 <= conf <= 1.0:
        raise SegmentRequestError("conf must be in [0, 1]")
    return {"rgb": decode_image(req.get("image_b64")), "prompt": prompt, "conf": conf,
            "max_detections": max(1, min(max_det, 100)), "overlay": req.get("overlay", True) is not False}  # fmt: skip


def segment(segmenter, req: dict) -> dict:
    """Run ``segmenter.segment(rgb, [prompt])`` for one parsed request (the caller
    holds whatever GPU lock the segmenter needs) and build the response."""
    p = parse_request(req)
    rgb = p["rgb"]
    t0 = time.perf_counter()
    dets = list(segmenter.segment(rgb, [p["prompt"]]) or [])
    if p["conf"] is not None:
        dets = [d for d in dets if float(d.score) >= p["conf"]]
    dets.sort(key=lambda d: -float(d.score))
    dets = dets[: p["max_detections"]]
    elapsed = time.perf_counter() - t0
    out = []
    for d in dets:
        x0, y0, x1, y1 = (float(v) for v in d.box)
        m = np.asarray(d.mask, dtype=bool)
        out.append({"label": d.label, "score": round(float(d.score), 4),
                    "box_xyxy": [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)],
                    "center_px": [round((x0 + x1) / 2, 1), round((y0 + y1) / 2, 1)],
                    "area_px": int(m.sum()), "mask_rle": mask_rle(m)})  # fmt: skip
    return {"ok": True, "prompt": p["prompt"], "width": int(rgb.shape[1]), "height": int(rgb.shape[0]),
            "elapsed_s": round(elapsed, 3), "detections": out,
            "overlay_jpeg_b64": overlay_jpeg_b64(rgb, dets) if p["overlay"] else None}  # fmt: skip
