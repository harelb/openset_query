"""Mask/bbox image annotation shared by the viz server and the chat agent.

Dependency-light on purpose (numpy + Pillow + stdlib only): the chat process has
no torch/rerun/SAM3, but still needs to bake SAM3 masks/boxes into images.
"""

from __future__ import annotations

import base64
import io
import os

import numpy as np
from PIL import Image

DEFAULT_COLOR = (255, 0, 255)
DEFAULT_ALPHA = 0.45
DEFAULT_THICKNESS = 2


def annotate_rgb(
    rgb,
    mask_path=None,
    bbox=None,
    color=DEFAULT_COLOR,
    alpha=DEFAULT_ALPHA,
    thickness=DEFAULT_THICKNESS,
    mask=None,
):
    """Return a new uint8 RGB array with the mask tinted and/or the bbox outlined.

    The mask is either a boolean/uint8 ``mask`` array (e.g. ``ImageStore.mask(obs)``)
    or the path of a mask PNG (``mask_path``); ``mask`` wins.
    """
    out = np.asarray(rgb).astype(np.uint8).copy()
    m = None
    if mask is not None:
        m = np.asarray(mask)
        m = (m.astype(np.uint8) * 255) if m.dtype == bool else m.astype(np.uint8)
    elif mask_path and os.path.exists(mask_path):
        m = np.asarray(Image.open(mask_path).convert("L"))
    if m is not None:
        if m.shape != out.shape[:2]:
            m = np.asarray(Image.fromarray(m).resize((out.shape[1], out.shape[0])))
        sel = m > 127
        f = out.astype(np.float32)
        for c in range(3):
            f[..., c][sel] = (1.0 - alpha) * f[..., c][sel] + alpha * color[c]
        out = f.astype(np.uint8)
    if bbox:
        h, w = out.shape[:2]
        x0, y0, x1, y1 = bbox
        x0 = max(0, min(w - 1, int(x0)))
        x1 = max(0, min(w - 1, int(x1)))
        y0 = max(0, min(h - 1, int(y0)))
        y1 = max(0, min(h - 1, int(y1)))
        c = np.array(color, dtype=np.uint8)
        t = max(1, int(thickness))
        out[y0 : y0 + t, x0 : x1 + 1] = c
        out[max(0, y1 - t + 1) : y1 + 1, x0 : x1 + 1] = c
        out[y0 : y1 + 1, x0 : x0 + t] = c
        out[y0 : y1 + 1, max(0, x1 - t + 1) : x1 + 1] = c
    return out


def render_annotated_data_uri(
    rgb_path,
    max_dim=512,
    color=DEFAULT_COLOR,
    alpha=DEFAULT_ALPHA,
    mask=None,
    mask_path=None,
    bbox=None,
) -> dict:
    """Composite an observation's mask/bbox onto an RGB frame and return a base64
    JPEG data URI. The mask/bbox come from the caller (the object's ``Observation``:
    ``ImageStore.mask(obs)`` / ``obs.mask_path`` and ``obs.bbox2d``); ``annotated`` is
    True iff a mask or bbox was applied."""
    img = Image.open(rgb_path).convert("RGB")
    arr = np.asarray(img)
    has_mask = mask is not None or bool(mask_path and os.path.exists(mask_path))
    bbox = tuple(bbox) if bbox is not None and len(bbox) >= 4 else None
    if has_mask or bbox:
        arr = annotate_rgb(
            arr,
            mask_path=mask_path if has_mask else None,
            mask=mask,
            bbox=bbox,
            color=color,
            alpha=alpha,
        )
    out = Image.fromarray(arr)
    if max(out.size) > max_dim:
        scale = max_dim / max(out.size)
        out = out.resize(
            (int(out.width * scale), int(out.height * scale)), Image.LANCZOS
        )
    buf = io.BytesIO()
    out.save(buf, format="JPEG")
    uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
    return {"uri": uri, "annotated": bool(has_mask or bbox)}
