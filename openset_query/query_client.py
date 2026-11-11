"""Thin client for the warm SAM3 query server (stdlib only: no torch/SAM3 import)."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


class Sam3Unavailable(RuntimeError):
    """The SAM3 query server could not be reached."""


class Sam3QueryError(RuntimeError):
    """The SAM3 query server was reached but returned an error."""

    def __init__(self, message, status=None):
        super().__init__(message)
        #: HTTP status when the server answered with an error code (404: the
        #: pending_id is unknown or expired)
        self.status = status


def sam3_base_url() -> str:
    return os.getenv("SAM3_QUERY_SERVER_URL", "http://localhost:8077").rstrip("/")


#: The serving query budgets every live client should use. Measured on an indoor
#: tour map (pool 4499, conf 0.25): batched/top_k=12/max_frames=12/patience=1 ran SAM3 on 12 frames
#: and reported "no matches" for objects plainly in the map; frustum with these
#: values processed 45 frames and found them. Offline benchmark strategies keep
#: their own historical budgets on purpose (cost-model continuity); a live
#: surface that narrows these will silently miss in-archive objects and corrupt
#: the epistemics (miss -> bogus escalation to acquire_view).
SERVING_QUERY_MODE = "frustum"
SERVING_QUERY_TOP_K = 60
SERVING_QUERY_MAX_FRAMES = 60
SERVING_QUERY_PATIENCE = 25


def _post(path: str, payload: dict, timeout_s: float, base_url=None) -> dict:
    base = (base_url or sam3_base_url()).rstrip("/")
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        base + path,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            result = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", "replace")[:300]
        except Exception:
            pass
        raise Sam3QueryError(
            f"SAM3 server HTTP {e.code}: {detail or e.reason}", status=e.code
        )
    except urllib.error.URLError as e:
        raise Sam3Unavailable(f"SAM3 query server unreachable at {base} ({e.reason})")
    if isinstance(result, dict) and result.get("error"):
        raise Sam3QueryError(f"SAM3 server error: {result['error']}")
    return result


def sam3_query(
    prompt,
    mode=SERVING_QUERY_MODE,
    top_k=SERVING_QUERY_TOP_K,
    max_frames=SERVING_QUERY_MAX_FRAMES,
    patience=SERVING_QUERY_PATIENCE,
    timeout_s=120.0,
    base_url=None,
    min_ts=None,
    dry_run=False,
) -> dict:
    """Defaults are sent EXPLICITLY in the payload, so they -- not the server's --
    are what every caller gets. The previous batched/12/12/1 searched 12 frames and
    reported "no matches" for objects plainly present; patience=1 in particular
    aborts after the first frame that adds nothing. Measured on an indoor
    tour map (pool 4499, conf 0.25): "a vending machine" 12 frames -> 0, 45 frames -> 2.

    ``dry_run=True`` writes nothing: the reply carries ``pending_id`` and
    ``candidates`` (label, score, centroid, bbox, best views with mask RLE and an
    annotated JPEG data URI); confirm one with :func:`sam3_commit`, drop the rest
    with :func:`sam3_discard` (or let them expire, 10 min).
    """
    payload = {
        "prompt": prompt,
        "mode": mode,
        "top_k": top_k,
        "max_frames": max_frames,
        "patience": patience,
    }
    if min_ts is not None:
        payload["min_ts"] = int(min_ts)  # frames at/after this ns timestamp
    if dry_run:
        payload["dry_run"] = True
    return _post("/query", payload, timeout_s, base_url)


def sam3_commit(pending_id: str, index: int, timeout_s=60.0, base_url=None) -> dict:
    """Write candidate ``index`` of a dry run to the image store + Neo4j (admitted,
    both map version counters bumped). Returns ``{"symbol", "created", "merged",
    ...}``; an expired ``pending_id`` raises :class:`Sam3QueryError` with
    ``status == 404``."""
    return _post(
        "/commit", {"pending_id": pending_id, "index": int(index)}, timeout_s, base_url
    )


def sam3_discard(pending_id: str, timeout_s=10.0, base_url=None) -> dict:
    """Drop a dry run's cached candidates. Returns ``{"discarded": bool}``."""
    return _post("/discard", {"pending_id": pending_id}, timeout_s, base_url)
