"""Durable admission curation: a per-map sidecar re-applied at every ingest.

The Neo4j map is rebuilt from the on-disk DSG at every viz start (and via
``/api/admin/reload-graph``), which wipes ``admission_status`` stamps — a
curated trusted prior silently reverts to ``candidate`` on the next respawn.
This module makes curation survive: ``/api/graph/admission`` records each
decision here as a CLASS-level rule (node symbols do not survive a re-ingest;
an object's class string does), and the ingest paths call :func:`apply_rules`
right after ``spark_dsg_to_db``.

Sidecar: ``curated_priors.json`` next to the map's ``dsg.json`` — a JSON list
of ``{"class": <exact class, case-insensitive>, "status": ..., "policy": ...,
"reason": ...?}``, newest decision per class wins.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Iterable

from openset_query.evidence import admission as adm

logger = logging.getLogger(__name__)

SIDECAR_NAME = "curated_priors.json"


def sidecar_path(dsg_path: str | Path) -> Path:
    """The curation sidecar for the map whose DSG lives at ``dsg_path``."""
    return Path(dsg_path).parent / SIDECAR_NAME


def load_rules(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    try:
        rules = json.loads(path.read_text())
    except Exception:
        logger.exception("unreadable curation sidecar %s (ignored)", path)
        return []
    if not isinstance(rules, list):
        logger.warning("curation sidecar %s is not a list (ignored)", path)
        return []
    return [
        r
        for r in rules
        if isinstance(r, dict)
        and r.get("class")
        and r.get("status") in adm.ADMISSION_VALUES
    ]


def record_rules(path: str | Path, new_rules: Iterable[dict]) -> int:
    """Merge ``new_rules`` into the sidecar (per-class, newest wins; atomic
    write). Returns the total rule count after the merge."""
    path = Path(path)
    merged: dict[str, dict] = {str(r["class"]).lower(): r for r in load_rules(path)}
    for r in new_rules:
        cls, status = r.get("class"), r.get("status")
        if not cls or status not in adm.ADMISSION_VALUES:
            continue
        rule = {
            "class": str(cls),
            "status": status,
            "policy": str(r.get("policy") or "curated_prior_v1"),
        }
        if r.get("reason"):
            rule["reason"] = str(r["reason"])
        merged[str(cls).lower()] = rule
    rules = list(merged.values())
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with open(fd, "w") as f:
            json.dump(rules, f, indent=1)
        Path(tmp).replace(path)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise
    return len(rules)


def apply_rules(db, path: str | Path) -> int:
    """Stamp every Object matching a sidecar rule (exact class,
    case-insensitive) through the sanctioned admission writer. Returns the
    number of objects stamped. Never raises — a broken sidecar must not cost
    the caller its ingest."""
    n = 0
    try:
        for rule in load_rules(path):
            records, _, _ = db.execute(
                "MATCH (o:Object) WHERE toLower(o.class) = toLower($cls) "
                "RETURN o.nodeSymbol AS sym",
                cls=rule["class"],
            )
            for rec in records:
                adm._set_admission(
                    db,
                    rec["sym"],
                    rule["status"],
                    policy_version=rule["policy"],
                    reason=rule.get("reason"),
                )
                n += 1
        if n:
            logger.info("curation sidecar %s: stamped %d object(s)", path, n)
    except Exception:
        logger.exception("curation sidecar apply failed (ingest unaffected)")
    return n
