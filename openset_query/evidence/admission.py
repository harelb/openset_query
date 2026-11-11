"""The mandatory evidence-admission gate.

The rule this module makes structural:

    a detection is not a planning fact until something *admitted* it.

Before this gate every ``:Object`` row in Neo4j was a planning fact by virtue
of existing. A SAM3 text query that fired once, on one frame, with no score
floor and no corroboration, wrote an ``:Object {source:'sam3_query'}`` node --
and from that instant ``load_class_instances`` returned it, ``DsgSync``
mirrored it into the planner's in-memory graph, and ``RunEvidence`` certified
``exists(<class>)`` at belief 1.0 off the ungraded structural channel. The
epistemic layer was therefore reasoning about a graph whose contents had never
been graded: "the DSG says so" *was* the evidence.

The gate splits "the detector wrote it" from "the system will plan on it" with
one explicit property, ``admission_status``:

``trusted_prior``
    Ingested from a committed scene graph (the prior map). Plannable.
``candidate``
    A detector wrote it; nobody has graded it. NOT plannable.
``admitted``
    A declared authority graded it and accepted it. Plannable.
``rejected``
    A declared authority graded it and refused it. NOT plannable, even if it
    would otherwise qualify as a trusted prior.

**Fail-closed for detector writes.** :data:`PLANNABLE_OBJECT_WHERE` treats a
missing ``admission_status`` as plannable *only* for nodes the detector did not
create -- ``coalesce(n.sam3_created, false) = false``. That is a deliberate,
versioned policy -- :data:`PRIOR_MAP_POLICY_VERSION` -- so the large body of
legacy prior-map objects written before this gate existed stays usable without
a migration, while any detector-created object lacking an explicit admission is
excluded. There is no auto-admit default and there must not be one:
:func:`admit_object` / :func:`reject_object` are the only sanctioned writers of
an admission decision. The callers with standing: the admission policies
(:mod:`~openset_query.evidence.admission_policy` -- relative-margin or
geometric), a simulator's live tier (direct observation), a
catalog oracle (declared benchmark ground truth), and human curation.

**Why ``sam3_created`` and not ``source``.** ``source`` is *informational
provenance only* -- it is not a discriminator, and the gate must never read it.
A SAM3 query that dedups onto an existing object OVERWRITES its ``source`` with
``'sam3_query'`` (``object_discovery._upsert_objects_write`` /
``upsert_via_graph_writer``), and ``SAM3_PRESERVED_PROPS`` did not originally
snapshot ``source``, so a database carrying pre-gate dedup state holds prior-map
objects stamped ``source='sam3_query'`` that no detector ever created. Reading
``source`` would hide exactly those base-map objects and then permanently stamp
them ``candidate`` on the next dedup. ``sam3_created`` is written ONLY on nodes
SAM3 itself created (and is what ``/api/sam3/clear`` deletes on), so it is the
one honest "the detector wrote this node" marker.

LIVE sam3-backend runs plan on fresh detections only after the warm server's
configured admission policy grades them (``--admission-policy margin`` or
``geometric``; ``off`` is the default: every find stays a
candidate and runs end ACQUIRE/UNDETERMINED). The offline ``catalog`` oracle
is a *declared* authority (it is ground truth by construction), so it admits
its own writes under ``catalog_oracle_v1`` and the benchmark matrix is
unaffected.

Import contract: stdlib only at module scope (same as the rest of
``evidence/``); the neo4j wrapper is never imported here at all -- callers pass
a ``db`` handle.
"""
from __future__ import annotations

from typing import Iterable

# ---------------------------------------------------------------------------
# Property names. Kept in lockstep with heracles.constants (drift-guard test in
# tests/test_admission_gate_unit.py) so this module stays importable without
# heracles/spark_dsg on the path.
# ---------------------------------------------------------------------------

ADMISSION_STATUS = "admission_status"
ADMISSION_POLICY_VERSION = "admission_policy_version"
ADMISSION_OBSERVATIONS = "admission_observation_ids"
ADMISSION_REASON = "admission_reason"
#: Measured confidence the admitting policy attached: the best detector
#: score among the views that carried the admission. NEVER invented — absent
#: (None) when the admitting authority had no graded signal.
ADMISSION_BELIEF = "admission_belief"

# ---------------------------------------------------------------------------
# Values.
# ---------------------------------------------------------------------------

TRUSTED_PRIOR = "trusted_prior"
CANDIDATE = "candidate"
ADMITTED = "admitted"
REJECTED = "rejected"

#: The statuses a reader will plan on.
PLANNABLE_STATUSES = (TRUSTED_PRIOR, ADMITTED)

#: Every legal value (validation + the drift guard).
ADMISSION_VALUES = (TRUSTED_PRIOR, CANDIDATE, ADMITTED, REJECTED)

#: The policy under which an admission-less, non-detector-written object is
#: read as a trusted prior. Bump this string if that grandfathering changes.
PRIOR_MAP_POLICY_VERSION = "prior_map_legacy_v0"

#: The catalog oracle's authority version (offline benchmark ground truth).
CATALOG_ORACLE_POLICY_VERSION = "catalog_oracle_v1"

#: The marker a node carries when the detector CREATED it (as opposed to
#: deduping onto a pre-existing one). The single discriminator this gate reads
#: -- see the module docstring on why ``source`` is not usable for this.
_DETECTOR_CREATED_PROP = "sam3_created"


def plannable_where(var: str = "n") -> str:
    """The ONE Cypher predicate every planning-fact reader must apply.

    Parameterised on the node variable so it can be spliced into any ``MATCH``::

        MATCH (n:Object)
        WHERE toLower(n.class) = toLower($cls) AND {plannable_where("n")}

    Semantics (see the module docstring): plannable iff the node carries an
    explicitly plannable ``admission_status``, OR it carries none AND the
    detector did not create it (the ``prior_map_legacy_v0`` grandfathering).
    ``source`` is informational provenance and is deliberately NOT read here.

    The fragment is a self-contained boolean expression -- it is emitted
    parenthesised, so callers can ``AND`` it in without worrying about
    precedence -- and contains no query parameters, so it never collides with a
    caller's parameter names.
    """
    v = var
    plannable = ", ".join(f"'{s}'" for s in PLANNABLE_STATUSES)
    return (
        f"({v}.{ADMISSION_STATUS} IN [{plannable}] "
        f"OR ({v}.{ADMISSION_STATUS} IS NULL "
        f"AND coalesce({v}.{_DETECTOR_CREATED_PROP}, false) = false))"
    )


#: Convenience: the predicate for the conventional ``n`` node variable.
PLANNABLE_OBJECT_WHERE = plannable_where("n")


def is_plannable(props) -> bool:
    """Python mirror of :func:`plannable_where`, for an object's property dict.

    The Cypher fragment is the authority (it is what every real reader runs
    against Neo4j); this exists for in-memory checks and test fakes that never
    reach a database. Keep the two in lockstep -- the DB acceptance test
    ``tests/test_admission_gate.py`` pins the Cypher side.
    """
    status = props.get(ADMISSION_STATUS)
    if status is not None:
        return status in PLANNABLE_STATUSES
    return not bool(props.get(_DETECTOR_CREATED_PROP) or False)


def dedup_admission_case(var: str = "o") -> str:
    """Cypher expression stamping an admission on a *dedup-updated* object.

    An object a query deduped onto is a candidate iff the detector CREATED it
    (``sam3_created``); anything else the query merely re-observed is a
    pre-existing map object and stays a trusted prior. Deciding this off
    ``source`` instead would be wrong twice over: the dedup path overwrites
    ``source`` with ``'sam3_query'`` (so ordering alone would decide the
    verdict), and a database carrying pre-gate dedup state already holds
    prior-map objects stamped ``source='sam3_query'`` that would be
    permanently downgraded to candidates. ``coalesce`` means an existing
    decision is never downgraded (an already-admitted object survives
    re-detection admitted)::

        coalesce(o.admission_status,
                 CASE WHEN coalesce(o.sam3_created, false) = true
                      THEN 'candidate' ELSE 'trusted_prior' END)
    """
    v = var
    return (
        f"coalesce({v}.{ADMISSION_STATUS}, "
        f"CASE WHEN coalesce({v}.{_DETECTOR_CREATED_PROP}, false) = true "
        f"THEN '{CANDIDATE}' ELSE '{TRUSTED_PRIOR}' END)"
    )


# ---------------------------------------------------------------------------
# The only sanctioned admission writers.
# ---------------------------------------------------------------------------


def _set_admission(db, node_symbol: str, status: str, *, policy_version: str,
                   observation_ids: Iterable[str] = (), reason=None,
                   belief=None) -> None:
    if status not in ADMISSION_VALUES:
        raise ValueError(f"unknown admission status {status!r}")
    if not policy_version:
        raise ValueError("policy_version is required: an admission is only "
                         "meaningful relative to the policy that made it")
    sets = [
        f"o.{ADMISSION_STATUS} = $status",
        f"o.{ADMISSION_POLICY_VERSION} = $policy",
        f"o.{ADMISSION_OBSERVATIONS} = $obs",
    ]
    params = {
        "sym": node_symbol,
        "status": status,
        "policy": policy_version,
        "obs": [str(o) for o in observation_ids],
    }
    if reason is not None:
        sets.append(f"o.{ADMISSION_REASON} = $reason")
        params["reason"] = str(reason)
    if belief is not None:
        sets.append(f"o.{ADMISSION_BELIEF} = $belief")
        params["belief"] = float(belief)
    db.execute(
        "MATCH (o:Object {nodeSymbol: $sym}) SET " + ", ".join(sets), **params)


def admit_object(db, node_symbol: str, *, policy_version: str,
                 observation_ids: Iterable[str] = (), belief=None) -> None:
    """Record that *node_symbol* passed admission under *policy_version*.

    Idempotent (a plain property SET). ``observation_ids`` is the evidence the
    policy actually weighed -- stored so an admission is auditable and can be
    invalidated when the policy version changes. ``belief`` is the policy's
    measured confidence (see :data:`ADMISSION_BELIEF`); omit it when the
    admitting authority has no graded signal.
    """
    _set_admission(db, node_symbol, ADMITTED, policy_version=policy_version,
                   observation_ids=observation_ids, belief=belief)


def reject_object(db, node_symbol: str, *, policy_version: str,
                  reason: str) -> None:
    """Record that *node_symbol* FAILED admission under *policy_version*.

    An explicit rejection beats the ``prior_map_legacy_v0`` grandfathering: a
    rejected object is unplannable even with no detector markers. Idempotent.
    """
    _set_admission(db, node_symbol, REJECTED, policy_version=policy_version,
                   reason=reason)


def mark_trusted_prior(db, node_symbol: str, *,
                       policy_version: str = PRIOR_MAP_POLICY_VERSION) -> None:
    """Stamp an ingested prior-map object as a trusted prior (explicit form of
    the grandfathering, used by the spark_dsg -> DB ingest)."""
    _set_admission(db, node_symbol, TRUSTED_PRIOR, policy_version=policy_version)


def admission_status(db, node_symbol: str):
    """Current ``admission_status`` of *node_symbol* (``None`` if unset/absent)."""
    records, _, _ = db.execute(
        f"MATCH (o:Object {{nodeSymbol: $sym}}) RETURN o.{ADMISSION_STATUS} AS s",
        sym=node_symbol)
    return records[0]["s"] if records else None
