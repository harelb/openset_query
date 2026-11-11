"""Typed evidence + assertion core for query-time planability assessment.

The semantic distinction this package makes concrete:

    perception:  "I detected something"
    evidence:    "I have sufficient evidence to use this proposition
                  as a planning fact."

and, just as important:

    no detection        != proposition is false
    historical support  != proposition is necessarily true now

``types``/``status``/``temporal`` are vendored from
``dcist_sim_isaac.grounding`` (dcist_sim @ 7e8bf5a — see each module's
docstring); ``assertions`` adds the persisted forms this project introduces:
:class:`Assertion` (a materialized verdict with identity, freshness, and
scope) and :class:`EvidenceObligation` (what an optimistic plan must verify).

Import contract: **stdlib only at module scope** — anything heavier (neo4j,
spark_dsg, yaml) is imported lazily inside the function that needs it, so the
semantic core runs in any venv and stays ROS-free.
"""

from .assertions import (
    Assertion,
    EvidenceObligation,
    ObligationStatus,
)
from .status import DEFAULT_COVERAGE_SUFFICIENCY, compute_status
from .store import EvidenceStore, SearchEpisode, assertion_id_for
from .temporal import FixedDecay, NoDecay, TemporalValidityModel, record_age_s
from .types import (
    DISCOVERY_SOURCE_MAP,
    EpistemicStatus,
    EvidenceRecord,
    EvidenceSource,
    GroundingQuery,
    GroundingResult,
    IdentityHypothesis,
    Polarity,
    ReasonCode,
    dumps_record,
)

__all__ = [
    "Assertion",
    "DEFAULT_COVERAGE_SUFFICIENCY",
    "DISCOVERY_SOURCE_MAP",
    "EpistemicStatus",
    "EvidenceObligation",
    "EvidenceRecord",
    "EvidenceSource",
    "EvidenceStore",
    "FixedDecay",
    "GroundingQuery",
    "GroundingResult",
    "IdentityHypothesis",
    "NoDecay",
    "ObligationStatus",
    "Polarity",
    "ReasonCode",
    "SearchEpisode",
    "TemporalValidityModel",
    "assertion_id_for",
    "compute_status",
    "dumps_record",
    "record_age_s",
]
