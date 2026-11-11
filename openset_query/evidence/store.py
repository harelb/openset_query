"""Neo4j evidence store: SearchEpisodes, materialized Assertions, JSONL sidecar.

DB-first (the graph is the source of truth for evidence; see the project's
DB-is-source-of-truth discipline), with one deliberate caveat: every heracles
ingest (``initialize_db``) DETACH-DELETEs the whole database, so ``(:Assertion)``
nodes are *recomputable materializations* of :func:`compute_status` over the
evidence — never the only copy. :meth:`EvidenceStore.export_jsonl` /
:meth:`EvidenceStore.import_jsonl` are the wipe-survival sidecar for the
epistemic layer (SearchEpisodes + Assertions); Observations are re-created by
the normal ingest pipeline.

Graph shape written/read here::

    (:SearchEpisode {episode_id, predicate, args_json, t_start_ns, t_end_ns,
                     coverage, mechanism, session_id, source, provenance_json})
        -[:COVERED]-> (frame: Agent|SubKeyframe|TrajectoryFrame)
    (:Assertion {assertion_id, predicate, args_json, status, reason, belief,
                 computed_at_ns, required_confidence, freshness_s,
                 policy_version, session_id, json})
        <-[:SUPPORTS]-    (:Observation) | (:SearchEpisode)
        <-[:CONTRADICTS]- (:Observation) | (:SearchEpisode)

The store never invents evidence: positives come from ``(:Object)-
[:HAS_OBSERVATION]->(:Observation)`` rows written by the SAM3/khronos upsert
paths (score/detector/mechanism provenance), negatives from
SearchEpisodes recorded by search/re-query mechanisms. ``db`` is any object
with the ``Neo4jWrapper.execute`` signature (returns ``records, summary, keys``).

Stdlib-only at module scope (package import contract); heracles is imported
lazily inside :meth:`EvidenceStore.ensure_schema`.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field

from .assertions import Assertion, _new_id
from .status import DEFAULT_COVERAGE_SUFFICIENCY, compute_status
from .types import (
    EvidenceRecord,
    EvidenceSource,
    GroundingQuery,
    Polarity,
    dumps_record,
)

logger = logging.getLogger("openset_query.evidence.store")

# Node/edge names, kept in lockstep with heracles.constants (drift-guard test
# in tests/test_evidence_store.py) so this module stays importable without
# heracles/spark_dsg on the path.
OBJECTS = "Object"
OBSERVATIONS = "Observation"
SEARCH_EPISODES = "SearchEpisode"
ASSERTIONS = "Assertion"
HAS_OBSERVATION = "HAS_OBSERVATION"
COVERED = "COVERED"
SUPPORTS = "SUPPORTS"
CONTRADICTS = "CONTRADICTS"

# Frame-bearing labels a SearchEpisode can COVER (mirrors the provenance
# helpers' _FRAME_LABELS in heracles.graph_interface).
FRAME_LABELS = ("Agent", "SubKeyframe", "TrajectoryFrame")


def assertion_id_for(query: GroundingQuery) -> str:
    """Deterministic Assertion identity for a query.

    Recomputing the same query (same predicate/args/thresholds) MERGEs onto the
    same node — the materialization updates in place — and a wiped-then-rebuilt
    database reproduces byte-identical assertion ids.
    """
    key = json.dumps(query.to_json(), sort_keys=True)
    return "assert_" + hashlib.sha256(key.encode()).hexdigest()[:12]


@dataclass(frozen=True)
class SearchEpisode:
    """One completed search for a proposition — the persistable form of
    negative evidence (module docstring rule 3 in ``types``).

    ``coverage`` is the search's completeness in [0, 1] (``None`` = unknown,
    which can never contradict); ``mechanism`` names the VerificationMechanism
    that ran (``reprocess_memory`` / ``acquire_view`` / ``frontier_search`` /
    ...); ``source`` is an :class:`EvidenceSource` value string.
    """

    episode_id: str
    predicate: str
    args: tuple
    t_start_ns: int
    t_end_ns: int
    coverage: float | None = None
    mechanism: str | None = None
    session_id: str | None = None
    source: str = EvidenceSource.CURRENT_ARCHIVE.value
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "args", tuple(self.args))

    @classmethod
    def create(
        cls, predicate: str, args, t_start_ns: int, t_end_ns: int, **kwargs
    ) -> "SearchEpisode":
        return cls(
            episode_id=_new_id("episode"),
            predicate=predicate,
            args=tuple(args),
            t_start_ns=int(t_start_ns),
            t_end_ns=int(t_end_ns),
            **kwargs,
        )

    @property
    def args_json(self) -> str:
        return json.dumps(list(self.args))

    def to_record(self) -> EvidenceRecord:
        """The episode as a NEGATIVE EvidenceRecord for ``compute_status``.

        ``requery_enabled`` (channel-alive flag) passes through ``provenance``
        exactly as the producer recorded it; absence means alive.
        """
        provenance = dict(self.provenance)
        if self.mechanism is not None:
            provenance.setdefault("mechanism", self.mechanism)
        provenance.setdefault("t_start_ns", self.t_start_ns)
        return EvidenceRecord(
            evidence_id=self.episode_id,
            source=EvidenceSource(self.source),
            polarity=Polarity.NEGATIVE,
            predicate=self.predicate,
            args=self.args,
            t_recorded_ns=self.t_end_ns,
            t_collected_ns=self.t_end_ns,
            session_id=self.session_id,
            coverage=self.coverage,
            provenance=provenance,
        )

    def to_json(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "predicate": self.predicate,
            "args": list(self.args),
            "t_start_ns": self.t_start_ns,
            "t_end_ns": self.t_end_ns,
            "coverage": self.coverage,
            "mechanism": self.mechanism,
            "session_id": self.session_id,
            "source": self.source,
            "provenance": self.provenance,
        }

    @classmethod
    def from_json(cls, d: dict) -> "SearchEpisode":
        return cls(
            episode_id=str(d["episode_id"]),
            predicate=str(d["predicate"]),
            args=tuple(d.get("args", ())),
            t_start_ns=int(d["t_start_ns"]),
            t_end_ns=int(d["t_end_ns"]),
            coverage=d.get("coverage"),
            mechanism=d.get("mechanism"),
            session_id=d.get("session_id"),
            source=d.get("source", EvidenceSource.CURRENT_ARCHIVE.value),
            provenance=dict(d.get("provenance", {})),
        )


class EvidenceStore:
    """Reads evidence from / writes the epistemic layer to a Neo4j scene graph."""

    def __init__(
        self, db, *, session_id: str | None = None, policy_version: str | None = None
    ):
        self.db = db
        self.session_id = session_id
        self.policy_version = policy_version

    # -- schema ------------------------------------------------------------

    def ensure_schema(self) -> None:
        """Create the epistemic constraints/indexes (idempotent; heracles owns
        the schema — lazy import keeps this module stdlib-only). Skipped, with a
        log line, when heracles has no ``create_epistemic_indexes``: the indexes
        only speed up the queries."""
        try:
            from heracles.graph_interface import create_epistemic_indexes
        except ImportError:
            logger.info(
                "heracles has no create_epistemic_indexes: evidence store indexes off"
            )
            return
        create_epistemic_indexes(self.db)

    # -- search episodes ---------------------------------------------------

    def add_episode(
        self, episode: SearchEpisode, covered_frames=None, covered_by_time: bool = False
    ) -> int:
        """Upsert one SearchEpisode; fan out ``COVERED`` edges to frames.

        ``covered_frames`` is an iterable of frame nodeSymbols (any of the
        three frame labels). ``covered_by_time=True`` instead covers every
        frame with ``t_start_ns <= timestamp_ns <= t_end_ns``. Frames that do
        not exist in the graph are skipped (best-effort, like the provenance
        helpers). Returns the number of COVERED edges present for the episode.
        """
        self.db.execute(
            f"""
        MERGE (e:{SEARCH_EPISODES} {{episode_id: $eid}})
        SET e.predicate = $predicate, e.args_json = $args_json,
            e.t_start_ns = $t0, e.t_end_ns = $t1,
            e.coverage = $coverage, e.mechanism = $mechanism,
            e.session_id = $session_id, e.source = $source,
            e.provenance_json = $provenance_json
        """,
            eid=episode.episode_id,
            predicate=episode.predicate,
            args_json=episode.args_json,
            t0=episode.t_start_ns,
            t1=episode.t_end_ns,
            coverage=episode.coverage,
            mechanism=episode.mechanism,
            session_id=episode.session_id
            if episode.session_id is not None
            else self.session_id,
            source=episode.source,
            provenance_json=json.dumps(episode.provenance, sort_keys=True),
        )
        total = 0
        for label in FRAME_LABELS:
            if covered_by_time:
                clause = "WHERE f.timestamp_ns >= $t0 AND f.timestamp_ns <= $t1"
                params = {"t0": episode.t_start_ns, "t1": episode.t_end_ns}
            elif covered_frames:
                clause = "WHERE f.nodeSymbol IN $syms"
                params = {"syms": list(covered_frames)}
            else:
                continue
            records, _, _ = self.db.execute(
                f"""
            MATCH (e:{SEARCH_EPISODES} {{episode_id: $eid}})
            MATCH (f:{label}) {clause}
            MERGE (e)-[:{COVERED}]->(f)
            RETURN count(f) AS n
            """,
                eid=episode.episode_id,
                **params,
            )
            total += records[0]["n"] if records else 0
        return total

    def episodes_for(self, query: GroundingQuery) -> list:
        """All SearchEpisodes recorded for this predicate + args (episode-id
        order, deterministic)."""
        args_json = json.dumps(list(query.args))
        records, _, _ = self.db.execute(
            f"""
        MATCH (e:{SEARCH_EPISODES} {{predicate: $predicate, args_json: $args_json}})
        RETURN e.episode_id AS episode_id, e.t_start_ns AS t0, e.t_end_ns AS t1,
               e.coverage AS coverage, e.mechanism AS mechanism,
               e.session_id AS session_id, e.source AS source,
               e.provenance_json AS provenance_json
        ORDER BY e.episode_id
        """,
            predicate=query.predicate,
            args_json=args_json,
        )
        episodes = []
        for r in records:
            episodes.append(
                SearchEpisode(
                    episode_id=r["episode_id"],
                    predicate=query.predicate,
                    args=query.args,
                    t_start_ns=int(r["t0"]),
                    t_end_ns=int(r["t1"]),
                    coverage=r["coverage"],
                    mechanism=r["mechanism"],
                    session_id=r["session_id"],
                    source=r["source"] or EvidenceSource.CURRENT_ARCHIVE.value,
                    provenance=json.loads(r["provenance_json"] or "{}"),
                )
            )
        return episodes

    # -- evidence gathering --------------------------------------------------

    def observation_records(self, query: GroundingQuery) -> list:
        """Positive EvidenceRecords from Observations of Objects whose
        ``class``/``name`` equals the query's first argument.

        Covers class-membership propositions (``exists(cls)`` and friends);
        richer predicates supply their own records via ``extra_records``.
        Observations without a persisted score are structural (``confidence
        None`` -> effective 1.0), matching the pre-provenance khronos rows.
        Deterministic order: (timestamp, nodeSymbol).
        """
        if not query.args:
            return []
        cls_name = str(query.args[0])
        records, _, _ = self.db.execute(
            f"""
        MATCH (o:{OBJECTS})-[:{HAS_OBSERVATION}]->(obs:{OBSERVATIONS})
        WHERE o.class = $cls OR o.name = $cls
        RETURN obs.nodeSymbol AS ns, obs.timestamp_ns AS ts, obs.score AS score,
               obs.detector AS detector, obs.mechanism AS mechanism,
               o.nodeSymbol AS obj
        ORDER BY obs.timestamp_ns, obs.nodeSymbol
        """,
            cls=cls_name,
        )
        out = []
        for r in records:
            provenance = {"object_symbol": r["obj"]}
            if r["detector"] is not None:
                provenance["detector"] = r["detector"]
            if r["mechanism"] is not None:
                provenance["mechanism"] = r["mechanism"]
            out.append(
                EvidenceRecord(
                    evidence_id=r["ns"],
                    source=EvidenceSource.COMMITTED_DSG,
                    polarity=Polarity.POSITIVE,
                    predicate=query.predicate,
                    args=query.args,
                    t_recorded_ns=int(r["ts"] or 0),
                    t_collected_ns=int(r["ts"]) if r["ts"] is not None else None,
                    confidence=r["score"],
                    session_id=self.session_id,
                    provenance=provenance,
                )
            )
        return out

    def gather_records(self, query: GroundingQuery, extra_records=()) -> list:
        return (
            self.observation_records(query)
            + [e.to_record() for e in self.episodes_for(query)]
            + list(extra_records)
        )

    # -- assertions ----------------------------------------------------------

    def compute_assertion(
        self,
        query: GroundingQuery,
        *,
        now_ns: int,
        temporal=None,
        coverage_sufficiency=DEFAULT_COVERAGE_SUFFICIENCY,
        channel_capable: bool = True,
        extra_records=(),
        scope: dict | None = None,
        persist: bool = True,
    ) -> Assertion:
        """Gather evidence from the graph, run ``compute_status``, materialize.

        The Assertion node is a cache: same query + same evidence reproduces
        the same id and status (JSONL re-import + recompute acceptance).
        """
        records = self.gather_records(query, extra_records)
        result = compute_status(
            query,
            records,
            temporal=temporal,
            now_ns=now_ns,
            coverage_sufficiency=coverage_sufficiency,
            channel_capable=channel_capable,
        )
        assertion = Assertion.from_result(
            result,
            computed_at_ns=now_ns,
            assertion_id=assertion_id_for(query),
            policy_version=self.policy_version,
            scope=scope,
        )
        if persist:
            self.persist_assertion(assertion)
        return assertion

    def persist_assertion(self, assertion: Assertion) -> None:
        """Upsert the Assertion node and refresh its evidence edges.

        Stale SUPPORTS/CONTRADICTS edges from a previous recompute are dropped
        first — the edges always mirror the persisted verdict exactly. Edge
        direction: evidence -> assertion. Evidence nodes are matched by
        Observation.nodeSymbol or SearchEpisode.episode_id; ids with no node
        (e.g. extra_records) simply get no edge.
        """
        self.db.execute(
            f"""
        MERGE (a:{ASSERTIONS} {{assertion_id: $aid}})
        SET a.predicate = $predicate, a.args_json = $args_json,
            a.status = $status, a.reason = $reason, a.belief = $belief,
            a.computed_at_ns = $computed_at_ns,
            a.required_confidence = $required_confidence,
            a.freshness_s = $freshness_s,
            a.policy_version = $policy_version, a.session_id = $session_id,
            a.json = $json
        """,
            aid=assertion.assertion_id,
            predicate=assertion.query.predicate,
            args_json=json.dumps(list(assertion.query.args)),
            status=assertion.status.value,
            reason=assertion.reason.value,
            belief=assertion.belief,
            computed_at_ns=assertion.computed_at_ns,
            required_confidence=assertion.query.required_confidence,
            freshness_s=assertion.freshness_s,
            policy_version=assertion.policy_version,
            session_id=self.session_id,
            json=dumps_record(assertion),
        )
        self.db.execute(
            f"""
        MATCH (a:{ASSERTIONS} {{assertion_id: $aid}})<-[r:{SUPPORTS}|{CONTRADICTS}]-()
        DELETE r
        """,
            aid=assertion.assertion_id,
        )
        for rel, ids in (
            (SUPPORTS, assertion.supporting),
            (CONTRADICTS, assertion.contradicting),
        ):
            if not ids:
                continue
            self.db.execute(
                f"""
            MATCH (a:{ASSERTIONS} {{assertion_id: $aid}})
            MATCH (ev:{OBSERVATIONS}) WHERE ev.nodeSymbol IN $ids
            MERGE (ev)-[:{rel}]->(a)
            """,
                aid=assertion.assertion_id,
                ids=list(ids),
            )
            self.db.execute(
                f"""
            MATCH (a:{ASSERTIONS} {{assertion_id: $aid}})
            MATCH (ev:{SEARCH_EPISODES}) WHERE ev.episode_id IN $ids
            MERGE (ev)-[:{rel}]->(a)
            """,
                aid=assertion.assertion_id,
                ids=list(ids),
            )

    def load_assertions(self, predicate: str | None = None) -> list:
        """Assertions materialized in the graph (id order, deterministic)."""
        clause = "WHERE a.predicate = $predicate" if predicate else ""
        records, _, _ = self.db.execute(
            f"MATCH (a:{ASSERTIONS}) {clause} "
            "RETURN a.json AS json ORDER BY a.assertion_id",
            **({"predicate": predicate} if predicate else {}),
        )
        return [Assertion.from_json(json.loads(r["json"])) for r in records]

    # -- JSONL sidecar (wipe survival) ----------------------------------------

    def export_jsonl(self, path) -> int:
        """Write every SearchEpisode (+ its covered frame symbols) and every
        Assertion to one JSONL file. Returns the number of lines written."""
        lines = []
        records, _, _ = self.db.execute(
            f"""
        MATCH (e:{SEARCH_EPISODES})
        OPTIONAL MATCH (e)-[:{COVERED}]->(f)
        WITH e, [s IN collect(f.nodeSymbol) WHERE s IS NOT NULL] AS covered
        RETURN e.episode_id AS episode_id, e.predicate AS predicate,
               e.args_json AS args_json, e.t_start_ns AS t0, e.t_end_ns AS t1,
               e.coverage AS coverage, e.mechanism AS mechanism,
               e.session_id AS session_id, e.source AS source,
               e.provenance_json AS provenance_json, covered
        ORDER BY e.episode_id
        """
        )
        for r in records:
            episode = SearchEpisode(
                episode_id=r["episode_id"],
                predicate=r["predicate"],
                args=tuple(json.loads(r["args_json"] or "[]")),
                t_start_ns=int(r["t0"]),
                t_end_ns=int(r["t1"]),
                coverage=r["coverage"],
                mechanism=r["mechanism"],
                session_id=r["session_id"],
                source=r["source"] or EvidenceSource.CURRENT_ARCHIVE.value,
                provenance=json.loads(r["provenance_json"] or "{}"),
            )
            lines.append(
                json.dumps(
                    {
                        "kind": "search_episode",
                        "episode": episode.to_json(),
                        "covered": sorted(r["covered"]),
                    },
                    sort_keys=True,
                )
            )
        for assertion in self.load_assertions():
            lines.append(
                json.dumps(
                    {"kind": "assertion", "assertion": assertion.to_json()},
                    sort_keys=True,
                )
            )
        with open(path, "w") as f:
            for line in lines:
                f.write(line + "\n")
        return len(lines)

    def import_jsonl(self, path) -> dict:
        """Re-materialize a sidecar into the graph (MERGE — idempotent, safe
        on a freshly re-ingested database). COVERED edges re-attach only to
        frames that exist again; SUPPORTS/CONTRADICTS likewise."""
        counts = {"search_episode": 0, "assertion": 0}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                kind = d.get("kind")
                if kind == "search_episode":
                    episode = SearchEpisode.from_json(d["episode"])
                    self.add_episode(episode, covered_frames=d.get("covered") or None)
                    counts["search_episode"] += 1
                elif kind == "assertion":
                    self.persist_assertion(Assertion.from_json(d["assertion"]))
                    counts["assertion"] += 1
        return counts
