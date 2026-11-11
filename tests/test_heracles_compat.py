"""heracles_compat: the writes the server needs work on any heracles from master on,
and optional heracles features degrade cleanly."""

import pytest

from openset_query import heracles_compat as hc


class FakeDb:
    """Just enough of Neo4jWrapper.execute for the version counters."""

    def __init__(self, map_version=None, state_version=None):
        self.map_version = map_version
        self.state_version = state_version
        self.calls = []

    def execute(self, query, parameters=None, **kw):
        self.calls.append((query, dict(parameters or {}, **kw)))
        if query.startswith("MERGE (m:_MapState)"):
            self.map_version = (self.map_version or 0) + 1
            return [{"v": self.map_version}], None, None
        if query.startswith("MATCH (s:_State) SET"):
            if self.state_version is not None:
                self.state_version += 1
            return [], None, None
        if query.startswith("MATCH (s:_State {key: 'global'})"):
            if self.state_version is None:
                return [], None, None
            return [{"v": self.state_version}], None, None
        if query.startswith("MATCH (m:_MapState)"):
            return [{"v": self.map_version}], None, None
        return [], None, None


def _only(monkeypatch, available: dict):
    """Pretend heracles provides exactly ``{(module, name): obj}``."""
    monkeypatch.setattr(hc, "_attr", lambda module, name: available.get((module, name)))


def test_notify_falls_back_to_cypher(monkeypatch):
    _only(monkeypatch, {})
    db = FakeDb()
    assert hc.map_change_backend() == "cypher"
    assert hc.notify_map_change(db, ["2:0"], ["O1"]) == (None, 1)
    db.state_version = 4
    assert hc.notify_map_change(db) == (5, 2)


def test_notify_uses_heracles_ros_with_changes(monkeypatch):
    seen = {}

    def notify(db, changed_layers=None, changed_ids=None):
        seen.update(layers=changed_layers, ids=changed_ids)
        return 7

    _only(monkeypatch, {("heracles_ros.map_state", "notify_map_change"): notify})
    assert hc.map_change_backend() == "heracles_ros.map_state (changes)"
    assert hc.notify_map_change(FakeDb(state_version=3), ["2:0"], ["O1"]) == (3, 7)
    assert seen == {"layers": ["2:0"], "ids": ["O1"]}


def test_notify_uses_heracles_ros_without_changes(monkeypatch):
    _only(monkeypatch, {("heracles_ros.map_state", "notify_map_change"): lambda db: 2})
    assert hc.map_change_backend() == "heracles_ros.map_state"
    assert hc.notify_map_change(FakeDb(), ["2:0"], ["O1"]) == (None, 2)


def test_notify_uses_map_reload_bump(monkeypatch):
    seen = {}

    def bump(db, changed_layers=None, changed_ids=None):
        seen.update(layers=changed_layers, ids=changed_ids)
        return 11

    _only(monkeypatch, {("heracles.map_reload", "bump_map_version"): bump})
    db = FakeDb(state_version=1)
    assert hc.notify_map_change(db, ["2:0"], ["O1"]) == (2, 11)
    assert seen == {"layers": ["2:0"], "ids": ["O1"]}


def test_notify_uses_legacy_graph_interface(monkeypatch):
    _only(
        monkeypatch,
        {("heracles.graph_interface", "notify_map_change"): lambda db, dl, ai: (4, 9)},
    )
    assert hc.notify_map_change(FakeDb(), ["2:0"], ["O1"]) == (4, 9)


def test_neo4j_settings_need_a_password(monkeypatch):
    monkeypatch.delenv("HERACLES_NEO4J_PASSWORD", raising=False)
    with pytest.raises(SystemExit, match="HERACLES_NEO4J_PASSWORD"):
        hc.neo4j_settings()
    monkeypatch.setenv("HERACLES_NEO4J_PASSWORD", "x")
    monkeypatch.delenv("HERACLES_NEO4J_URI", raising=False)
    monkeypatch.delenv("HERACLES_NEO4J_USERNAME", raising=False)
    assert hc.neo4j_settings() == ("bolt://localhost:7687", ("neo4j", "x"))


def test_require_names_the_missing_function(monkeypatch):
    _only(monkeypatch, {})
    with pytest.raises(RuntimeError, match="query_similar_nodes"):
        hc.require("query_similar_nodes", "TrajectoryFrame vector search")
    assert hc.optional_features() == {name: False for name in hc.OPTIONAL}


def test_provenance_without_frame_edges(monkeypatch):
    linked = []
    _only(
        monkeypatch,
        {("heracles.graph_interface", "link_observations_to_frames"): linked.append},
    )
    hc.link_observation_provenance("db", [("tf1", "O1"), None])
    assert linked == ["db"]


def test_provenance_with_frame_edges(monkeypatch):
    edges = []
    _only(
        monkeypatch,
        {
            ("heracles.graph_interface", "insert_frame_edges"): (
                lambda db, place_pairs, obj_pairs: edges.append(obj_pairs)
            )
        },
    )
    hc.link_observation_provenance("db", [("tf1", "O1"), None])
    assert edges == [[("tf1", "O1")]]


def test_insert_observations_writes_one_merge():
    db = FakeDb()
    rows = [{"nodeSymbol": "O1_5", "timestamp_ns": 5, "frame_id": 3}]
    hc.insert_observations(db, rows)
    ((query, params),) = db.calls
    assert "MERGE (o:Observation {nodeSymbol: obs.nodeSymbol})" in query
    assert params == {"observations": rows}


def test_evidence_store_schema_is_optional(monkeypatch):
    import builtins

    from openset_query.evidence.store import EvidenceStore

    real_import = builtins.__import__

    def no_epistemic(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "heracles.graph_interface" and "create_epistemic_indexes" in (
            fromlist or ()
        ):
            raise ImportError("no create_epistemic_indexes")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", no_epistemic)
    EvidenceStore(FakeDb()).ensure_schema()  # logs, does not raise
