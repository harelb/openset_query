"""evidence.curation_store — the per-map admission-curation sidecar."""

import json

from openset_query.evidence import curation_store as cs


class FakeDb:
    """Captures execute() calls; serves symbol lookups from a class->syms map."""

    def __init__(self, by_class):
        self.by_class = {k.lower(): v for k, v in by_class.items()}
        self.writes = []

    def execute(self, query, **params):
        if query.startswith("MATCH (o:Object) WHERE toLower(o.class)"):
            syms = self.by_class.get(str(params["cls"]).lower(), [])
            return [{"sym": s} for s in syms], None, None
        self.writes.append((query, params))
        return [], None, None


def test_sidecar_path_is_next_to_dsg(tmp_path):
    assert (
        cs.sidecar_path(tmp_path / "hydra" / "dsg.json")
        == tmp_path / "hydra" / "curated_priors.json"
    )


def test_record_merge_newest_wins_and_roundtrip(tmp_path):
    p = tmp_path / "curated_priors.json"
    n = cs.record_rules(
        p,
        [
            {"class": "mug", "status": "trusted_prior", "policy": "curated_prior_v1"},
            {"class": "Coffee Maker", "status": "trusted_prior"},
        ],
    )
    assert n == 2
    # newest decision per class wins (case-insensitive key)
    n = cs.record_rules(
        p, [{"class": "MUG", "status": "rejected", "reason": "phantom"}]
    )
    assert n == 2
    rules = {r["class"].lower(): r for r in cs.load_rules(p)}
    assert rules["mug"]["status"] == "rejected"
    assert rules["mug"]["reason"] == "phantom"
    assert rules["coffee maker"]["policy"] == "curated_prior_v1"


def test_load_rules_drops_garbage(tmp_path):
    p = tmp_path / "curated_priors.json"
    p.write_text(
        json.dumps(
            [
                {"class": "mug", "status": "trusted_prior", "policy": "p"},
                {"class": "bad-status", "status": "very_trusted"},
                {"status": "trusted_prior"},  # no class
                "not a dict",
            ]
        )
    )
    assert [r["class"] for r in cs.load_rules(p)] == ["mug"]
    (tmp_path / "broken.json").write_text("{nope")
    assert cs.load_rules(tmp_path / "broken.json") == []
    assert cs.load_rules(tmp_path / "absent.json") == []


def test_apply_rules_stamps_matching_objects(tmp_path):
    p = tmp_path / "curated_priors.json"
    cs.record_rules(
        p,
        [
            {"class": "mug", "status": "trusted_prior", "policy": "curated_prior_v1"},
            {"class": "kayak", "status": "rejected", "reason": "phantom"},
        ],
    )
    db = FakeDb({"mug": ["O14", "O77"], "kayak": []})
    assert cs.apply_rules(db, p) == 2
    stamped = {w[1]["sym"]: w[1] for w in db.writes}
    assert set(stamped) == {"O14", "O77"}
    assert stamped["O14"]["status"] == "trusted_prior"
    assert stamped["O14"]["policy"] == "curated_prior_v1"


def test_apply_rules_never_raises(tmp_path):
    class ExplodingDb:
        def execute(self, *a, **k):
            raise RuntimeError("db down")

    p = tmp_path / "curated_priors.json"
    cs.record_rules(p, [{"class": "mug", "status": "trusted_prior"}])
    assert cs.apply_rules(ExplodingDb(), p) == 0
