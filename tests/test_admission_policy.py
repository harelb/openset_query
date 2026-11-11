"""Pure (no-DB) tests for the admission policies (evidence/admission_policy).

Covers the behaviour that settles without Neo4j:

- a high raw SAM3 score with a failed relative-margin gate is rejected;
- a zero-hit query produces no admission (stays UNKNOWN);
- every admitted decision is traceable to observation IDs and raw scores;
- the margin / geometric / off policy switch;
- drift guards keeping the persisted evidence props, the read-back query, and
  the writer in lockstep.

The Neo4j round-trips (planner-side grounding of admitted/rejected objects)
live with the planner, not in this repository.
"""

import json

import pytest

from openset_query.evidence.admission import (
    ADMISSION_BELIEF,
    ADMITTED,
    CANDIDATE,
    REJECTED,
    TRUSTED_PRIOR,
)
from openset_query.evidence.admission_policy import (
    AdmissionCalibration,
    GeometricPolicy,
    RelativeMarginPolicy,
    UNKNOWN,
    grade_object,
    load_calibration,
    load_observation_evidence,
    make_admission_policy,
)
from openset_query.evidence.margins import ObservationEvidence


def _cal(**over):
    d = dict(
        version="test_cal_v1",
        control_prompts=("a wall", "a floor"),
        margin_admit=0.05,
        margin_reject=-0.02,
        min_views=2,
        min_sam3_score=0.25,
        min_depth_valid_px=50,
        max_view_spread_m=1.0,
    )
    d.update(over)
    return AdmissionCalibration(**d)


def _ev(
    obs_id,
    ts,
    *,
    sam3=0.9,
    target=0.30,
    wall=0.10,
    depth=500,
    xyz=(1.0, 2.0, 0.0),
    prompt="a mug",
):
    return ObservationEvidence(
        obs_id=obs_id,
        timestamp_ns=ts,
        sam3_score=sam3,
        target_score=target,
        control_scores={"a wall": wall},
        depth_valid_px=depth,
        view_xyz=xyz,
        target_prompt=prompt,
    )


class FakeDb:
    def __init__(self, rows=()):
        self.rows = list(rows)
        self.calls = []

    def execute(self, query, params=None, **kwargs):
        self.calls.append((query, dict(params or {}, **kwargs)))
        return self.rows, None, None


# ---------------------------------------------------------------------------
# Calibration artifact
# ---------------------------------------------------------------------------


def test_calibration_round_trips_through_json(tmp_path):
    cal = _cal()
    path = tmp_path / "cal.json"
    path.write_text(json.dumps(cal.to_json()))
    assert load_calibration(str(path)) == cal


def test_calibration_missing_keys_fail_loud(tmp_path):
    path = tmp_path / "cal.json"
    path.write_text(json.dumps({"version": "v1"}))
    with pytest.raises(ValueError, match="missing keys"):
        load_calibration(str(path))


def test_calibration_validates_threshold_order_and_controls():
    with pytest.raises(ValueError, match="margin_reject"):
        _cal(margin_reject=0.2)  # above margin_admit
    with pytest.raises(ValueError, match="control_prompts"):
        _cal(control_prompts=())


def test_missing_artifact_file_fails_loud(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_calibration(str(tmp_path / "nope.json"))


# ---------------------------------------------------------------------------
# The policy switch
# ---------------------------------------------------------------------------


def test_make_admission_policy_is_the_policy_switch():
    cal = _cal()
    assert isinstance(make_admission_policy("margin", cal), RelativeMarginPolicy)
    assert isinstance(make_admission_policy("geometric", cal), GeometricPolicy)
    assert make_admission_policy("off", cal) is None
    assert make_admission_policy("off", None) is None
    with pytest.raises(ValueError, match="unknown admission policy"):
        make_admission_policy("vibes", cal)
    with pytest.raises(ValueError, match="requires a calibration"):
        make_admission_policy("margin", None)


def test_policy_versions_name_the_calibration_and_the_arm():
    cal = _cal()
    assert RelativeMarginPolicy(cal).policy_version == "test_cal_v1:margin"
    assert GeometricPolicy(cal).policy_version == "test_cal_v1:geometric"


# ---------------------------------------------------------------------------
# RelativeMarginPolicy semantics
# ---------------------------------------------------------------------------


def test_margin_policy_admits_multiview_separated_evidence():
    pol = RelativeMarginPolicy(_cal())
    d = pol.grade(
        [_ev("o1", 1, sam3=0.8), _ev("o2", 2, sam3=0.95, xyz=(1.2, 2.0, 0.0))]
    )
    assert d.status == ADMITTED
    # Traceability (spec test 6): the decision names the views it weighed and
    # its belief is the best MEASURED detector score among them.
    assert d.observation_ids == ("o1", "o2")
    assert d.belief == pytest.approx(0.95)
    assert d.policy_version == "test_cal_v1:margin"


def test_high_sam3_score_with_failed_margin_is_rejected():
    """Spec test 1 (unit half): SAM3 confidence cannot rescue a candidate whose
    crops match the negative controls better than the target."""
    pol = RelativeMarginPolicy(_cal())
    d = pol.grade(
        [
            _ev("o1", 1, sam3=0.99, target=0.05, wall=0.20),
            _ev("o2", 2, sam3=0.97, target=0.06, wall=0.21),
        ]
    )
    assert d.status == REJECTED
    assert "negative-control margin" in d.reason


def test_margin_between_thresholds_stays_unknown():
    pol = RelativeMarginPolicy(_cal())
    d = pol.grade(
        [
            _ev("o1", 1, target=0.11, wall=0.10),  # margin 0.01
            _ev("o2", 2, target=0.11, wall=0.10),
        ]
    )
    assert d.status == UNKNOWN
    assert "between" in d.reason


def test_margin_policy_needs_min_views():
    pol = RelativeMarginPolicy(_cal(min_views=2))
    d = pol.grade([_ev("o1", 1)])
    assert d.status == UNKNOWN
    assert "1 passing view" in d.reason
    assert (
        RelativeMarginPolicy(_cal(min_views=1)).grade([_ev("o1", 1)]).status == ADMITTED
    )


def test_cotimestamped_detections_do_not_double_count_views():
    pol = RelativeMarginPolicy(_cal(min_views=2))
    assert pol.grade([_ev("o1", 7), _ev("o2", 7)]).status == UNKNOWN


def test_geometrically_inconsistent_passing_views_stay_unknown():
    pol = RelativeMarginPolicy(_cal())
    d = pol.grade(
        [_ev("o1", 1, xyz=(0.0, 0.0, 0.0)), _ev("o2", 2, xyz=(9.0, 0.0, 0.0))]
    )
    assert d.status == UNKNOWN
    assert "inconsist" in d.reason


def test_invalid_views_cannot_vouch_or_condemn():
    """Views failing mask/depth validity or the detector floor are ungradable
    — they neither admit nor reject (fail-closed to UNKNOWN)."""
    pol = RelativeMarginPolicy(_cal())
    d = pol.grade(
        [_ev("o1", 1, sam3=0.1), _ev("o2", 2, depth=3), _ev("o3", 3, target=None)]
    )
    assert d.status == UNKNOWN
    assert "no valid margin-scored views" in d.reason


def test_zero_evidence_is_unknown_never_a_verdict():
    """Spec test 5 (policy half): a zero-hit query hands the policy nothing;
    nothing must stay UNKNOWN (the coverage/episode side lives in the
    planability tests)."""
    for pol in (RelativeMarginPolicy(_cal()), GeometricPolicy(_cal())):
        assert pol.grade([]).status == UNKNOWN


def test_multiple_true_frames_never_cause_a_rejection():
    """Spec test 4: margins are target-vs-control per view, so adding MORE
    passing views can only help. 7 views of a true object admit; they can
    never flip to REJECTED by any frame-vs-frame comparison."""
    pol = RelativeMarginPolicy(_cal())
    many = [_ev(f"o{i}", i, xyz=(1.0 + 0.01 * i, 2.0, 0.0)) for i in range(7)]
    assert pol.grade(many).status == ADMITTED
    assert pol.grade(many[:2]).status == ADMITTED


# ---------------------------------------------------------------------------
# GeometricPolicy semantics
# ---------------------------------------------------------------------------


def test_geometric_policy_admits_consistent_multiview_without_margins():
    pol = GeometricPolicy(_cal())
    d = pol.grade(
        [
            _ev("o1", 1, target=None, xyz=(1.0, 2.0, 0.0)),
            _ev("o2", 2, target=None, xyz=(1.3, 2.0, 0.0)),
        ]
    )
    assert d.status == ADMITTED
    assert d.policy_version == "test_cal_v1:geometric"


def test_geometric_policy_admits_the_consistent_phantom_the_margin_rejects():
    """The margin-vs-geometric difference: a multi-view-consistent phantom (high SAM3,
    control-dominated crops) passes geometric validation and fails the margin
    gate. Both graded on identical evidence."""
    evs = [
        _ev("o1", 1, sam3=0.95, target=0.05, wall=0.20),
        _ev("o2", 2, sam3=0.93, target=0.06, wall=0.21, xyz=(1.1, 2.0, 0.0)),
    ]
    assert GeometricPolicy(_cal()).grade(evs).status == ADMITTED
    assert RelativeMarginPolicy(_cal()).grade(evs).status == REJECTED


def test_geometric_policy_never_rejects():
    pol = GeometricPolicy(_cal())
    assert pol.grade([_ev("o1", 1, sam3=0.05)]).status == UNKNOWN
    assert (
        pol.grade([_ev("o1", 1, xyz=(0, 0, 0)), _ev("o2", 2, xyz=(9, 0, 0))]).status
        == UNKNOWN
    )


# ---------------------------------------------------------------------------
# grade_object: write-through + skip semantics
# ---------------------------------------------------------------------------


class StatusDb(FakeDb):
    """FakeDb whose FIRST call answers the admission_status read."""

    def __init__(self, status):
        super().__init__(rows=[{"s": status}])
        self._status_answered = False

    def execute(self, query, params=None, **kwargs):
        self.calls.append((query, dict(params or {}, **kwargs)))
        if not self._status_answered:
            self._status_answered = True
            return self.rows, None, None
        return [], None, None


def test_grade_object_admits_through_the_sanctioned_writer():
    db = StatusDb(CANDIDATE)
    d = grade_object(
        db,
        "O7",
        RelativeMarginPolicy(_cal()),
        evidences=[_ev("O7_1", 1), _ev("O7_2", 2)],
    )
    assert d.status == ADMITTED
    q, p = db.calls[-1]
    assert f"o.{ADMISSION_BELIEF} = $belief" in q
    assert p["status"] == ADMITTED
    assert p["policy"] == "test_cal_v1:margin"
    assert p["obs"] == ["O7_1", "O7_2"]
    assert p["belief"] == pytest.approx(0.9)


def test_grade_object_rejects_with_reason():
    db = StatusDb(CANDIDATE)
    d = grade_object(
        db,
        "O7",
        RelativeMarginPolicy(_cal()),
        evidences=[_ev("O7_1", 1, target=0.05, wall=0.2)],
    )
    assert d.status == REJECTED
    assert db.calls[-1][1]["status"] == REJECTED


def test_grade_object_unknown_writes_nothing():
    db = StatusDb(CANDIDATE)
    d = grade_object(db, "O7", RelativeMarginPolicy(_cal()), evidences=[])
    assert d.status == UNKNOWN
    assert len(db.calls) == 1  # only the status read


@pytest.mark.parametrize("status", [TRUSTED_PRIOR, ADMITTED, REJECTED, None])
def test_grade_object_only_grades_candidates(status):
    """Trusted priors and settled decisions are never re-graded — a later bad
    view must not revoke a sound admission, and rejections are sticky."""
    db = StatusDb(status)
    d = grade_object(
        db, "O7", RelativeMarginPolicy(_cal()), evidences=[_ev("a", 1), _ev("b", 2)]
    )
    assert d.status == UNKNOWN
    assert "not graded" in d.reason
    assert len(db.calls) == 1


def test_load_observation_evidence_scopes_margins_to_the_prompt():
    rows = [
        {
            "ns": "O7_1",
            "ts": 1,
            "sam3": 0.9,
            "target": 0.3,
            "controls": json.dumps({"a wall": 0.1}),
            "depth_px": 400,
            "prompt": "a mug",
            "vx": 1.0,
            "vy": 2.0,
            "vz": 0.0,
        },
        {
            "ns": "O7_2",
            "ts": 2,
            "sam3": 0.8,
            "target": 0.4,
            "controls": json.dumps({"a wall": 0.1}),
            "depth_px": 300,
            "prompt": "a coffee maker",
            "vx": 1.1,
            "vy": 2.0,
            "vz": 0.0,
        },
    ]
    evs = load_observation_evidence(FakeDb(rows), "O7", prompt="A Mug")
    same, other = evs
    assert same.margin == pytest.approx(0.2)  # prompt matches (case-insensitive)
    # The other prompt's view keeps geometry + detector terms but its margin is
    # not comparable -> stripped (an object admitted-for-"mug" must not vouch
    # for "coffee maker").
    assert other.margin is None
    assert other.target_score is None
    assert other.sam3_score == pytest.approx(0.8)
    assert other.view_xyz == (1.1, 2.0, 0.0)


# ---------------------------------------------------------------------------
# Drift guards: persisted props <-> read-back query <-> heracles writer
# ---------------------------------------------------------------------------


def test_obs_evidence_props_match_the_read_back_query_and_heracles_writer():
    """One view's evidence must survive _obs_dict -> heracles MERGE ->
    _EVIDENCE_QUERY intact; pin the property names on all three artifacts."""
    from openset_query.evidence import admission_policy as ap
    from openset_query.object_discovery import Observation, _obs_evidence_props

    o = Observation(
        timestamp_ns=5,
        mask_file="",
        bbox_2d=(0, 0, 4, 4),
        score=0.9,
        depth_valid_px=123,
        view_xyz=(1.0, 2.0, 3.0),
        target_prompt="a mug",
        target_score=0.3,
        control_scores={"a wall": 0.1},
    )
    props = _obs_evidence_props(o)
    assert props["depth_valid_px"] == 123
    assert props["margin"] == pytest.approx(0.2)
    assert props["control_max_score"] == pytest.approx(0.1)
    assert json.loads(props["control_scores_json"]) == {"a wall": 0.1}
    for key in (
        "target_prompt",
        "target_score",
        "control_scores_json",
        "depth_valid_px",
        "view_x",
        "view_y",
        "view_z",
    ):
        assert key in props
        assert (
            f"obs.{key}"
            in ap._EVIDENCE_QUERY.replace(
                "obs.control_scores_json AS controls", "obs.control_scores_json"
            )
            or f"obs.{key}" in ap._EVIDENCE_QUERY
        )

    from openset_query.heracles_compat import INSERT_OBSERVATIONS

    for key in props:
        assert f"obs.{key}" in INSERT_OBSERVATIONS, f"observation writer drops {key}"


def test_admission_belief_constant_matches_heracles():
    heracles_constants = pytest.importorskip("heracles.constants")
    if not hasattr(heracles_constants, "ADMISSION_BELIEF"):
        pytest.skip("this heracles has no admission constants")
    assert ADMISSION_BELIEF == heracles_constants.ADMISSION_BELIEF


# ---------------------------------------------------------------------------
# The server's evidence payload (pure helper)
# ---------------------------------------------------------------------------


def test_build_evidence_payload_merges_dedup_and_carries_decisions():
    from openset_query.evidence.admission_policy import AdmissionDecision
    from openset_query.object_discovery import DiscoveredObject, Observation
    from openset_query.query_server import build_evidence_payload

    def disc(ts, score):
        return DiscoveredObject(
            obj=None,
            observations=[
                Observation(
                    timestamp_ns=ts,
                    mask_file="",
                    bbox_2d=(0, 0, 4, 4),
                    score=score,
                    depth_valid_px=99,
                    view_xyz=(1.0, 2.0, 0.0),
                    target_prompt="a mug",
                    target_score=0.3,
                    control_scores={"a wall": 0.1},
                )
            ],
        )

    decision = AdmissionDecision(
        status=ADMITTED,
        reason="ok",
        policy_version="test_cal_v1:margin",
        belief=0.9,
        observation_ids=("O5_1",),
    )
    payload = build_evidence_payload(
        [disc(1, 0.9), disc(2, 0.8)], ["O5", "O5"], {"O5": decision}
    )

    assert set(payload) == {"O5"}
    rows = payload["O5"]["observations"]
    assert [r["obs_id"] for r in rows] == ["O5_1", "O5_2"]
    assert rows[0]["margin"] == pytest.approx(0.2)
    assert rows[0]["control_scores"] == {"a wall": 0.1}  # raw terms travel
    assert payload["O5"]["admission"]["status"] == ADMITTED
    assert payload["O5"]["admission"]["policy_version"] == "test_cal_v1:margin"


def test_build_evidence_payload_admission_none_when_grading_off():
    from openset_query.object_discovery import DiscoveredObject, Observation
    from openset_query.query_server import build_evidence_payload

    d = DiscoveredObject(
        obj=None,
        observations=[
            Observation(timestamp_ns=1, mask_file="", bbox_2d=(0, 0, 4, 4), score=0.9)
        ],
    )
    payload = build_evidence_payload([d], ["O5"], {})
    assert payload["O5"]["admission"] is None
    assert payload["O5"]["observations"][0]["margin"] is None


# ---------------------------------------------------------------------------
# Calibration math (eval/calibrate_admission.py)
# ---------------------------------------------------------------------------


def test_derive_thresholds_sits_above_the_phantom_population():
    from openset_query.eval.calibrate_admission import derive_thresholds

    out = derive_thresholds(
        tp_margins=[0.15, 0.2, 0.25, 0.3],
        fp_margins=[-0.05, 0.0, 0.04],
        fp_quantile=1.0,
        reject_quantile=0.5,
        epsilon=0.01,
    )
    assert out["margin_admit"] == pytest.approx(0.05)
    assert out["margin_reject"] == pytest.approx(0.0)
    assert out["tp_kept_fraction"] == pytest.approx(1.0)
    assert out["fp_admitted_fraction"] == pytest.approx(0.0)


def test_derive_thresholds_falls_back_without_phantoms():
    from openset_query.eval.calibrate_admission import derive_thresholds

    out = derive_thresholds(
        tp_margins=[0.1, 0.2, 0.3],
        fp_margins=[],
        fp_quantile=1.0,
        reject_quantile=0.5,
        epsilon=0.01,
    )
    assert "fallback" in out["basis"]
    assert out["margin_reject"] <= out["margin_admit"]


def test_derived_artifact_passes_the_loader_contract(tmp_path):
    """The calibration script's output must be loadable by the serving policy
    — pin the schema through the real validator."""
    artifact = {
        "version": "sam3_admission_v1",
        "control_prompts": ["a wall"],
        "margin_admit": 0.05,
        "margin_reject": 0.0,
        "min_views": 2,
        "min_sam3_score": 0.25,
        "min_depth_valid_px": 50,
        "max_view_spread_m": 1.2,
        "calibrated_on": {"archive": "x"},
    }
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(artifact))
    pol = make_admission_policy("margin", load_calibration(str(path)))
    assert pol.policy_version == "sam3_admission_v1:margin"
