"""Proxy metric behavior: abstention is not success and invented references fail."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

spec = importlib.util.spec_from_file_location("gemma_proxy_evaluation", Path(__file__).resolve().parents[1] / "evaluate_adapter.py")
evaluate = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = evaluate
spec.loader.exec_module(evaluate)


def target(hazards=True):
    return {"status":"ok", "observed_activity":"An outdoor path is visible.", "scene_uncertain":False,
            "hazards":[{"danger_type":"trip_fall", "description":"A curb may cause a trip.",
                        "frame_index":0, "track_ids":[1], "uncertain":False}] if hazards else []}


def score(prediction, reference=None):
    return evaluate.score_prediction(json.dumps(prediction), reference or target(), frame_count=1, track_ids={1,2})


def test_valid_unknown_is_counted_but_not_agreement_with_known_hazard():
    prediction = {**target(False), "status":"unknown", "scene_uncertain":True}
    result = score(prediction)
    assert result["valid_json"] and result["valid_contract"]
    assert not result["determinate"] and result["hazard_presence_agrees"] is False
    assert result["id_false_negative"] == 1
    report = evaluate.summarize([{"mode":"adapter", "generation_seconds":2.5, "metrics":result}])["adapter"]
    assert report["valid_contract_rate"] == 1 and report["track_id_recall"] == 0
    assert report["generation_seconds_mean"] == 2.5


def test_invented_track_is_rejected_instead_of_counted_as_correct_hazard_presence():
    prediction = target()
    prediction["hazards"][0]["track_ids"] = [999]
    result = score(prediction)
    assert result["valid_json"] and result["valid_schema"]
    assert not result["valid_references"] and not result["valid_contract"]
    assert result["hazard_presence_agrees"] is False


def test_correct_object_with_wrong_category_reduces_category_agreement_only():
    prediction = target()
    prediction["hazards"][0]["danger_type"] = "electrical"
    result = score(prediction)
    assert result["hazard_presence_agrees"] is True
    assert result["id_true_positive"] == 1 and result["id_false_positive"] == result["id_false_negative"] == 0
    assert result["category_matched_ids"] == 1 and result["category_agreeing_ids"] == 0


def test_unknown_teacher_target_is_not_ground_truth_for_safe_or_dangerous():
    reference = {**target(False), "status":"unknown", "scene_uncertain":True}
    result = score(target(), reference)
    assert result["target_determinate"] is False and result["hazard_presence_agrees"] is None
    report = evaluate.summarize([{"mode":"base", "generation_seconds":1.0, "metrics":result}])["base"]
    assert report["hazard_presence_agreement"] is None and report["track_id_precision"] is None


def test_malformed_json_is_not_a_successful_negative():
    result = evaluate.score_prediction("not json", target(False), frame_count=1, track_ids={1})
    assert result["valid_json"] is False and result["hazard_presence_agrees"] is False


def test_deadline_requires_timezone_and_parses_utc():
    with pytest.raises(ValueError, match="explicit timezone"):
        evaluate.deadline_timestamp("2026-10-09T23:00:00")
    assert evaluate.deadline_timestamp("2026-10-09T23:00:00Z") == evaluate.deadline_timestamp("2026-10-09T19:00:00-04:00")


def test_complete_json_fence_fails_strict_output_but_scores_separate_content():
    raw = "```json\n" + json.dumps(target()) + "\n```"
    strict = evaluate.score_prediction(raw, target(), frame_count=1, track_ids={1, 2})
    content = evaluate.score_content_proxy(raw, target(), frame_count=1, track_ids={1, 2})
    assert strict["valid_json"] is False and strict["valid_contract"] is False
    assert strict["hazard_presence_agrees"] is False and strict["id_false_negative"] == 1
    assert content["valid_json"] is True and content["valid_contract"] is True
    assert content["hazard_presence_agrees"] is True and content["id_true_positive"] == 1


@pytest.mark.parametrize("wrapper", [
    "1.0\n{body}",
    "1.0\n```json\n{body}\n```",
    "```json\n{body}\n```\nextra prose",
    "```json\n{body}\n```\n```json\n{body}\n```",
])
def test_content_proxy_does_not_scrape_prefixed_prose_or_multiple_blocks(wrapper):
    raw = wrapper.format(body=json.dumps(target()))
    strict = evaluate.score_prediction(raw, target(), frame_count=1, track_ids={1})
    content = evaluate.score_content_proxy(raw, target(), frame_count=1, track_ids={1})
    assert strict["valid_json"] is False and content["valid_json"] is False
    assert content["valid_contract"] is False and content["hazard_presence_agrees"] is False


def test_report_separates_format_difference_from_identical_hazard_content(tmp_path):
    reference = target()
    plain = json.dumps(reference)
    records = []
    for mode, raw in [("base", "```json\n" + plain + "\n```"), ("adapter", plain)]:
        records.append({"example_id": "same-example", "mode": mode, "generation_seconds": 1.0,
                        "metrics": evaluate.score_prediction(raw, reference, frame_count=1, track_ids={1}),
                        evaluate.CONTENT_PROXY_KEY: evaluate.score_content_proxy(raw, reference, frame_count=1, track_ids={1})})
    evaluate.write_report(tmp_path, {"interpretation": "Teacher agreement, not human accuracy"}, records, "completed")
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["results"]["base"]["valid_contract_rate"] == 0
    assert report["results"]["adapter"]["valid_contract_rate"] == 1
    for mode in ("base", "adapter"):
        assert report[evaluate.CONTENT_PROXY_KEY][mode]["valid_contract_rate"] == 1
        assert report[evaluate.CONTENT_PROXY_KEY][mode]["hazard_presence_agreement"] == 1
        assert report[evaluate.CONTENT_PROXY_KEY][mode]["track_id_recall"] == 1
