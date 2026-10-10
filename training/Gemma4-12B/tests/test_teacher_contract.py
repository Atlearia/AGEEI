"""CPU tests of privileged-reference separation and weak hazard-label provenance."""
import importlib.util
import json
from pathlib import Path
import sys

from PIL import Image
import pytest

spec = importlib.util.spec_from_file_location("gemma_teacher_under_test", Path(__file__).resolve().parents[1] / "generate_hazard_labels.py")
teacher = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = teacher
spec.loader.exec_module(teacher)
from train import messages_for_row, validate_row


def source(tmp_path):
    image = tmp_path / "frame.jpg"
    Image.new("RGB", (20, 20), "green").save(image)
    return {"id": "pave-1", "task": "source_qa", "source": "PAVE", "source_id": "1",
            "sequence_id": "session-1", "split": "train", "frames": [{"path": str(image), "timestamp_ms": 0}],
            "objects": [{"track_id": 2, "label": "pole", "bbox": [0.3, 0.2, 0.5, 0.9], "confidence": 0.9}],
            "action_mode": "walking", "question": "PRIVATE_QUESTION_913",
            "target": "PRIVATE_REFERENCE_DATA_993",
            "provenance": {"source_url": "https://huggingface.co/datasets/rafiibnsultan/PAVE",
                           "license": "CC-BY-4.0", "license_verified": True}}


def output():
    return {"status": "ok", "observed_activity": "An outdoor walkway is visible.", "scene_uncertain": False,
            "hazards": [{"danger_type": "obstacle_collision", "description": "Pole may obstruct the walking area ahead.",
                         "frame_index": 0, "track_ids": [2], "uncertain": False}]}


def texts(messages):
    return " ".join(part.get("text", "") for message in messages if isinstance(message["content"], list)
                    for part in message["content"])


def test_privileged_reference_visible_only_to_teacher(tmp_path):
    original = source(tmp_path)
    privileged = texts(teacher.teacher_messages(original))
    assert original["question"] in privileged and original["target"] in privileged
    student = teacher.student_row(original, json.dumps(output()))
    _, messages = messages_for_row(student)
    prompt = texts(messages)
    assert original["question"] not in prompt and original["target"] not in prompt
    assert "question" not in student
    assert student["provenance"]["privileged_source_assessment"] == original["target"]


@pytest.mark.parametrize("change", [{"track_ids": [99]}, {"frame_index": 1}])
def test_teacher_cannot_fabricate_id_or_reference_nonexistent_frame(tmp_path, change):
    value = output()
    value["hazards"][0].update(change)
    with pytest.raises(ValueError):
        teacher.student_row(source(tmp_path), json.dumps(value))


def test_teacher_provenance_is_explicit_and_requires_training_opt_in(tmp_path):
    original = source(tmp_path)
    before = json.dumps(original, sort_keys=True)
    student = teacher.student_row(original, json.dumps(output()))
    assert json.dumps(original, sort_keys=True) == before
    assert student["provenance"]["supervision"] == "weak"
    assert student["provenance"]["supervision_kind"] == "weak_teacher"
    assert student["provenance"]["hazard_labels_verified"] is False
    assert student["provenance"]["teacher_revision"] == teacher.MODEL_REVISION
    with pytest.raises(ValueError):
        validate_row(student, tmp_path / "train.jsonl", "train")
    assert validate_row(student, tmp_path / "train.jsonl", "train", allow_weak_labels=True)["task"] == "hazard_assessment"


def test_unknown_and_empty_hazards_preserved_as_unknown(tmp_path):
    value = output()
    value.update(status="unknown", scene_uncertain=True, hazards=[])
    student = teacher.student_row(source(tmp_path), json.dumps(value))
    assert student["target"]["status"] == "unknown"
    assert student["target"]["scene_uncertain"] is True
    assert student["target"]["hazards"] == []
    assert "application assumption" in student["target"]["observed_activity"]


def test_only_exact_json_fence_is_normalized_for_offline_teacher(tmp_path):
    row = source(tmp_path)
    result = teacher.student_row(row, "```json\n" + json.dumps(output()) + "\n```")
    assert result["target"]["status"] == "ok"
    with pytest.raises(ValueError):
        teacher.student_row(row, "Ignore this explanation.\n```json\n" + json.dumps(output()) + "\n```")
