"""Meaningful CPU checks for supervision boundaries and leakage preflight."""
import importlib.util
import json
from pathlib import Path
import sys

from PIL import Image
import pytest

spec = importlib.util.spec_from_file_location("gemma_train_under_test", Path(__file__).resolve().parents[1] / "train.py")
train = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = train
spec.loader.exec_module(train)


def row(path, split="train", **updates):
    value = {"id": f"example-{split}", "task": "source_qa", "source": "test-source",
             "source_id": f"image-{split}", "sequence_id": f"sequence-{split}", "split": split,
             "frames": [{"path": str(path), "timestamp_ms": 0}], "objects": [], "action_mode": "walking",
             "question": "Describe the visible obstacle.", "target": "A pole is visible.",
             "provenance": {"source_url": "https://example.com/source", "license": "CC-BY-4.0", "license_verified": True}}
    value.update(updates)
    return value


def manifests(tmp_path, *, duplicate=False, group_overlap=False):
    paths = []
    for split, color in (("train", "green"), ("validation", "red")):
        path = tmp_path / f"{split}.png"
        Image.new("RGB", (16, 16), "green" if duplicate else color).save(path)
        value = row(path, split)
        if group_overlap:
            value["sequence_id"] = "same-recording"
        manifest = tmp_path / f"{split}.jsonl"
        manifest.write_text(json.dumps(value) + "\n", encoding="utf-8")
        paths.append(manifest)
    return paths


def test_only_assistant_tokens_receive_loss():
    assert train.assistant_labels([10, 20, 30, 40, 50], [10, 20, 30], 10) == [-100, -100, -100, 40, 50]


def test_token_boundary_mismatch_is_fatal():
    with pytest.raises(ValueError, match="exact token prefix"):
        train.assistant_labels([10, 20, 99, 40], [10, 20, 30], 10)


@pytest.mark.parametrize("full,prefix,budget", [([1], [1], 10), ([1, 2, 3], [1], 1)])
def test_empty_or_overlong_target_cannot_silently_train(full, prefix, budget):
    with pytest.raises(ValueError):
        train.assistant_labels(full, prefix, budget)


def test_valid_manifests_and_dry_run_without_gpu(tmp_path):
    paths = manifests(tmp_path)
    data = train.validate_manifests(*paths)
    assert len(data["train"]) == len(data["validation"]) == 1
    assert train.main(["--train", str(paths[0]), "--validation", str(paths[1]),
                       "--output", str(tmp_path / "output"), "--dry-run"]) == 0
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("kind", ["duplicate", "group_overlap"])
def test_cross_split_frame_or_recording_leakage_rejected(tmp_path, kind):
    paths = manifests(tmp_path, **{kind: True})
    with pytest.raises(ValueError, match="across"):
        train.validate_manifests(*paths)


def test_unknown_terms_rejected(tmp_path):
    path = tmp_path / "image.png"
    Image.new("RGB", (16, 16)).save(path)
    value = row(path)
    value["provenance"]["license_verified"] = False
    with pytest.raises(ValueError, match="provenance"):
        train.validate_row(value, tmp_path / "train.jsonl", "train")


def test_partial_dataset_download_is_rejected(tmp_path):
    paths = manifests(tmp_path)
    (tmp_path / "data_report.json").write_text(json.dumps({"status": "failed"}))
    with pytest.raises(ValueError, match="partially prepared"):
        train.validate_manifests(*paths)


def test_hazard_supervision_cannot_be_invented_from_qa(tmp_path):
    path = tmp_path / "image.png"
    Image.new("RGB", (16, 16)).save(path)
    value = row(path, task="hazard_assessment", target={"status": "ok", "observed_activity": "Sidewalk visible.",
                "scene_uncertain": False, "hazards": []})
    with pytest.raises(ValueError, match="verified source hazard"):
        train.validate_row(value, tmp_path / "train.jsonl", "train")


def test_pinned_model_and_language_only_lora_config():
    config = json.loads(train.DEFAULT_CONFIG.read_text())
    train.validate_config(config)
    assert train.re.fullmatch(config["target_modules_regex"], "model.language_model.layers.5.self_attn.q_proj")
    assert not train.re.fullmatch(config["target_modules_regex"], "model.embed_vision.q_proj")
    config["model_revision"] = "main"
    with pytest.raises(ValueError):
        train.validate_config(config)


def test_weak_teacher_targets_require_explicit_training_mode_and_policy(tmp_path):
    path = tmp_path / "image.png"
    Image.new("RGB", (16, 16)).save(path)
    value = row(path, task="hazard_assessment", target={"status": "ok", "observed_activity": "A sidewalk is visible.",
                "scene_uncertain": False, "hazards": []})
    value["provenance"].update(hazard_labels_verified=False, supervision="weak", supervision_kind="weak_teacher",
                               label_policy_id="pave-rf-gemma-teacher-v1", label_policy_version=1)
    with pytest.raises(ValueError, match="verified"):
        train.validate_row(value, tmp_path / "train.jsonl", "train")
    accepted = train.validate_row(value, tmp_path / "train.jsonl", "train", allow_weak_labels=True)
    assert accepted["provenance"]["hazard_labels_verified"] is False
    del value["provenance"]["label_policy_version"]
    with pytest.raises(ValueError):
        train.validate_row(value, tmp_path / "train.jsonl", "train", allow_weak_labels=True)


def test_absolute_deadline_wins_and_soft_stop_reserves_export(monkeypatch):
    monkeypatch.setattr(train.time, "time", lambda: 1000)
    budget = train.DeadlineBudget(7200, "1970-01-01T00:30:00Z", 300)
    assert budget.remaining() == 800
    budget.check_training()
    monkeypatch.setattr(train.time, "time", lambda: 1501)
    with pytest.raises(train.BudgetStop):
        budget.check_training()


def test_deadline_needs_timezone():
    with pytest.raises(ValueError, match="timezone"):
        train.DeadlineBudget(7200, "2026-10-09T23:00:00")
