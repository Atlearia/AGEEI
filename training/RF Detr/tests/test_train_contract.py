"""CPU contract checks: prevent costly jobs with wrong data/config or unsafe paths."""
from __future__ import annotations

import builtins
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("rf_detr_training_contract", ROOT / "train.py")
training = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(training)


@pytest.fixture(autouse=True)
def disallow_ml_imports(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "rfdetr", "pytorch_lightning", "transformers"}:
            pytest.fail(f"CPU config/preflight unexpectedly imported {name}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


def config_file(tmp_path, **overrides):
    config = training.load_json(ROOT / "configs" / "h100_medium.json")
    config["train"].update(overrides)
    target = tmp_path / "config.json"
    target.write_text(json.dumps(config), encoding="utf-8")
    return target


def write_split(dataset, split, *, category_name="pole", file_name="frame.jpg"):
    folder = dataset / split
    folder.mkdir(parents=True)
    (folder / "frame.jpg").write_bytes(b"existence-only fixture; full image validation belongs to prepare_data.py")
    annotation = {
        "images": [{"id": 1, "file_name": file_name, "width": 20, "height": 10}],
        "categories": [{"id": 1, "name": category_name, "supercategory": "object"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 2, 2]}],
    }
    (folder / "_annotations.coco.json").write_text(json.dumps(annotation), encoding="utf-8")
    return annotation


def test_dry_run_needs_no_dataset_dependencies_or_output(tmp_path, capsys):
    data = tmp_path / "not-downloaded"
    output = tmp_path / "not-created"
    assert training.main(["--data", str(data), "--output", str(output), "--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["model_class"] == "RFDETRMedium"
    assert plan["model"]["resolution"] == 576
    assert plan["effective_batch_size"] == 16
    assert plan["train"]["run_test"] is False
    assert not data.exists() and not output.exists()


def test_benchmark_does_not_overwrite_production_directory(tmp_path, capsys):
    output = tmp_path / "production"
    output.mkdir()
    checkpoint = output / "last.ckpt"
    checkpoint.write_bytes(b"keep")
    training.main(["--output", str(output), "--epochs", "18", "--benchmark", "--dry-run"])
    plan = json.loads(capsys.readouterr().out)
    assert Path(plan["train"]["output_dir"]) == tmp_path / "production-benchmark"
    assert plan["train"]["epochs"] == 1
    assert plan["planned_full_epochs"] == 18
    assert plan["train"]["early_stopping"] is False
    assert checkpoint.read_bytes() == b"keep"
    assert not (tmp_path / "production-benchmark").exists()


@pytest.mark.parametrize("override, message", [
    ({"epochs": 0}, "epochs"),
    ({"batch_size": True}, "batch_size"),
    ({"grad_accum_steps": -1}, "grad_accum_steps"),
    ({"num_workers": -1}, "num_workers"),
    ({"lr": -0.1}, "lr"),
    ({"lr_encoder": float("nan")}, "lr_encoder"),
    ({"amp_dtype": "bf61"}, "amp_dtype"),
    ({"devices": 4}, "one H100"),
    ({"run_test": True}, "test split"),
])
def test_invalid_training_settings_rejected_before_gpu(tmp_path, override, message):
    config = config_file(tmp_path, **override)
    with pytest.raises(ValueError, match=message):
        training.main(["--config", str(config), "--dry-run"])


def test_architecture_changes_need_new_verified_recipe(tmp_path):
    config = training.load_json(ROOT / "configs" / "h100_medium.json")
    config["model"]["resolution"] = 560
    file = tmp_path / "config.json"
    file.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="576"):
        training.main(["--config", str(file), "--dry-run"])


def test_benchmark_cannot_resume_existing_run(tmp_path):
    with pytest.raises(ValueError, match="cannot be combined"):
        training.main(["--benchmark", "--resume", str(tmp_path / "last.ckpt"), "--dry-run"])


def test_inference_checkpoint_cannot_resume_optimizer(tmp_path):
    with pytest.raises(ValueError, match="optimizer state"):
        training.main(["--resume", str(tmp_path / "checkpoint_best_total.pth"), "--dry-run"])


def test_category_mismatch_fails_before_dependency_or_gpu_checks(tmp_path):
    data = tmp_path / "data"
    write_split(data, "train", category_name="pole")
    write_split(data, "valid", category_name="car")
    with pytest.raises(ValueError, match="category schemas differ"):
        training.main(["--data", str(data), "--output", str(tmp_path / "output")])


def test_valid_split_reports_counts_and_fingerprint(tmp_path):
    write_split(tmp_path, "train")
    report = training.check_split(tmp_path, "train")
    assert report["images"] == report["annotations"] == 1
    assert report["categories"][0]["name"] == "pole"
    assert len(report["annotation_sha256"]) == 64


def test_path_traversal_is_rejected_even_when_target_exists(tmp_path):
    (tmp_path / "secret.jpg").write_bytes(b"outside split")
    write_split(tmp_path, "train", file_name="../secret.jpg")
    with pytest.raises(ValueError, match="escapes split"):
        training.check_split(tmp_path, "train")


def test_missing_image_fails_before_gpu(tmp_path):
    write_split(tmp_path, "train", file_name="absent.jpg")
    with pytest.raises(ValueError, match="Image missing"):
        training.check_split(tmp_path, "train")


def test_unresolved_annotation_category_fails(tmp_path):
    annotation = write_split(tmp_path, "train")
    annotation["annotations"][0]["category_id"] = 99
    path = tmp_path / "train" / "_annotations.coco.json"
    path.write_text(json.dumps(annotation), encoding="utf-8")
    with pytest.raises(ValueError, match="Unresolved annotation IDs"):
        training.check_split(tmp_path, "train")


def test_output_checkpoint_is_not_overwritten_accidentally(tmp_path):
    data = tmp_path / "data"
    write_split(data, "train")
    write_split(data, "valid")
    output = tmp_path / "output"
    output.mkdir()
    marker = output / "last.ckpt"
    marker.write_bytes(b"important")
    with pytest.raises(ValueError, match="nonempty"):
        training.main(["--data", str(data), "--output", str(output)])
    assert marker.read_bytes() == b"important"
