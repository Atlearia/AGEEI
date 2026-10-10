import importlib.util
import io
import json
from pathlib import Path
import sys
import urllib.request

import pytest
from PIL import Image

MODULE_PATH = Path(__file__).resolve().parents[1] / "prepare_data.py"
spec = importlib.util.spec_from_file_location("gemma_prepare_data", MODULE_PATH)
data = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = data
spec.loader.exec_module(data)


def source_row(session="recording_a", index="000001", source_id=1):
    return {"id": source_id, "session": f"PAVE/{session}/camera_head/left/", "index": index,
            "question": "How accessible is this path?",
            "assessment": "[assessment] A curb borders the path. [/assessment] [distance] 2.0m [/distance] [p] curb [/p][SEG]"}


def test_source_target_preserved_without_missing_modality_tokens():
    row = data.parse_row(source_row())
    assert row["target"] == "A curb borders the path."
    assert "video_frames/000001.png" in row["image_url"]
    assert "severity" not in row and "track_ids" not in row


@pytest.mark.parametrize("field,value", [
    ("session", "PAVE/../../secret/camera_head/left/"),
    ("session", "C:/private/file"), ("index", "../001"),
    ("index", "000001.png"), ("assessment", "No explicit assessment paragraph"),
    ("assessment", "[assessment] A [/assessment][assessment] B [/assessment]"),
    ("question", ""),
])
def test_unsafe_or_unsupported_source_rejected(field, value):
    raw = source_row()
    raw[field] = value
    with pytest.raises(ValueError):
        data.parse_row(raw)


def test_recording_splits_and_official_test_exclusion():
    training = [data.parse_row(source_row(f"session_{i}", f"{j:06}", i * 10 + j))
                for i in range(12) for j in range(3)]
    evaluation = [data.parse_row(source_row("official_eval", source_id=123))]
    splits, excluded = data.partition_rows(training, evaluation, {"session_0"})
    assert len(excluded) == 3
    groups = {split: {row["session_id"] for row in rows} for split, rows in splits.items()}
    assert not groups["train"] & groups["validation"]
    assert not groups["train"] & groups["test"]
    assert "session_0" not in groups["train"] | groups["validation"]
    assert groups["test"] == {"official_eval"}
    assert data.partition_rows(training, evaluation, {"session_0"}) == (splits, excluded)


def test_split_leakage_refused():
    row = data.parse_row(source_row())
    with pytest.raises(ValueError, match="overlap"):
        data.partition_rows([row], [row], set())


def test_budget_samples_across_groups_and_deduplicates_frames():
    rows = [data.parse_row(source_row(f"r{i}", f"{j:06}", i * 20 + j)) for i in range(4) for j in range(4)]
    selected = data.select_frames(rows + rows, 6)
    assert len(selected) == 6
    assert len({row["frame_key"] for row in selected}) == 6
    assert len({row["session_id"] for row in selected[:4]}) == 4


def test_output_path_boundaries(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        data.safe_path(tmp_path, "../outside.jpg")
    assert data.safe_path(tmp_path, "media/frame.jpg").is_relative_to(tmp_path)


def test_image_and_real_training_row_contract(tmp_path):
    raw = io.BytesIO()
    Image.new("RGB", (1600, 900), "green").save(raw, "PNG")
    image = tmp_path / "media" / "frame.jpg"
    metadata = data.image_asset(raw.getvalue(), image, 896)
    row = data.make_example(data.parse_row(source_row()), "train", image, metadata)
    assert image.is_file()
    assert metadata["prepared_size"] == [896, 504]
    assert row["task"] == "source_qa"
    assert row["target"] == "A curb borders the path."
    assert row["objects"] == []
    assert row["provenance"]["license_verified"] is True
    assert row["frames"][0]["path"] == str(image.resolve())


def test_cached_corruption_fails(tmp_path):
    (tmp_path / "PAVE_train85.jsonl").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        data.cached_get(tmp_path, "PAVE_train85.jsonl", "https://huggingface.co/not-used", 100)


def test_authorization_does_not_follow_external_redirect():
    request = urllib.request.Request("https://huggingface.co/datasets/a", headers={"Authorization": "Bearer test"})
    redirected = data.ScopedRedirect().redirect_request(request, None, 302, "", {}, "https://cdn.example/media")
    assert not redirected.has_header("Authorization")


def test_shared_download_budget_refuses_excess():
    budget = data.DownloadBudget(10)
    budget.claim(6)
    budget.claim(4)
    with pytest.raises(ValueError, match="budget"):
        budget.claim(1)


def test_bounded_parallel_results_keep_source_order(monkeypatch):
    def fake(url, limit, budget):
        payload = url.encode()
        budget.claim(len(payload))
        return payload
    monkeypatch.setattr(data, "bounded_get", fake)
    rows = [{"image_url": str(i)} for i in range(17)]
    budget = data.DownloadBudget(1000)
    results = list(data.download_rows(rows, 3, budget))
    assert [row for row, _ in results] == rows
    assert [raw for _, raw in results] == [str(i).encode() for i in range(17)]


def test_resume_reuses_only_matching_source_and_intact_media(tmp_path):
    row = data.parse_row(source_row())
    raw = io.BytesIO()
    Image.new("RGB", (100, 50), "green").save(raw, "PNG")
    image = tmp_path/"frame.jpg"
    media = data.image_asset(raw.getvalue(), image, 896)
    prior = data.make_example(row, "train", image, media)
    cached = {prior["id"]: prior}
    assert data.reusable_media(row, "train", image, cached, 896) == media
    assert data.reusable_media(row, "validation", image, cached, 896) is None
    image.write_bytes(b"corrupt")
    assert data.reusable_media(row, "train", image, cached, 896) is None
