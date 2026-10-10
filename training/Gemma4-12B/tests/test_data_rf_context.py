import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location("gemma_rf_context", Path(__file__).resolve().parents[1]/"attach_rf_boxes.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def result(boxes, scores, labels):
    return SimpleNamespace(xyxy=boxes, confidence=scores, data={"class_name": labels})


def test_actual_rf_metadata_normalization_and_frame_ids():
    detections = result([[-2, 3, 110, 48], [1, 1, 1, 2], [20, 5, 60, 30]], [.7, .8, .6], ["curb", "pole", "stairs"])
    boxes = module.object_rows(detections, 100, 50)
    assert len(boxes) == 2
    assert boxes[0] == {"track_id": 1, "bbox": [0.0, .06, 1.0, .96], "label": "curb", "confidence": .7}
    assert boxes[1]["track_id"] == 2
    assert boxes[1]["confidence"] == .6


def test_background_ignored_and_limit_respected():
    detections = result([[0, 0, 50, 50]]*3, [.8]*3, ["__background__", "person", "car"])
    assert len(module.object_rows(detections, 100, 100, max_objects=1)) == 1
    assert module.object_rows(detections, 100, 100)[0]["label"] == "person"


def test_nonfinite_detector_prediction_fails_closed():
    with pytest.raises(ValueError, match="invalid"):
        module.object_rows(result([[0, 0, 50, 50]], [float("nan")], ["car"]), 100, 100)
