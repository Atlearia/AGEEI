import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("rfdetr_evaluate", Path(__file__).resolve().parents[1] / "evaluate.py")
evaluate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate)


def test_numpy_coco_category_ids_serialize_and_missing_metrics_stay_null():
    precision = np.full((10, 101, 2, 1, 3), -1.0, dtype=np.float64)
    recall = np.full((10, 2, 1, 3), -1.0, dtype=np.float64)
    precision[:, :, 0, 0, -1] = 0.625
    recall[:, 0, 0, -1] = 0.75
    evaluator = SimpleNamespace(params=SimpleNamespace(catIds=np.array([1, 2], dtype=np.int64)),
                                eval={"precision": precision, "recall": recall})
    truth = SimpleNamespace(cats={1: {"name": "curb"}, 2: {"name": "stairs"}},
                            getAnnIds=lambda catIds: [10, 11] if catIds == [1] else [])
    report = evaluate.summarize_per_class(evaluator, truth)
    restored = json.loads(json.dumps({"per_class": report}, allow_nan=False))["per_class"]
    assert type(report[0]["category_id"]) is int
    assert restored[0]["category_id"] == 1
    assert restored[0]["annotations"] == 2
    assert restored[0]["AP_50_95"] == pytest.approx(0.625)
    assert restored[0]["AR_100"] == pytest.approx(0.75)
    assert restored[1]["AP_50_95"] is None
    assert restored[1]["AP_50"] is None
    assert restored[1]["AR_100"] is None
