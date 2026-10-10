import base64
import importlib.util
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from PIL import Image
import pytest

spec = importlib.util.spec_from_file_location("rf_worker", Path(__file__).resolve().parents[1] / "worker.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def encoded_image():
    stream = BytesIO()
    Image.new("RGB", (100, 50)).save(stream, format="PNG")
    return base64.b64encode(stream.getvalue()).decode()


def test_health_does_not_load_weights():
    client = TestClient(worker.create_app())
    response = client.get("/health").json()
    assert response["loaded"] is False
    assert response["checkpoint_kind"] == "pretrained_coco"


def test_invalid_image_rejected_without_model():
    client = TestClient(worker.create_app())
    assert client.post("/detect", json={"image_base64": "not an image"}).status_code == 422
    assert client.get("/health").json()["loaded"] is False


def test_normalization_and_clipping():
    detector = worker.Detector()
    prediction = SimpleNamespace(xyxy=[[-5, 10, 70, 60]], confidence=[.8], data={"class_name": ["pole"]})
    detector.model = SimpleNamespace(predict=lambda *args, **kwargs: prediction)
    result = TestClient(worker.create_app(detector)).post("/detect", json={"image_base64": encoded_image()}).json()
    assert result["boxes"] == [{"bbox": [0., .2, .7, 1.], "label": "pole", "confidence": .8}]


def test_failure_never_returns_empty_success():
    detector = worker.Detector()
    def fail(*args, **kwargs):
        raise RuntimeError("CUDA unavailable")
    detector.model = SimpleNamespace(predict=fail)
    response = TestClient(worker.create_app(detector)).post("/detect", json={"image_base64": encoded_image()})
    assert response.status_code == 503


def test_busy_rejects_instead_of_queuing_frames():
    detector = worker.Detector()
    detector.lock.acquire()
    try:
        response = TestClient(worker.create_app(detector)).post("/detect", json={"image_base64": encoded_image()})
        assert response.status_code == 503
    finally:
        detector.lock.release()


@pytest.mark.parametrize("threshold", [0, 1, -1, float("nan")])
def test_invalid_threshold(threshold):
    with pytest.raises(ValueError):
        worker.Detector(threshold=threshold)


def test_exported_class_names_require_unique_nonempty_list(tmp_path):
    path = tmp_path / "class_names.json"
    path.write_text('["curb", "pole"]', encoding="utf-8")
    assert worker.read_class_names(path) == ["curb", "pole"]
    path.write_text('["pole", "pole"]', encoding="utf-8")
    with pytest.raises(ValueError):
        worker.read_class_names(path)
