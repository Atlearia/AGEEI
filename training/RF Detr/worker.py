"""Local HTTP inference worker for RF-DETR Medium; imports do not load GPU weights."""
from __future__ import annotations

import argparse
import base64
import binascii
from io import BytesIO
import importlib.metadata
import json
import logging
import math
from pathlib import Path
import threading

from fastapi import FastAPI, HTTPException
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

MAX_IMAGE_PIXELS = 4_000_000


class DetectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image_base64: str = Field(min_length=1, max_length=4_000_000)


def decode_image(encoded: str):
    if encoded.startswith("data:image/"):
        encoded = encoded.split(",", 1)[-1]
    try:
        raw = base64.b64decode(encoded, validate=True)
        with Image.open(BytesIO(raw)) as image:
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise ValueError("Image exceeds four million pixels")
            if image.format not in {"JPEG", "PNG", "WEBP"}:
                raise ValueError("Use JPEG, PNG or WebP")
            return ImageOps.exif_transpose(image).convert("RGB")
    except (binascii.Error, OSError, UnidentifiedImageError) as error:
        raise ValueError("Invalid base64 image") from error


def read_class_names(path):
    names = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(names, list) or not names or any(not isinstance(v, str) or not v.strip() for v in names) or len(set(names)) != len(names):
        raise ValueError("Class metadata must be a nonempty JSON list of unique class names")
    return names


class Detector:
    def __init__(self, checkpoint=None, threshold=0.3, expected_class_names=None):
        if not math.isfinite(threshold) or not 0 < threshold < 1:
            raise ValueError("threshold must be between zero and one")
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self.threshold = threshold
        self.expected_class_names = expected_class_names
        self.model = None
        self.lock = threading.Lock()

    def load(self):
        if self.model is not None:
            return
        if self.checkpoint and not self.checkpoint.is_file():
            raise RuntimeError("Requested detector checkpoint does not exist; no fallback was loaded")
        if importlib.metadata.version("rfdetr") != "1.11.2":
            raise RuntimeError("Expected RF-DETR 1.11.2")
        import torch
        from rfdetr import RFDETRMedium
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA with BF16 support is required")
        if self.checkpoint:
            model = RFDETRMedium.from_checkpoint(str(self.checkpoint), device="cuda")
        else:
            model = RFDETRMedium(device="cuda")
        if not isinstance(model, RFDETRMedium):
            raise RuntimeError("Checkpoint is not RF-DETR Medium")
        if self.expected_class_names is not None and list(model.class_names) != self.expected_class_names:
            raise RuntimeError("Checkpoint class names differ from the exported class metadata")
        model.inference(compile=False, dtype=torch.bfloat16)
        self.model = model

    def detect(self, image):
        if not self.lock.acquire(blocking=False):
            raise HTTPException(503, "Detector busy; submit a fresh frame later")
        try:
            self.load()
            result = self.model.predict(image, threshold=self.threshold, include_source_image=False)
            width, height = image.size
            boxes = []
            for box, score, name in zip(result.xyxy, result.confidence, result.data["class_name"], strict=True):
                x0, y0, x1, y1 = map(float, box)
                score = float(score)
                if not all(math.isfinite(v) for v in (x0, y0, x1, y1, score)):
                    raise RuntimeError("Detector returned non-finite predictions")
                if name == "__background__" or score < self.threshold:
                    continue
                normalized = [max(0., min(1., x0 / width)), max(0., min(1., y0 / height)),
                              max(0., min(1., x1 / width)), max(0., min(1., y1 / height))]
                if normalized[2] <= normalized[0] or normalized[3] <= normalized[1]:
                    continue
                boxes.append({"bbox": normalized, "label": str(name), "confidence": score})
            boxes.sort(key=lambda item: item["confidence"], reverse=True)
            return {"boxes": boxes[:100], "width": width, "height": height,
                    "status": "ok", "checkpoint_kind": "fine_tuned" if self.checkpoint else "pretrained_coco"}
        finally:
            self.lock.release()


def create_app(detector=None):
    service = detector or Detector()
    application = FastAPI(title="RF-DETR Medium worker")
    application.state.detector = service

    @application.get("/health")
    def health():
        return {"service": "rf-detr-medium", "loaded": service.model is not None,
                "checkpoint_kind": "fine_tuned" if service.checkpoint else "pretrained_coco",
                "class_names": list(getattr(service.model, "class_names", []))}

    @application.post("/detect")
    def detect(request: DetectRequest):
        try:
            frame = decode_image(request.image_base64)
        except (ValueError, Image.DecompressionBombError) as error:
            raise HTTPException(422, str(error)) from error
        try:
            return service.detect(frame)
        except HTTPException:
            raise
        except Exception as error:
            logging.error("Detector inference failed (%s)", type(error).__name__)
            raise HTTPException(503, "Detector unavailable; no detection result") from error

    return application


app = create_app()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--checkpoint", type=Path, help="Own fine-tuned .pth; omit for clearly marked COCO baseline")
    parser.add_argument("--class-names", type=Path, help="Exported class_names.json; require exact checkpoint label order")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--preload", action="store_true", help="Load weights before serving; failure exits")
    args = parser.parse_args()
    if args.class_names and not args.checkpoint:
        parser.error("--class-names requires --checkpoint")
    service = Detector(args.checkpoint, args.threshold, read_class_names(args.class_names) if args.class_names else None)
    if args.preload:
        service.load()
    import uvicorn
    uvicorn.run(create_app(service), host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
