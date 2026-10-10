"""Gemma 4 12B outdoor hazard worker. Sampled frames in, validated hazard JSON out."""
from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import io
import logging
from pathlib import Path
import sys
import threading
from typing import Literal

from fastapi import FastAPI, HTTPException
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import Field, model_validator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from contract import (Assessment, Box, Contract, MODEL_ID, MODEL_REVISION, CONTRACT_VERSION,
                      build_messages, load_json, unknown_assessment, validate_assessment)
from adapter_io import read_pth_metadata, load_pth_adapter, load_full_pth_model, read_quantized_manifest

LOGGER = logging.getLogger("gemma_worker")


class Frame(Contract):
    image_base64: str = Field(min_length=1, max_length=6_000_000)
    timestamp_ms: float = Field(ge=0)


class AssessRequest(Contract):
    frames: list[Frame] = Field(min_length=1, max_length=8)
    boxes: list[Box] = Field(max_length=100)
    action_mode: Literal["video", "walking"] = "video"

    @model_validator(mode="after")
    def consistent(self):
        if any(a.timestamp_ms >= b.timestamp_ms for a, b in zip(self.frames, self.frames[1:])):
            raise ValueError("Frames must have strictly increasing timestamps")
        if len({box.track_id for box in self.boxes}) != len(self.boxes):
            raise ValueError("Track IDs must be unique")
        return self


def decode_image(encoded: str) -> Image.Image:
    if encoded.startswith("data:"):
        header, separator, encoded = encoded.partition(",")
        if not separator or header not in ("data:image/jpeg;base64", "data:image/png;base64", "data:image/webp;base64"):
            raise ValueError("Unsupported image data URL")
    try:
        raw = base64.b64decode(encoded, validate=True)
        if len(raw) > 4_500_000:
            raise ValueError("Image exceeds size limit")
        with Image.open(io.BytesIO(raw)) as image:
            if image.format not in ("JPEG", "PNG", "WEBP") or image.width * image.height > 12_000_000:
                raise ValueError("Unsupported image format or too many pixels")
            image.load()
            result = ImageOps.exif_transpose(image).convert("RGB")
            result.thumbnail((512, 512), Image.Resampling.LANCZOS)
            return result
    except (binascii.Error, UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError("Invalid image") from error


def adapter_manifest(path: Path, model_id: str, revision: str) -> dict:
    manifest = load_json((path / "training_manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != "gemma-outdoor-adapter-v1" or manifest.get("completed") is not True
            or manifest.get("model_id") != model_id or manifest.get("model_revision") != revision
            or manifest.get("output_contract_version") != CONTRACT_VERSION):
        raise ValueError("Adapter manifest does not match this model and output contract")
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        digest = hashlib.sha256((path / filename).read_bytes()).hexdigest()
        if manifest.get("artifact_sha256", {}).get(filename) != digest:
            raise ValueError("Adapter artifact hash mismatch")
    return manifest


class GemmaRuntime:
    def __init__(self, model_id=MODEL_ID, revision=MODEL_REVISION, adapter=None, max_input_tokens=8192, max_new_tokens=768):
        self.model_id, self.revision = model_id, revision
        self.adapter = Path(adapter).resolve() if adapter else None
        # A downloader may supply a local snapshot path, but adapter provenance
        # always names the canonical Hub model rather than a machine-local path.
        if self.adapter and self.adapter.is_dir() and (self.adapter / "quantization_manifest.json").exists():
            self.checkpoint_metadata = read_quantized_manifest(self.adapter, MODEL_ID, revision)
            self.manifest = self.checkpoint_metadata["training_manifest"]
        elif self.adapter and self.adapter.is_file():
            self.checkpoint_metadata = read_pth_metadata(self.adapter, MODEL_ID, revision)
            self.manifest = self.checkpoint_metadata["training_manifest"]
        else:
            self.checkpoint_metadata = {}
            self.manifest = adapter_manifest(self.adapter, MODEL_ID, revision) if self.adapter else {}
        self.max_input_tokens = max_input_tokens
        if not 1 <= max_new_tokens <= 768:
            raise ValueError("max_new_tokens must be between 1 and 768")
        self.max_new_tokens = max_new_tokens
        self.model = self.processor = None
        self.last_error = None

    @property
    def loaded(self):
        return self.model is not None and self.processor is not None

    def load(self):
        if self.loaded:
            return
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA with BF16 support is required")
        quantized_checkpoint = self.checkpoint_metadata.get("format") == "gemma-bnb4-hf-v1"
        processor = AutoProcessor.from_pretrained(str(self.adapter) if quantized_checkpoint else self.model_id, revision=self.revision)
        full_checkpoint = self.checkpoint_metadata.get("format") == "gemma-full-state-dict-v1"
        if quantized_checkpoint:
            model = AutoModelForMultimodalLM.from_pretrained(str(self.adapter), local_files_only=True,
                dtype=torch.bfloat16, device_map={"": "cuda:0"}, attn_implementation="sdpa")
            if not getattr(model, "is_loaded_in_4bit", False):
                raise RuntimeError("Quantized serving artifact was not loaded in 4-bit mode")
        elif full_checkpoint:
            model = load_full_pth_model(self.adapter, MODEL_ID, self.revision)
        else:
            model = AutoModelForMultimodalLM.from_pretrained(
                self.model_id, revision=self.revision, dtype=torch.bfloat16,
                device_map={"": "cuda:0"}, attn_implementation="sdpa")
        if self.adapter and not full_checkpoint and not quantized_checkpoint:
            if self.adapter.is_file():
                model = load_pth_adapter(model, self.adapter, MODEL_ID, self.revision)
            else:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, str(self.adapter), is_trainable=False)
        model.eval()
        self.processor, self.model = processor, model

    def generate(self, messages):
        self.load()
        import torch
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, enable_thinking=False,
            tokenize=True, return_dict=True, return_tensors="pt")
        input_length = inputs["input_ids"].shape[-1]
        if input_length > self.max_input_tokens:
            raise ValueError("Visual request exceeds input token budget")
        inputs = inputs.to(self.model.device)
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False, use_cache=True)
        # Thinking is disabled and its empty channel prefix is already in the prompt.
        return self.processor.decode(outputs[0, input_length:], skip_special_tokens=True)


def create_app(runtime=None):
    engine = runtime or GemmaRuntime()
    app = FastAPI(title="Gemma outdoor hazard worker", version="1.0")
    app.state.runtime = engine
    lock = threading.Lock()
    app.state.inference_lock = lock

    @app.get("/health")
    def health():
        manifest = getattr(engine, "manifest", {})
        return {"service": "gemma", "model": engine.model_id, "revision": engine.revision,
                "loaded": engine.loaded, "busy": lock.locked(), "last_error": engine.last_error,
                "load_error": None if engine.loaded else engine.last_error,
                "adapter_enabled": bool(getattr(engine, "adapter", None)),
                "trained_for_contract": manifest.get("trained_for_contract", False),
                "trained_for_verified_hazards": manifest.get("trained_for_verified_hazards", False),
                "supervision_tasks": manifest.get("supervision_tasks", []),
                "supervision_quality": manifest.get("supervision_quality", "none"),
                "label_policy_ids": manifest.get("label_policy_ids", []),
                "checkpoint_format": getattr(engine, "checkpoint_metadata", {}).get("format", "adapter" if getattr(engine, "adapter", None) else "base_model"),
                "output_contract_version": CONTRACT_VERSION}

    @app.post("/assess", response_model=Assessment)
    def assess(body: AssessRequest):
        if not lock.acquire(blocking=False):
            raise HTTPException(503, "Gemma worker is busy")
        try:
            try:
                images = [decode_image(frame.image_base64) for frame in body.frames]
                messages = build_messages(images, [frame.timestamp_ms for frame in body.frames],
                                          [box.model_dump() for box in body.boxes], body.action_mode)
            except ValueError as error:
                raise HTTPException(422, str(error)) from error
            try:
                raw = engine.generate(messages)
            except ValueError as error:
                raise HTTPException(422, str(error)) from error
            except Exception:
                engine.last_error = "model_inference_failed"
                LOGGER.error("Gemma inference failed; request data and exception details omitted")
                raise HTTPException(503, "Gemma inference unavailable") from None
            try:
                answer = validate_assessment(raw, frame_count=len(images),
                                             track_ids={box.track_id for box in body.boxes})
                engine.last_error = None
                return answer
            except (ValueError, TypeError):
                engine.last_error = "invalid_model_output"
                return unknown_assessment()
        finally:
            lock.release()

    return app


app = create_app()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--adapter", type=Path, help="Completed LoRA adapter directory or gemma_model.pth; omitted means pretrained baseline")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--preload", action="store_true")
    args = parser.parse_args()
    app = create_app(GemmaRuntime(args.model, args.revision, args.adapter))
    if args.preload:
        app.state.runtime.load()
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, workers=1)
