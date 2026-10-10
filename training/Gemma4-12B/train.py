"""Validate local multimodal manifests and train a bounded Gemma 4 task LoRA.

Source QA is auxiliary. Hazard targets use the shared serving prompt; documented
AI-generated labels require explicit weak-supervision opt-in and remain unverified.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import threading
import time

from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from contract import (Box, CONTRACT_VERSION, MODEL_ID, MODEL_REVISION, build_messages,
                      load_json, validate_assessment)
from adapter_io import export_pth

DEFAULT_CONFIG = Path(__file__).with_name("configs") / "h100_lora.json"


class BudgetStop(RuntimeError):
    """End training early while leaving time to validate and export the adapter."""


class DeadlineBudget:
    def __init__(self, max_seconds=7200, deadline_utc=None, reserve_seconds=300):
        if not math.isfinite(max_seconds) or max_seconds <= 0 or reserve_seconds < 60:
            raise ValueError("Budget must be positive and reserve at least 60 seconds")
        now = time.time()
        self.deadline = now + max_seconds
        if deadline_utc:
            parsed = datetime.fromisoformat(deadline_utc.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                raise ValueError("deadline-utc needs an explicit timezone")
            self.deadline = min(self.deadline, parsed.timestamp())
        self.reserve_seconds = reserve_seconds
        self.timer = None

    def remaining(self):
        return self.deadline - time.time()

    def check_training(self):
        if self.remaining() <= self.reserve_seconds:
            raise BudgetStop("Training deadline reserve reached; exporting current adapter")

    def arm_hard_stop(self, output):
        def stop():
            try:
                (output / "deadline_status.json").write_text(json.dumps({
                    "status": "hard_deadline_reached", "completed_export": False,
                    "message": "Process stopped at user deadline. Previously saved checkpoints are retained."}), encoding="utf-8")
            finally:
                # The hard limit also covers a blocked CUDA call. The soft limit
                # normally leaves five minutes for final validation and export.
                os._exit(124)
        self.timer = threading.Timer(max(0, self.remaining()), stop)
        self.timer.daemon = True
        self.timer.start()

    def cancel(self):
        if self.timer:
            self.timer.cancel()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rgb(path: Path):
    with Image.open(path) as image:
        if image.width * image.height > 12_000_000 or image.format not in ("JPEG", "PNG", "WEBP"):
            raise ValueError(f"Unsupported image: {path.name}")
        image.load()
        result = ImageOps.exif_transpose(image).convert("RGB")
        result.thumbnail((512, 512), Image.Resampling.LANCZOS)
        return result


def validate_row(row: dict, manifest_path: Path, expected_split: str, *, allow_weak_labels=False) -> dict:
    if not isinstance(row, dict):
        raise ValueError("A manifest row must be an object")
    for key in ("id", "source", "source_id", "sequence_id"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise ValueError(f"Missing nonempty {key}")
    if row.get("split") != expected_split:
        raise ValueError(f"Row {row['id']} split must be {expected_split}")
    provenance = row.get("provenance", {})
    if (not isinstance(provenance, dict) or provenance.get("license_verified") is not True
            or not provenance.get("license") or not provenance.get("source_url")):
        raise ValueError("Require provenance source_url, license and license_verified=true")
    frames = row.get("frames")
    if not isinstance(frames, list) or not 1 <= len(frames) <= 8:
        raise ValueError("Require 1-8 frames")
    result = dict(row)
    result["frames"] = []
    times = []
    for frame in frames:
        if not isinstance(frame, dict) or not isinstance(frame.get("path"), str):
            raise ValueError("Frame needs a local path")
        t = frame.get("timestamp_ms")
        if isinstance(t, bool) or not isinstance(t, (float, int)) or not math.isfinite(t) or t < 0:
            raise ValueError("Invalid timestamp")
        times.append(t)
        image_path = Path(frame["path"])
        if not image_path.is_absolute():
            image_path = manifest_path.parent / image_path
        image_path = image_path.resolve(strict=True)
        if not image_path.is_file():
            raise ValueError("Frame must be a regular file")
        result["frames"].append({"path": str(image_path), "timestamp_ms": t})
    if any(a >= b for a, b in zip(times, times[1:])):
        raise ValueError("Timestamps must be strictly increasing")
    if row.get("action_mode") not in ("walking", "video"):
        raise ValueError("action_mode must be walking or video")
    objects = row.get("objects")
    if not isinstance(objects, list) or len(objects) > 100:
        raise ValueError("objects must be a list of at most 100 newest-frame detections")
    boxes = [Box.model_validate(box, strict=True) for box in objects]
    if len({box.track_id for box in boxes}) != len(boxes):
        raise ValueError("Duplicate track IDs")
    if row.get("task") == "hazard_assessment":
        target = validate_assessment(row.get("target"), frame_count=len(frames),
                                     track_ids={box.track_id for box in boxes})
        weak = (provenance.get("hazard_labels_verified") is False and provenance.get("supervision") == "weak"
                and isinstance(provenance.get("label_policy_id"), str) and bool(provenance["label_policy_id"])
                and type(provenance.get("label_policy_version")) is int and provenance["label_policy_version"] > 0)
        if provenance.get("hazard_labels_verified") is not True and not (allow_weak_labels and weak):
            raise ValueError("Hazard targets require verified source hazard labels, not pseudo-labels")
        result["target"] = target.model_dump()
    elif row.get("task") == "source_qa":
        if not isinstance(row.get("question"), str) or not 1 <= len(row["question"].strip()) <= 3000:
            raise ValueError("source_qa needs its original question")
        if not isinstance(row.get("target"), str) or not 1 <= len(row["target"].strip()) <= 6000:
            raise ValueError("source_qa needs its original answer")
    else:
        raise ValueError("Only source_qa and hazard_assessment are supported")
    target_text = result["target"] if isinstance(result["target"], str) else json.dumps(result["target"], allow_nan=False)
    if "<|" in target_text or "|>" in target_text:
        raise ValueError("Target cannot contain chat control tokens")
    return result


def validate_manifests(train: Path, validation: Path, *, allow_weak_labels=False):
    splits = {}
    ids, sequences, pixels = {}, {}, {}
    fingerprint_cache = {}
    for split, path in (("train", train), ("validation", validation)):
        report_path = path.parent / "data_report.json"
        if report_path.exists() and load_json(report_path.read_text(encoding="utf-8")).get("status") != "prepared":
            raise ValueError("Data preparation report is not complete; refuse partially prepared data")
        rows = []
        for number, text in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not text.strip():
                continue
            try:
                row = validate_row(load_json(text), path, split, allow_weak_labels=allow_weak_labels)
                if row["id"] in ids:
                    raise ValueError("Duplicate example ID")
                ids[row["id"]] = split
                group = (row["source"], row["sequence_id"])
                if group in sequences and sequences[group] != split:
                    raise ValueError("Sequence leakage across train/validation")
                sequences[group] = split
                for frame in row["frames"]:
                    image_path = frame["path"]
                    if image_path not in fingerprint_cache:
                        image = load_rgb(Path(image_path))
                        fingerprint_cache[image_path] = hashlib.sha256(
                            repr(image.size).encode() + image.tobytes()).hexdigest()
                    fingerprint = fingerprint_cache[image_path]
                    if fingerprint in pixels and pixels[fingerprint] != split:
                        raise ValueError("Identical decoded/resized frame across train/validation")
                    pixels[fingerprint] = split
                rows.append(row)
            except Exception as error:
                raise ValueError(f"{path.name}:{number}: {error}") from error
        if not rows:
            raise ValueError(f"Empty {split} manifest")
        splits[split] = rows
    return splits


def messages_for_row(row):
    images = [load_rgb(Path(frame["path"])) for frame in row["frames"]]
    messages = build_messages(images, [frame["timestamp_ms"] for frame in row["frames"]],
                              row["objects"], row["action_mode"],
                              question=row.get("question") if row["task"] == "source_qa" else None)
    return images, messages


def assistant_labels(full_ids: list[int], prefix_ids: list[int], max_target_tokens: int) -> list[int]:
    """Mask every prompt/vision token; fail closed if tokenization moves the boundary."""
    if full_ids[:len(prefix_ids)] != prefix_ids:
        raise ValueError("Assistant boundary is not an exact token prefix; cannot safely mask loss")
    remaining = len(full_ids) - len(prefix_ids)
    if not 1 <= remaining <= max_target_tokens:
        raise ValueError("Assistant answer is empty or exceeds the target token budget")
    return [-100] * len(prefix_ids) + full_ids[len(prefix_ids):]


class MultimodalCollator:
    def __init__(self, processor, config):
        self.processor, self.config = processor, config

    def __call__(self, rows):
        # One visual example per device; accumulation controls effective batch size.
        if len(rows) != 1:
            raise ValueError("Only microbatch 1 is supported by this variable-image collator")
        row = rows[0]
        images, messages = messages_for_row(row)
        prefix = self.processor.apply_chat_template(messages, tokenize=False,
                                                      add_generation_prompt=True, enable_thinking=False)
        target = row["target"] if isinstance(row["target"], str) else json.dumps(row["target"], allow_nan=False)
        # Pinned Gemma canonical template closes an assistant turn with <turn|>.
        ending = "<turn|>"
        token_id = self.processor.tokenizer.convert_tokens_to_ids(ending)
        if token_id is None or token_id == self.processor.tokenizer.unk_token_id:
            raise ValueError("Pinned assistant turn terminator missing from tokenizer")
        kwargs = dict(images=[images], return_tensors="pt", padding=False, add_special_tokens=False)
        prefix_batch = self.processor(text=[prefix], **kwargs)
        full = self.processor(text=[prefix + target + ending], **kwargs)
        full_ids = full["input_ids"][0].tolist()
        if len(full_ids) > self.config["max_sequence_tokens"]:
            raise ValueError(f"Example {row['id']} exceeds token budget; no silent image/answer truncation")
        labels = assistant_labels(full_ids, prefix_batch["input_ids"][0].tolist(), self.config["max_target_tokens"])
        import torch
        full["labels"] = torch.tensor([labels], dtype=torch.long)
        return full


def validate_config(config):
    if config.get("model_id") != MODEL_ID or config.get("model_revision") != MODEL_REVISION:
        raise ValueError("This trainer requires the verified Gemma model revision")
    for key in ("lora_rank", "lora_alpha", "gradient_accumulation_steps", "max_steps", "max_sequence_tokens", "max_target_tokens", "eval_steps", "save_steps"):
        if type(config.get(key)) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if config["max_sequence_tokens"] > 8192 or config["max_target_tokens"] > 768:
        raise ValueError("Training token budget cannot exceed serving budget")
    if config["save_steps"] % config["eval_steps"]:
        raise ValueError("save_steps must be a multiple of eval_steps")
    re.compile(config["target_modules_regex"])
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--model-path", help="Optional local snapshot of the pinned model; revision provenance stays canonical")
    parser.add_argument("--allow-weak-labels", action="store_true", help="Explicitly allow documented weak teacher/policy hazard targets")
    parser.add_argument("--max-wall-seconds", type=float, default=7200)
    parser.add_argument("--deadline-utc", help="Absolute ISO-8601 deadline; the earlier time limit wins")
    parser.add_argument("--export-reserve-seconds", type=float, default=300)
    parser.add_argument("--pilot", action="store_true", help="20 steps and at most 32 validation rows; no production adapter promotion")
    parser.add_argument("--dry-run", action="store_true", help="Validate local files, sources and leakage without downloading a model")
    parser.add_argument("--processor-only", action="store_true", help="Also verify real tokenization and loss masks on CPU; download processor assets only")
    args = parser.parse_args(argv)
    budget = DeadlineBudget(args.max_wall_seconds, args.deadline_utc, args.export_reserve_seconds)
    config = load_json(args.config.read_text(encoding="utf-8"))
    if args.max_steps is not None:
        config["max_steps"] = args.max_steps
    if args.pilot:
        config.update(max_steps=20, eval_steps=10, save_steps=10)
    validate_config(config)
    splits = validate_manifests(args.train.resolve(), args.validation.resolve(), allow_weak_labels=args.allow_weak_labels)
    report = {"model_id": MODEL_ID, "model_revision": MODEL_REVISION,
              "examples": {split: len(rows) for split, rows in splits.items()},
              "tasks": {split: dict(Counter(row["task"] for row in rows)) for split, rows in splits.items()},
              "source_manifest_sha256": {"train": file_sha256(args.train), "validation": file_sha256(args.validation)},
              "config": config, "is_pilot": args.pilot, "allow_weak_labels": args.allow_weak_labels,
              "deadline_utc": datetime.fromtimestamp(budget.deadline, timezone.utc).isoformat()}
    print(json.dumps(report, indent=2))
    if args.dry_run:
        print("Local data checks passed. Processor masks and GPU execution have not been tested.")
        return 0
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Output must be new or empty; use a fresh run directory")
    args.output.mkdir(parents=True, exist_ok=True)
    budget.arm_hard_stop(args.output)
    (args.output / "run_plan.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    import torch
    from transformers import AutoProcessor
    if args.pilot:
        splits["validation"] = splits["validation"][:32]
    budget.check_training()
    model_source = args.model_path or MODEL_ID
    processor = AutoProcessor.from_pretrained(model_source, revision=MODEL_REVISION)
    collator = MultimodalCollator(processor, config)
    # Preflight all targets before allocating model weights on GPU. This validates
    # real processor image expansion and assistant-only loss masking, not a mock.
    lengths = []
    for rows in splits.values():
        for row in rows:
            budget.check_training()
            encoded = collator([row])
            lengths.append(encoded["input_ids"].shape[-1])
    (args.output / "processor_preflight.json").write_text(json.dumps({
        "examples": len(lengths), "max_tokens": max(lengths), "assistant_masks_verified": True,
        "truncation": False}), encoding="utf-8")
    if args.processor_only:
        budget.cancel()
        print("Real processor image expansion and assistant loss masks verified; no model weights loaded.")
        return 0
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This training recipe requires a CUDA BF16 GPU, planned for H100 80 GB")
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForMultimodalLM, Trainer, TrainingArguments, TrainerCallback
    budget.check_training()
    model = AutoModelForMultimodalLM.from_pretrained(model_source, revision=MODEL_REVISION,
        dtype=torch.bfloat16, device_map={"": "cuda:0"}, attn_implementation="sdpa")
    targets = [name for name, module in model.named_modules()
               if re.fullmatch(config["target_modules_regex"], name) and isinstance(module, torch.nn.Linear)]
    if not targets:
        raise RuntimeError("No verified language attention projections found for LoRA")
    model = get_peft_model(model, LoraConfig(r=config["lora_rank"], lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"], target_modules=targets, bias="none", task_type="CAUSAL_LM"))
    # PEFT otherwise records a machine-specific local snapshot directory.
    model.peft_config["default"].base_model_name_or_path = MODEL_ID
    model.peft_config["default"].revision = MODEL_REVISION
    model.config.use_cache = False
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False
    model.enable_input_require_grads()
    trainable = [(name, parameter.numel()) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any("lora_" not in name for name, _ in trainable):
        raise RuntimeError("Expected only LoRA parameters to be trainable")
    (args.output / "trainable_parameters.json").write_text(json.dumps({
        "count": sum(count for _, count in trainable), "names": [name for name, _ in trainable],
        "target_modules": targets}, indent=2), encoding="utf-8")
    training_args = TrainingArguments(output_dir=str(args.output), per_device_train_batch_size=1,
        per_device_eval_batch_size=1, gradient_accumulation_steps=config["gradient_accumulation_steps"],
        max_steps=config["max_steps"], learning_rate=config["learning_rate"], weight_decay=config["weight_decay"],
        warmup_steps=math.ceil(config["max_steps"] * config["warmup_ratio"]), lr_scheduler_type="cosine", bf16=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps", eval_steps=config["eval_steps"], save_strategy="steps", save_steps=config["save_steps"],
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        prediction_loss_only=True, remove_unused_columns=False, dataloader_num_workers=0,
        save_total_limit=2, logging_steps=1, report_to=["tensorboard"], seed=config["seed"],
        optim="adamw_torch", push_to_hub=False)
    class BudgetCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if budget.remaining() <= budget.reserve_seconds:
                control.should_training_stop = True
                control.should_evaluate = False
                control.should_save = True
            return control

    class BudgetTrainer(Trainer):
        def training_step(self, *args, **kwargs):
            budget.check_training()
            return super().training_step(*args, **kwargs)

        def prediction_step(self, *args, **kwargs):
            if budget.remaining() <= 60:
                raise BudgetStop("Reserving final minute for adapter export")
            return super().prediction_step(*args, **kwargs)

    trainer = BudgetTrainer(model=model, args=training_args, train_dataset=splits["train"],
                           eval_dataset=splits["validation"], data_collator=collator,
                           callbacks=[BudgetCallback()])
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    train_metrics = {}
    stop_reason = "max_steps"
    try:
        result = trainer.train()
        train_metrics = result.metrics
        if trainer.state.global_step < config["max_steps"]:
            stop_reason = "wall_clock_budget"
    except BudgetStop:
        stop_reason = "wall_clock_budget"
    # A bounded final sample leaves enough time to export under the hard limit.
    final_validation = splits["validation"] if budget.remaining() > 600 else splits["validation"][:32]
    metrics = {}
    if trainer.state.global_step > 0 and budget.remaining() > 60:
        try:
            metrics = trainer.evaluate(eval_dataset=final_validation)
        except BudgetStop:
            stop_reason = "wall_clock_budget_before_final_validation"
    elapsed = time.perf_counter() - started
    loss = float(metrics["eval_loss"]) if "eval_loss" in metrics else None
    if loss is not None and not math.isfinite(loss):
        loss = None
    complete = loss is not None and trainer.state.global_step > 0
    adapter = args.output / "adapter"
    model.save_pretrained(adapter, safe_serialization=True)
    tasks = sorted({row["task"] for row in splits["train"]})
    all_rows = splits["train"] + splits["validation"]
    weak = any(row["provenance"].get("supervision") == "weak" for row in all_rows)
    policies = sorted({row["provenance"]["label_policy_id"] for row in all_rows if row["provenance"].get("label_policy_id")})
    hazard_train = any(row["task"] == "hazard_assessment" for row in splits["train"])
    hazard_valid = any(row["task"] == "hazard_assessment" for row in splits["validation"])
    manifest = {"schema_version": "gemma-outdoor-adapter-v1", "completed": complete,
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "supervision_tasks": tasks,
        "steps": trainer.state.global_step, "train_examples": len(splits["train"]),
        "validation_examples": len(final_validation) if metrics else 0, "validation_loss": loss,
        "source_manifest_sha256": report["source_manifest_sha256"], "output_contract_version": CONTRACT_VERSION,
        "trained_for_contract": complete and hazard_train and hazard_valid and not args.pilot,
        "trained_for_verified_hazards": complete and hazard_train and hazard_valid and not weak and not args.pilot,
        "supervision_quality": "weak" if weak else "verified", "label_policy_ids": policies,
        "stop_reason": stop_reason,
        "is_pilot": args.pilot, "created_utc": datetime.now(timezone.utc).isoformat(),
        "selected_checkpoint": trainer.state.best_model_checkpoint,
        "best_validation_loss": trainer.state.best_metric,
        "artifact_sha256": {name: file_sha256(adapter / name) for name in ("adapter_config.json", "adapter_model.safetensors")},
        "training_scope": "language attention LoRA; base weights and multimodal projectors frozen",
        "validation_scope": "held-out assistant token loss; not hazard accuracy, grounding accuracy or severity calibration"}
    (adapter / "training_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pth_metadata = export_pth(model, adapter, args.output / "gemma_model.pth", manifest)
    timing = {"train_and_eval_seconds": elapsed, "steps": trainer.state.global_step,
              "seconds_per_step_including_eval": elapsed / trainer.state.global_step,
              "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
              "train_metrics": train_metrics, "validation_metrics": metrics,
              "stop_reason": stop_reason, "remaining_wall_seconds": budget.remaining(),
              "pth_sha256": pth_metadata["sha256"]}
    (args.output / "timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
    print(json.dumps({"adapter": str(adapter), "trained_for_contract": manifest["trained_for_contract"], **timing}, indent=2))
    budget.cancel()
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
