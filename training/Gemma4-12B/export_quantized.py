"""Create and verify an NF4 serving copy of the completed merged trained Gemma.

The full checkpoint stays intact. Use --prepare-only to stage HF shards on CPU
while another GPU task runs; omit it after that GPU task has finished.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from adapter_io import load_full_pth_model, read_pth_metadata, read_quantized_manifest, sha256
from contract import MODEL_ID, MODEL_REVISION, validate_assessment
from train import DeadlineBudget, messages_for_row, validate_row


def file_hashes(directory):
    return {path.relative_to(directory).as_posix(): sha256(path) for path in sorted(directory.rglob("*"))
            if path.is_file() and path.name not in ("prepared.json", "quantization_manifest.json", "deadline_status.json")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--intermediate", type=Path, required=True, help="HF BF16 staging directory; /dev/shm avoids filling pod disk")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation", type=Path, help="Held-out hazard JSONL; required for quantize/reload smoke")
    parser.add_argument("--prepare-only", action="store_true", help="CPU-only full checkpoint to HF shards, no CUDA model load")
    parser.add_argument("--deadline-utc", required=True)
    parser.add_argument("--max-wall-seconds", type=float, default=1800)
    args = parser.parse_args(argv)
    budget = DeadlineBudget(args.max_wall_seconds, args.deadline_utc, 60)
    budget.check_training()
    metadata = read_pth_metadata(args.checkpoint, MODEL_ID, MODEL_REVISION)
    if metadata["format"] != "gemma-full-state-dict-v1":
        raise ValueError("Quantization requires the actual trained full merged checkpoint")
    args.intermediate.mkdir(parents=True, exist_ok=True)
    budget.arm_hard_stop(args.intermediate)
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig
    started = time.perf_counter()
    prepared_path = args.intermediate / "prepared.json"
    if not prepared_path.exists():
        if any(args.intermediate.iterdir()):
            raise ValueError("Intermediate directory is nonempty without a completed preparation receipt")
        model = load_full_pth_model(args.checkpoint, MODEL_ID, MODEL_REVISION, device="cpu")
        model.save_pretrained(args.intermediate, safe_serialization=True, max_shard_size="2GB")
        processor_source = args.checkpoint.parent / metadata["processor_directory"]
        AutoProcessor.from_pretrained(processor_source, local_files_only=True).save_pretrained(args.intermediate)
        model.generation_config.save_pretrained(args.intermediate)
        del model
        gc.collect()
        prepared = {"completed": True, "source_checkpoint_sha256": metadata["sha256"],
                    "artifact_sha256": file_hashes(args.intermediate)}
        prepared_path.write_text(json.dumps(prepared, indent=2), encoding="utf-8")
    else:
        prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
        if prepared.get("completed") is not True or prepared.get("source_checkpoint_sha256") != metadata["sha256"]:
            raise ValueError("Prepared HF model belongs to a different source checkpoint")
        if file_hashes(args.intermediate) != prepared["artifact_sha256"]:
            raise ValueError("Prepared HF model checksum mismatch")
    if args.prepare_only:
        budget.cancel()
        print(json.dumps({"status": "prepared_on_cpu", "intermediate": str(args.intermediate),
                          "elapsed_seconds": time.perf_counter() - started, "gpu_used": False}))
        return 0
    if args.validation is None:
        raise ValueError("--validation is required for the quantized reload smoke")
    if args.output.exists():
        raise ValueError("Use a new quantized output directory")
    budget.check_training()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 GPU is required for quantization and verification")
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=["lm_head", "model.embed_vision", "model.embed_audio"])
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForMultimodalLM.from_pretrained(args.intermediate, local_files_only=True,
        dtype=torch.bfloat16, device_map={"": "cuda:0"}, attn_implementation="sdpa", quantization_config=quantization)
    if not getattr(model, "is_loaded_in_4bit", False):
        raise RuntimeError("Model did not enter bitsandbytes 4-bit mode")
    model.save_pretrained(args.output, safe_serialization=True, max_shard_size="2GB")
    AutoProcessor.from_pretrained(args.intermediate, local_files_only=True).save_pretrained(args.output)
    model.generation_config.save_pretrained(args.output)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    budget.check_training()
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    model = AutoModelForMultimodalLM.from_pretrained(args.output, local_files_only=True,
        dtype=torch.bfloat16, device_map={"": "cuda:0"}, attn_implementation="sdpa")
    if not getattr(model, "is_loaded_in_4bit", False):
        raise RuntimeError("Saved quantized model failed to reload as 4-bit")
    model.eval()
    processor = AutoProcessor.from_pretrained(args.output, local_files_only=True)
    load_seconds = time.perf_counter() - load_started
    row = validate_row(json.loads(next(line for line in args.validation.read_text().splitlines() if line.strip())),
                       args.validation.resolve(), "validation", allow_weak_labels=True)
    _, messages = messages_for_row(row)
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False,
        tokenize=True, return_dict=True, return_tensors="pt").to(model.device)
    length = inputs["input_ids"].shape[-1]
    if length > 8192:
        raise ValueError("Smoke input exceeds serving token budget")
    generation_started = time.perf_counter()
    with torch.inference_mode():
        tokens = model.generate(**inputs, max_new_tokens=384, do_sample=False, use_cache=True)
    raw = processor.decode(tokens[0, length:], skip_special_tokens=True)
    generation_seconds = time.perf_counter() - generation_started
    answer = validate_assessment(raw, frame_count=len(row["frames"]), track_ids={box["track_id"] for box in row["objects"]})
    verification = {"status": "passed", "checkpoint_reloaded": True, "loaded_in_4bit": True,
        "load_seconds": load_seconds, "generation_seconds": generation_seconds, "row_id": row["id"],
        "assessment": answer.model_dump(), "gpu": torch.cuda.get_device_name(0),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024 ** 3,
        "model_memory_bytes": model.get_memory_footprint(), "accuracy_evaluated": False,
        "limitation": "One-example load/schema smoke; quantization quality and deployment latency need separate evaluation"}
    (args.output / "quantization_verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    manifest = {"format": "gemma-bnb4-hf-v1", "completed": True, "base_model_id": MODEL_ID,
        "base_model_revision": MODEL_REVISION, "source_full_checkpoint_sha256": metadata["sha256"],
        "training_manifest": metadata["training_manifest"], "artifact_sha256": file_hashes(args.output),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "quantization": {"method": "bitsandbytes", "bits": 4, "quant_type": "nf4", "double_quant": True,
                         "compute_dtype": "bfloat16", "bitsandbytes_version": __import__("bitsandbytes").__version__},
        "elapsed_seconds": time.perf_counter() - started,
        "file_bytes": sum(path.stat().st_size for path in args.output.rglob("*") if path.is_file())}
    (args.output / "quantization_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    read_quantized_manifest(args.output, MODEL_ID, MODEL_REVISION)
    budget.cancel()
    print(json.dumps({"output": str(args.output), "file_bytes": manifest["file_bytes"], "verification": verification}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
