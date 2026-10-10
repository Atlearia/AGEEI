"""Export actual merged Gemma weights to a full .pth state_dict checkpoint.

Requires ~24 GB output space plus working memory. Processor assets are saved
separately. This is distinct from the small adapter-only .pth produced by train.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from contract import MODEL_ID, MODEL_REVISION
from adapter_io import sha256


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True, help="Completed safetensors adapter directory")
    parser.add_argument("--model-path", required=True, help="Local pinned base snapshot")
    parser.add_argument("--output", type=Path, required=True, help="Full .pth checkpoint destination, preferably on roomy storage")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda:0"])
    parser.add_argument("--deadline-utc", help="ISO-8601 absolute deadline; cancel before it rather than run indefinitely")
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise ValueError("Use a fresh full checkpoint output path")
    manifest = json.loads((args.adapter / "training_manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("completed") is not True or manifest.get("model_id") != MODEL_ID
            or manifest.get("model_revision") != MODEL_REVISION):
        raise ValueError("Only a completed adapter for the pinned model can be merged")
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        if sha256(args.adapter / name) != manifest["artifact_sha256"][name]:
            raise ValueError("Adapter checksum mismatch")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Reuse the explicit hard deadline, with five minutes reserved by the caller
    # for transfer/verification. The full checkpoint is atomically promoted.
    from train import DeadlineBudget
    budget = DeadlineBudget(1800, args.deadline_utc, 60)
    budget.arm_hard_stop(args.output.parent)
    budget.check_training()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForMultimodalLM, AutoProcessor
    start = time.perf_counter()
    base = AutoModelForMultimodalLM.from_pretrained(args.model_path, revision=MODEL_REVISION,
        dtype=torch.bfloat16, device_map={"": args.device}, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(base, str(args.adapter), is_trainable=False)
    model = model.merge_and_unload(safe_merge=True)
    if any("lora_" in name for name, _ in model.named_parameters()):
        raise RuntimeError("Adapter merge left LoRA parameters behind")
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    payload = {"format": "gemma-full-state-dict-v1", "base_model_id": MODEL_ID,
               "base_model_revision": MODEL_REVISION, "config": model.config.to_dict(),
               "generation_config": model.generation_config.to_dict(),
               "training_manifest": manifest, "state_dict": state}
    temporary = args.output.with_suffix(".pth.tmp")
    torch.save(payload, temporary)
    temporary.replace(args.output)
    assets = args.output.parent / (args.output.stem + "_processor")
    AutoProcessor.from_pretrained(args.model_path, revision=MODEL_REVISION).save_pretrained(assets)
    model.config.save_pretrained(assets)
    model.generation_config.save_pretrained(assets)
    metadata = {key: value for key, value in payload.items() if key not in ("state_dict", "config", "generation_config")}
    metadata.update(sha256=sha256(args.output), file_bytes=args.output.stat().st_size,
                    tensors=len(state), tensor_elements=sum(t.numel() for t in state.values()),
                    requires_base_model=False, artifact_kind="Full merged Gemma state_dict",
                    processor_directory=assets.name, export_seconds=time.perf_counter() - start)
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    budget.cancel()
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
