"""Portable .pth LoRA export: real tensors plus config, never standalone base weights."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_quantized_manifest(path: Path, model_id: str, revision: str):
    """Validate a standalone trained 4-bit Hugging Face artifact without GPU imports."""
    path = path.resolve()
    value = json.loads((path / "quantization_manifest.json").read_text(encoding="utf-8"))
    if (value.get("format") != "gemma-bnb4-hf-v1" or value.get("completed") is not True
            or value.get("base_model_id") != model_id or value.get("base_model_revision") != revision
            or value.get("training_manifest", {}).get("completed") is not True):
        raise ValueError("Quantized artifact identity or completion status does not match")
    hashes = value.get("artifact_sha256", {})
    required = {"config.json", "generation_config.json", "processor_config.json", "tokenizer.json", "tokenizer_config.json", "quantization_verification.json"}
    if not isinstance(hashes, dict) or not required.issubset(hashes):
        raise ValueError("Quantized artifact is missing hashed model/processor configuration")
    if not ({"chat_template.jinja", "chat_template.json"} & set(hashes)):
        raise ValueError("Quantized artifact is missing its hashed chat template")
    for name, digest in hashes.items():
        if not isinstance(name, str) or "\\" in name:
            raise ValueError("Invalid quantized artifact filename")
        candidate = (path / name).resolve()
        if not candidate.is_relative_to(path) or not candidate.is_file() or sha256(candidate) != digest:
            raise ValueError("Quantized artifact file missing or checksum mismatch")
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    verification = json.loads((path / "quantization_verification.json").read_text(encoding="utf-8"))
    if (verification.get("status") != "passed" or verification.get("checkpoint_reloaded") is not True
            or verification.get("loaded_in_4bit") is not True):
        raise ValueError("Quantized artifact has no successful serialized reload verification")
    quant = config.get("quantization_config", {})
    if (config.get("model_type") != "gemma4_unified" or quant.get("quant_method") != "bitsandbytes"
            or quant.get("load_in_4bit", quant.get("_load_in_4bit")) is not True
            or quant.get("bnb_4bit_quant_type") != "nf4" or quant.get("bnb_4bit_use_double_quant") is not True):
        raise ValueError("Expected Gemma4Unified bitsandbytes NF4 double quantization")
    index = path / "model.safetensors.index.json"
    if index.exists():
        if index.name not in hashes:
            raise ValueError("Quantized shard index is not hashed")
        shards = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
        if not shards or not shards.issubset(hashes):
            raise ValueError("Quantized model is missing a referenced shard")
    elif "model.safetensors" not in hashes:
        raise ValueError("Quantized model has no verified weight file")
    return value


def read_pth_metadata(path: Path, model_id: str, revision: str):
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if (metadata.get("format") not in ("gemma-lora-pth-v1", "gemma-full-state-dict-v1") or metadata.get("base_model_id") != model_id
            or metadata.get("base_model_revision") != revision or metadata.get("sha256") != sha256(path)):
        raise ValueError("LoRA .pth metadata, pinned base identity or file checksum does not match")
    manifest = metadata.get("training_manifest", {})
    if manifest.get("completed") is not True:
        raise ValueError("LoRA .pth training was not completed")
    return metadata


def export_pth(model, adapter_dir: Path, path: Path, manifest: dict):
    import torch
    from peft import get_peft_model_state_dict
    config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    state = {key: value.detach().cpu().contiguous() for key, value in get_peft_model_state_dict(model).items()}
    if not state or any("lora_" not in key for key in state):
        raise ValueError("Expected only LoRA adapter tensors in portable export")
    payload = {"format": "gemma-lora-pth-v1", "base_model_id": manifest["model_id"],
               "base_model_revision": manifest["model_revision"], "peft_config": config,
               "training_manifest": manifest, "state_dict": state}
    temporary = path.with_suffix(".pth.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    metadata = {key: value for key, value in payload.items() if key not in ("state_dict", "peft_config")}
    metadata.update(sha256=sha256(path), tensors=len(state), trainable_parameters=sum(t.numel() for t in state.values()),
                    requires_base_model=True, artifact_kind="LoRA adapter, not a standalone full model")
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def load_pth_adapter(base_model, path: Path, model_id: str, revision: str):
    import torch
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
    metadata = read_pth_metadata(path, model_id, revision)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (payload.get("format") != metadata["format"] or payload.get("base_model_id") != model_id
            or payload.get("base_model_revision") != revision
            or payload.get("training_manifest") != metadata["training_manifest"]):
        raise ValueError("LoRA .pth payload does not match its checked metadata")
    config = LoraConfig(**payload["peft_config"])
    config.inference_mode = True
    model = get_peft_model(base_model, config)
    expected = get_peft_model_state_dict(model)
    actual = payload["state_dict"]
    if set(actual) != set(expected) or any(not isinstance(t, torch.Tensor) or t.shape != expected[k].shape for k, t in actual.items()):
        raise ValueError("LoRA .pth tensor names or shapes do not match the pinned architecture")
    set_peft_model_state_dict(model, actual)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_full_pth_model(path: Path, model_id: str, revision: str, device="cuda:0"):
    """Reconstruct full merged weights without downloading a second base checkpoint."""
    import torch
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForMultimodalLM, GenerationConfig
    metadata = read_pth_metadata(path, model_id, revision)
    if metadata["format"] != "gemma-full-state-dict-v1":
        raise ValueError("Expected a full merged state_dict checkpoint")
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if (payload.get("format") != metadata["format"] or payload.get("base_model_id") != model_id
            or payload.get("base_model_revision") != revision
            or payload.get("training_manifest") != metadata["training_manifest"]):
        raise ValueError("Full checkpoint payload does not match checked metadata")
    config_dict = dict(payload["config"])
    model_type = config_dict.pop("model_type")
    config = AutoConfig.for_model(model_type, **config_dict)
    with init_empty_weights():
        model = AutoModelForMultimodalLM.from_config(config, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.load_state_dict(payload["state_dict"], strict=True, assign=True)
    model.tie_weights()
    if not isinstance(payload.get("generation_config"), dict):
        raise ValueError("Full checkpoint is missing its generation configuration")
    model.generation_config = GenerationConfig.from_dict(payload["generation_config"])
    model = model.to(device=device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model
