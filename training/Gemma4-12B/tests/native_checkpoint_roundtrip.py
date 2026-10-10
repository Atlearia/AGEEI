"""CPU smoke for native Gemma4Unified + PEFT .pth loading; tiny random fixture only."""
import argparse
import json
from pathlib import Path
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    helpers = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(helpers))
    from adapter_io import export_pth, load_pth_adapter, load_full_pth_model, sha256
    from contract import MODEL_ID, MODEL_REVISION
    import torch
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
    from transformers import AutoModelForMultimodalLM, Gemma4UnifiedConfig
    torch.set_num_threads(2)
    config = Gemma4UnifiedConfig(text_config={"vocab_size": 128, "hidden_size": 32, "intermediate_size": 64,
        "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
        "layer_types": ["sliding_attention", "sliding_attention"], "per_layer_config": {}},
        vision_config=None, audio_config=None)
    def make_base():
        return AutoModelForMultimodalLM.from_config(config, dtype=torch.bfloat16, attn_implementation="sdpa")
    model = get_peft_model(make_base(), LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"],
                                                  task_type="CAUSAL_LM", bias="none"))
    model.enable_input_require_grads()
    tokens = torch.tensor([[2, 5, 6, 7]])
    loss = model(input_ids=tokens, attention_mask=torch.ones_like(tokens), labels=tokens).loss
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in model.named_parameters() if "lora_" in n)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gemma-pth-fixture-", dir=args.receipt.parent) as directory:
        directory = Path(directory)
        adapter = directory / "adapter"
        model.save_pretrained(adapter, safe_serialization=True)
        manifest = {"completed": True, "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
                    "verification_fixture_only": True}
        path = directory / "adapter.pth"
        export_pth(model, adapter, path, manifest)
        loaded = load_pth_adapter(make_base(), path, MODEL_ID, MODEL_REVISION)
        expected, actual = get_peft_model_state_dict(model), get_peft_model_state_dict(loaded)
        assert set(expected) == set(actual)
        assert all(torch.equal(expected[k].cpu(), actual[k].cpu()) for k in expected)
        merged = model.merge_and_unload(safe_merge=True)
        merged.generation_config.eos_token_id = [1, 9]
        merged.generation_config.pad_token_id = 0
        full_path = directory / "full.pth"
        payload = {"format": "gemma-full-state-dict-v1", "base_model_id": MODEL_ID,
                   "base_model_revision": MODEL_REVISION, "training_manifest": manifest,
                   "config": merged.config.to_dict(), "generation_config": merged.generation_config.to_dict(),
                   "state_dict": merged.state_dict()}
        torch.save(payload, full_path)
        full_path.with_suffix(".json").write_text(json.dumps({k: v for k, v in payload.items() if k not in ("config", "state_dict", "generation_config")}
                                                            | {"sha256": sha256(full_path)}))
        restored = load_full_pth_model(full_path, MODEL_ID, MODEL_REVISION, device="cpu")
        assert restored.generation_config.to_dict() == merged.generation_config.to_dict()
        assert restored.generation_config.eos_token_id == [1, 9]
        assert restored.generation_config.pad_token_id == 0
        assert all(torch.equal(v.cpu(), restored.state_dict()[k].cpu()) for k, v in merged.state_dict().items())
        merged.eval()
        with torch.no_grad():
            before = merged(input_ids=tokens, attention_mask=torch.ones_like(tokens)).logits
            after = restored(input_ids=tokens, attention_mask=torch.ones_like(tokens)).logits
        assert torch.equal(before, after)
    receipt = {"model": "tiny random Gemma4Unified fixture, not real 12B weights", "device": "cpu",
               "native_forward_backward": "passed", "lora_pth_tensor_roundtrip": "passed",
               "full_pth_tensor_and_logits_roundtrip": "passed", "task_accuracy_evaluated": False}
    receipt["generation_config_eos_pad_roundtrip"] = "passed"
    args.receipt.write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
