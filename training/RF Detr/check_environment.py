"""Check the RF-DETR environment and CUDA without downloading data or weights."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-h100", action="store_true", help="Fail if the selected GPU is not an H100")
    args = parser.parse_args()
    versions = {name: importlib.metadata.version(name) for name in
                ("rfdetr", "torch", "torchvision", "transformers", "pytorch-lightning", "pycocotools")}
    if versions["rfdetr"] != "1.11.2":
        raise RuntimeError("Install requirements.txt; expected rfdetr==1.11.2")
    import torch
    import torchvision
    from rfdetr.config import RFDETRMediumConfig, TrainConfig

    config_path = Path(__file__).resolve().parent / "configs" / "h100_medium.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for key, schema in (("model", RFDETRMediumConfig), ("train", TrainConfig)):
        unknown = set(config[key]) - set(schema.model_fields)
        if unknown:
            raise RuntimeError(f"Unsupported {key} options in pinned API: {sorted(unknown)}")
    RFDETRMediumConfig(**config["model"])
    TrainConfig(**config["train"], dataset_dir="environment-check-only", output_dir="environment-check-only")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A CUDA GPU with BF16 support is required")
    gpu = torch.cuda.get_device_name(0)
    if args.require_h100 and "H100" not in gpu:
        raise RuntimeError(f"Expected an H100, found {gpu}")
    # Catch driver/wheel mismatch before downloading the dataset or model weights.
    x = torch.ones((64, 64), dtype=torch.bfloat16, device="cuda")
    if not torch.isfinite(x @ x).all().item():
        raise RuntimeError("BF16 CUDA matrix multiplication returned non-finite values")
    boxes = torch.tensor([[0., 0., 1., 1.]], device="cuda")
    torchvision.ops.nms(boxes, torch.ones(1, device="cuda"), 0.5)
    torch.cuda.synchronize()
    print(json.dumps({"python": platform.python_version(), "packages": versions,
                      "gpu": gpu, "cuda": torch.version.cuda,
                      "vram_gib": round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1),
                      "bf16_cuda_check": "passed", "config_schema_check": "passed"}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ImportError, RuntimeError, ValueError, importlib.metadata.PackageNotFoundError) as error:
        print(f"Environment check failed: {error}", file=sys.stderr)
        raise SystemExit(2)
