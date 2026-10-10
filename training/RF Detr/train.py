"""Fine-tune official RF-DETR Medium; no GPU work occurs for --help/--dry-run.

API pinned to roboflow/rf-detr tag 1.11.2 (Lightning training stack).
Use a persistent cloud volume for --output. Test data is never used in fit().
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
import time

HERE = Path(__file__).resolve().parent
RFDETR_VERSION = "1.11.2"


def load_json(path: Path):
    with path.open(encoding="utf-8-sig") as handle:
        return json.load(handle)


def save_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def check_split(dataset: Path, split: str) -> dict:
    """Cheap preflight; prepare_data.py performs the full image/box audit."""
    annotation = dataset / split / "_annotations.coco.json"
    if not annotation.is_file():
        raise ValueError(f"Missing {annotation}; run prepare_data.py first.")
    coco = load_json(annotation)
    images = coco.get("images", [])
    categories = sorted(coco.get("categories", []), key=lambda item: item["id"])
    if not images or not categories:
        raise ValueError(f"{annotation} has no images or categories.")
    category_ids = {item["id"] for item in categories}
    image_ids = {item["id"] for item in images}
    if len(image_ids) != len(images) or len(category_ids) != len(categories):
        raise ValueError(f"Duplicate IDs in {annotation}.")
    for item in images:
        image_path = (dataset / split / item["file_name"]).resolve()
        if not image_path.is_relative_to((dataset / split).resolve()):
            raise ValueError(f"Image path escapes split directory: {item['file_name']}")
        if not image_path.is_file():
            raise ValueError(f"Image missing: {image_path}")
    for box in coco.get("annotations", []):
        if box["category_id"] not in category_ids or box["image_id"] not in image_ids:
            raise ValueError(f"Unresolved annotation IDs in {annotation}.")
    return {
        "images": len(images),
        "annotations": len(coco.get("annotations", [])),
        "categories": categories,
        "annotation_sha256": hashlib.sha256(annotation.read_bytes()).hexdigest(),
    }


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data", type=Path, default=HERE / "data" / "ood")
    cli.add_argument("--output", type=Path, default=HERE / "runs" / "medium-ood")
    cli.add_argument("--config", type=Path, default=HERE / "configs" / "h100_medium.json")
    cli.add_argument("--epochs", type=int, help="Override 30-epoch budget; validate every epoch.")
    cli.add_argument("--batch-size", type=int)
    cli.add_argument("--grad-accum-steps", type=int)
    cli.add_argument("--num-workers", type=int)
    cli.add_argument("--resume", type=Path, help="Own trusted Lightning last.ckpt; use the original output directory.")
    cli.add_argument("--dry-run", action="store_true", help="Show config without importing ML packages or reading data.")
    cli.add_argument("--benchmark", action="store_true", help="Train one real full epoch in a sibling output-benchmark directory, then estimate the full budget.")
    return cli


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    config = load_json(args.config)
    if not isinstance(config, dict) or set(config) != {"model", "train"}:
        raise ValueError("Config must have exactly model and train objects.")
    if not isinstance(config["model"], dict) or not isinstance(config["train"], dict):
        raise ValueError("Config model and train must be objects.")
    model_args = dict(config["model"])
    train_args = dict(config["train"])
    for name in ("epochs", "batch_size", "grad_accum_steps", "num_workers"):
        value = getattr(args, name)
        if value is not None:
            train_args[name] = value
    for name in ("epochs", "batch_size", "grad_accum_steps"):
        if type(train_args.get(name)) is not int or train_args[name] <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if type(train_args.get("num_workers")) is not int or train_args["num_workers"] < 0:
        raise ValueError("num_workers must be a nonnegative integer.")
    for name in ("lr", "lr_encoder"):
        value = train_args.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a finite positive number.")
    if train_args.get("amp_dtype") != "bf16":
        raise ValueError("This H100 recipe requires amp_dtype=bf16.")
    if train_args.get("devices") != 1 or train_args.get("num_nodes") != 1:
        raise ValueError("This recipe is for one H100. Multi-GPU requires retuning global batch size.")
    if train_args.get("run_test"):
        raise ValueError("Do not use the test split during training; use evaluate.py after selection.")
    if model_args.get("resolution") != 576:
        raise ValueError("This verified Medium recipe uses the native 576-pixel resolution.")
    if args.benchmark and args.resume:
        raise ValueError("Benchmark and resume cannot be combined.")
    requested_epochs = train_args["epochs"]
    output = args.output.resolve()
    if args.benchmark:
        output = output.with_name(output.name + "-benchmark")
    train_args.update(dataset_dir=str(args.data.resolve()), output_dir=str(output), device="cuda")
    if args.resume:
        if args.resume.suffix != ".ckpt":
            raise ValueError("Resume needs last.ckpt (optimizer state), not an inference .pth.")
        train_args["resume"] = str(args.resume.resolve())
    if args.benchmark:
        train_args.update(epochs=1, early_stopping=False, warmup_epochs=0.0)
    plan = {
        "package": f"rfdetr=={RFDETR_VERSION}",
        "model_class": "RFDETRMedium",
        "model": model_args,
        "train": train_args,
        "effective_batch_size": train_args["batch_size"] * train_args["grad_accum_steps"],
        "benchmark": args.benchmark,
        "planned_full_epochs": requested_epochs,
        "test_used_in_training": False,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0

    dataset = args.data.resolve()
    splits = {name: check_split(dataset, name) for name in ("train", "valid")}
    if splits["train"]["categories"] != splits["valid"]["categories"]:
        raise ValueError("Train/valid category schemas differ. Run prepare_data.py.")
    class_names = [item["name"] for item in splits["train"]["categories"]]
    if len(set(class_names)) != len(class_names):
        raise ValueError("Category names must be unique.")
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError(f"Output directory is nonempty: {output}. Choose a new output or --resume.")
    if args.resume and not args.resume.is_file():
        raise ValueError(f"Missing resume checkpoint: {args.resume}")

    # Lazy imports keep --help, config review and lightweight checks GPU-independent.
    if importlib.metadata.version("rfdetr") != RFDETR_VERSION:
        raise RuntimeError(f"Install requirements.txt: this script requires rfdetr=={RFDETR_VERSION}.")
    import torch
    from rfdetr import RFDETRMedium
    from rfdetr.config import RFDETRMediumConfig, TrainConfig

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required; this script will not silently train on CPU.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This H100 recipe requires native BF16 support.")
    train_args["class_names"] = class_names
    # Validate the exact pinned API before allocating or downloading model weights.
    RFDETRMediumConfig(**model_args, device="cuda")
    TrainConfig(**{key: value for key, value in train_args.items() if key != "device"})
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "run_plan.json", plan)
    save_json(output / "data_fingerprint.json", splits)
    save_json(output / "class_names.json", class_names)
    metadata = {"python": platform.python_version(), "gpu": torch.cuda.get_device_name(0)}
    for name in ("rfdetr", "torch", "torchvision", "pytorch-lightning", "transformers"):
        metadata[name] = importlib.metadata.version(name)
    save_json(output / "environment.json", metadata)
    print(json.dumps({"hardware": metadata, "plan": plan}, indent=2), flush=True)
    setup_start = time.perf_counter()
    model = RFDETRMedium(**model_args, device="cuda")
    model_setup_seconds = time.perf_counter() - setup_start
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model.train(**train_args)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    report = {
        "model_setup_seconds": model_setup_seconds,
        "fit_wall_seconds": elapsed,
        "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "training_images": splits["train"]["images"],
        "validation_images": splits["valid"]["images"],
        "requested_epochs": train_args["epochs"],
        "inference_checkpoint": str(output / "checkpoint_best_total.pth"),
        "note": "Fit time includes validation and saving. Early stopping may finish before the epoch budget.",
    }
    if args.benchmark:
        report.update({
            "projected_full_fit_seconds": elapsed * requested_epochs,
            "projected_total_seconds": model_setup_seconds + elapsed * requested_epochs,
            "projection_note": "One-epoch extrapolation, not a measured full-run time or guarantee. Repeats first-epoch overhead; storage, warmup and early stopping change runtime.",
            "nominal_optimizer_steps_per_epoch": math.ceil(splits["train"]["images"] / plan["effective_batch_size"]),
        })
    save_json(output / "timing.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileNotFoundError, importlib.metadata.PackageNotFoundError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2)
