"""Evaluate a fine-tuned Medium checkpoint with COCO AP/AR and per-class metrics.

Select the checkpoint on valid first. Run --split test once after selection.
Uses all predictions (no deployment confidence cutoff) and COCO maxDets=100.
"""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent


def summarize_per_class(evaluator, ground_truth) -> list[dict]:
    """Convert COCO's NumPy IDs and scalar metrics to JSON-native values."""
    def mean_valid(values):
        values = values[values >= 0]
        return float(values.mean()) if values.size else None

    # COCO precision [IoU,recall,class,area,maxDets], recall [IoU,class,area,maxDets].
    results = []
    for index, raw_category_id in enumerate(evaluator.params.catIds):
        category_id = int(raw_category_id)
        precision = evaluator.eval["precision"][:, :, index, 0, -1]
        recall = evaluator.eval["recall"][:, index, 0, -1]
        category = ground_truth.cats[category_id]
        results.append({
            "category_id": category_id, "name": category["name"],
            "annotations": len(ground_truth.getAnnIds(catIds=[category_id])),
            "AP_50_95": mean_valid(precision), "AP_50": mean_valid(precision[0]),
            "AR_100": mean_valid(recall),
        })
    return results


def main(argv=None) -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data", type=Path, default=HERE / "data" / "ood")
    cli.add_argument("--checkpoint", type=Path, required=True)
    cli.add_argument("--split", choices=("valid", "test"), default="valid")
    cli.add_argument("--batch-size", type=int, default=8)
    cli.add_argument("--output", type=Path, help="New output directory; defaults beside checkpoint.")
    args = cli.parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive.")
    annotation_path = args.data.resolve() / args.split / "_annotations.coco.json"
    if not annotation_path.is_file() or not args.checkpoint.is_file():
        raise ValueError("Dataset annotations and checkpoint must already exist.")
    with annotation_path.open(encoding="utf-8-sig") as handle:
        annotation = json.load(handle)
    names_to_ids = {item["name"]: item["id"] for item in annotation["categories"]}
    if len(names_to_ids) != len(annotation["categories"]):
        raise ValueError("Duplicate class names prevent an unambiguous mapping.")
    output = args.output or args.checkpoint.parent / f"evaluation-{args.split}"
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Evaluation output is nonempty: {output}; use a new --output.")
    if importlib.metadata.version("rfdetr") != "1.11.2":
        raise RuntimeError("Install requirements.txt; expected rfdetr==1.11.2.")

    import numpy as np
    import torch
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from rfdetr import RFDETRMedium

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for this evaluation recipe.")
    model = RFDETRMedium.from_checkpoint(str(args.checkpoint.resolve()), device="cuda")
    if not isinstance(model, RFDETRMedium):
        raise ValueError("Checkpoint is not RF-DETR Medium.")
    if set(model.class_names) != set(names_to_ids):
        raise ValueError("Checkpoint class names do not match the dataset. Do not compare mismatched label spaces.")
    # Compile is disabled to avoid a long first-run compilation. Explicit BF16 inference.
    model.inference(compile=False, dtype=torch.bfloat16)
    output.mkdir(parents=True, exist_ok=True)
    ground_truth = COCO(str(annotation_path))
    ground_truth.dataset.setdefault("info", {})
    images = sorted(annotation["images"], key=lambda item: item["id"])
    if not images:
        raise ValueError("The chosen split is empty.")
    predictions = []
    started = time.perf_counter()
    for start in range(0, len(images), args.batch_size):
        batch = images[start:start + args.batch_size]
        paths = [str(annotation_path.parent / item["file_name"]) for item in batch]
        detections = model.predict(paths, threshold=0.0, include_source_image=False)
        for record, result in zip(batch, detections, strict=True):
            for box, score, name in zip(result.xyxy, result.confidence, result.data["class_name"], strict=True):
                if name == "__background__":
                    continue
                if name not in names_to_ids:
                    raise ValueError(f"Prediction has unknown class {name!r}.")
                x1, y1, x2, y2 = map(float, box)
                values = (x1, y1, x2, y2, float(score))
                if not all(np.isfinite(value) for value in values):
                    raise ValueError("Non-finite prediction detected.")
                if x2 <= x1 or y2 <= y1:
                    continue
                predictions.append({
                    "image_id": record["id"], "category_id": names_to_ids[name],
                    "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(score),
                })
        print(f"Evaluated {min(start + args.batch_size, len(images))}/{len(images)} images", flush=True)
    torch.cuda.synchronize()
    prediction_seconds = time.perf_counter() - started
    (output / "predictions.coco.json").write_text(json.dumps(predictions), encoding="utf-8")
    if predictions:
        detections_coco = ground_truth.loadRes(predictions)
    else:
        # COCO.loadRes([]) indexes element zero; an empty result still needs a valid report.
        detections_coco = COCO()
        detections_coco.dataset = copy.deepcopy(ground_truth.dataset)
        detections_coco.dataset["annotations"] = []
        detections_coco.createIndex()
    evaluator = COCOeval(ground_truth, detections_coco, "bbox")
    evaluator.params.imgIds = [item["id"] for item in images]
    evaluator.params.maxDets = [1, 10, 100]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    per_class = summarize_per_class(evaluator, ground_truth)
    names = ("AP_50_95", "AP_50", "AP_75", "AP_small", "AP_medium", "AP_large", "AR_1", "AR_10", "AR_100", "AR_small", "AR_medium", "AR_large")
    report = {
        "checkpoint": str(args.checkpoint.resolve()), "split": args.split,
        "images": len(images), "precision": "bf16", "max_detections": 100,
        "prediction_wall_seconds": prediction_seconds,
        "metrics": {name: float(value) if value >= 0 else None for name, value in zip(names, evaluator.stats, strict=True)},
        "per_class": per_class,
        "limitations": "Object localization evaluation only; does not measure hazard severity, blind-user usefulness or live-video alert latency.",
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileNotFoundError, importlib.metadata.PackageNotFoundError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2)
