"""Attach actual frozen RF detections to PAVE teacher-source examples.

This does not create hazard labels. Native source descriptions remain privileged
teacher context; the teacher must produce the separate task-contract targets.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def object_rows(result, width: int, height: int, max_objects: int = 100) -> list[dict]:
    objects = []
    for box, confidence, name in zip(result.xyxy, result.confidence, result.data["class_name"], strict=True):
        if name == "__background__":
            continue
        values = [float(v) for v in box] + [float(confidence)]
        if not all(math.isfinite(value) for value in values) or not 0 <= values[4] <= 1:
            raise ValueError("RF returned an invalid box/confidence")
        x1, y1, x2, y2 = values[:4]
        normalized = [max(0., min(1., x1/width)), max(0., min(1., y1/height)),
                      max(0., min(1., x2/width)), max(0., min(1., y2/height))]
        if normalized[0] >= normalized[2] or normalized[1] >= normalized[3]:
            continue
        objects.append({"track_id": len(objects)+1, "bbox": normalized,
                        "label": str(name), "confidence": values[4]})
        if len(objects) >= max_objects:
            break
    return objects


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Prepared PAVE source-QA directory")
    parser.add_argument("--output", type=Path, required=True, help="New teacher-source manifest directory")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1 or args.batch_size < 1:
        parser.error("Require threshold 0..1 and positive batch size")
    source_report = json.loads((args.source/"data_report.json").read_text(encoding="utf-8"))
    if source_report.get("status") != "prepared":
        raise ValueError("Source data preparation is not complete")
    import torch
    from PIL import Image
    from rfdetr import RFDETRMedium

    if not torch.cuda.is_available():
        raise RuntimeError("This RF metadata recipe requires a CUDA GPU")
    checkpoint_sha = digest(args.checkpoint)
    model = RFDETRMedium.from_checkpoint(str(args.checkpoint.resolve()), device="cuda")
    model.inference(compile=False, dtype=torch.bfloat16)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "status": "preparing",
              "task": "teacher_source", "source_data_report": source_report,
              "rf_checkpoint_sha256": checkpoint_sha, "rf_threshold": args.threshold,
              "rf_max_objects": 100, "splits": {},
              "note": "Actual detections; source QA is privileged teacher context, not a final hazard target. IDs are frame-local."}
    try:
        for split in ["train", "validation"]:
            rows = [json.loads(line) for line in (args.source/f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            total_boxes = 0
            no_boxes = 0
            with (args.output/f"{split}.jsonl").open("w", encoding="utf-8") as stream:
                for start in range(0, len(rows), args.batch_size):
                    batch = rows[start:start+args.batch_size]
                    paths = [row["frames"][-1]["path"] for row in batch]
                    results = model.predict(paths, threshold=args.threshold, include_source_image=False)
                    for row, path, result in zip(batch, paths, results, strict=True):
                        with Image.open(path) as image:
                            objects = object_rows(result, image.width, image.height)
                        row["objects"] = objects
                        row["provenance"]["rf_context"] = {"checkpoint_sha256": checkpoint_sha,
                            "threshold": args.threshold, "object_ids": "Frame-local IDs assigned to actual frozen RF detections"}
                        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
                        total_boxes += len(objects)
                        no_boxes += not bool(objects)
                    print(f"RF context {split}: {min(start+args.batch_size,len(rows))}/{len(rows)}", flush=True)
            report["splits"][split] = {"examples": len(rows), "boxes": total_boxes,
                                       "images_without_boxes": no_boxes,
                                       "manifest_sha256": digest(args.output/f"{split}.jsonl")}
        report["status"] = "prepared"
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        (args.output/"data_report.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
