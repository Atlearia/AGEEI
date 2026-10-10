#!/usr/bin/env python3
"""Validate a local Roboflow COCO export and prepare RF-DETR train/valid/test.

Preserves published split membership, except exact decoded-pixel duplicates are
removed with test > valid > train precedence. Never overwrites source or output.
No model, API key, network connection, or GPU is needed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import struct
import tempfile

from PIL import Image

SPLITS = ("train", "valid", "test")
SOURCE_URL = "https://universe.roboflow.com/fpn/ood-pbnro/dataset/1"


class DataError(ValueError):
    """The input is unsafe or unsuitable for training."""


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _inside(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root.resolve()) or candidate == root.resolve():
        raise DataError(f"Image path escapes split directory: {relative!r}")
    return candidate


def _signature(annotations):
    # Compare semantic labels rather than split-local category or annotation IDs.
    return sorted((a["class_name"], tuple(a["bbox"]), a["iscrowd"]) for a in annotations)


def _validate_annotation(annotation, categories, width, height):
    category_id = annotation.get("category_id")
    if category_id not in categories:
        raise DataError(f"Unknown annotation category_id: {category_id!r}")
    box = annotation.get("bbox")
    if not isinstance(box, list) or len(box) != 4:
        raise DataError("Every bbox must be COCO [x, y, width, height]")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box):
        raise DataError("Bounding box contains non-finite or non-numeric values")
    x, y, w, h = map(float, box)
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width + 1e-6 or y + h > height + 1e-6:
        raise DataError(f"Out-of-bounds/empty bbox {box} in {width}x{height} image")
    crowd = annotation.get("iscrowd", 0)
    if crowd not in (0, 1):
        raise DataError("iscrowd must be 0 or 1")
    return {"class_name": categories[category_id], "bbox": [x, y, w, h], "iscrowd": int(crowd)}


def _load_split(source: Path, split: str, invalid_images: str, report):
    directory = source / split
    if split == "valid" and not directory.exists():
        directory = source / "val"
    annotations_path = directory / "_annotations.coco.json"
    if not annotations_path.is_file():
        raise DataError(f"Missing annotations: {annotations_path}")
    raw = _read(annotations_path)
    report["source_annotations"][split] = {
        "path": str(annotations_path),
        "sha256": hashlib.sha256(annotations_path.read_bytes()).hexdigest(),
        "images": len(raw.get("images", [])),
        "annotations": len(raw.get("annotations", [])),
    }
    categories = {}
    for category in raw.get("categories", []):
        category_id, name = category.get("id"), category.get("name")
        if not isinstance(category_id, int) or isinstance(category_id, bool) or not isinstance(name, str) or not name.strip():
            raise DataError("Categories must have integer IDs and nonempty names")
        if category_id in categories or name in categories.values():
            raise DataError(f"Duplicate category ID/name in {annotations_path}")
        categories[category_id] = name
    report["declared_categories"][split] = categories
    by_image = defaultdict(list)
    seen_annotation_ids = set()
    for annotation in raw.get("annotations", []):
        annotation_id = annotation.get("id")
        if not isinstance(annotation_id, int) or isinstance(annotation_id, bool) or annotation_id in seen_annotation_ids:
            raise DataError(f"Missing/duplicate annotation ID in {annotations_path}")
        seen_annotation_ids.add(annotation_id)
        by_image[annotation.get("image_id")].append(annotation)
    seen_ids = set()
    records = []
    for image in raw.get("images", []):
        image_id = image.get("id")
        if not isinstance(image_id, int) or isinstance(image_id, bool) or image_id in seen_ids:
            raise DataError(f"Missing/duplicate image ID in {annotations_path}")
        seen_ids.add(image_id)
        try:
            name = image.get("file_name")
            if not isinstance(name, str) or not name:
                raise DataError("Missing image filename")
            path = _inside(directory, name)
            if not path.is_file():
                raise DataError(f"Missing image: {name}")
            width, height = image.get("width"), image.get("height")
            if any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in (width, height)):
                raise DataError("Image dimensions must be positive integers")
            normalized = [_validate_annotation(a, categories, width, height) for a in by_image[image_id]]
            with Image.open(path) as pixels:
                pixels.load()
                if pixels.size != (width, height):
                    raise DataError(f"Recorded image dimensions differ from decoded size: {name}")
                if pixels.getexif().get(274, 1) not in (None, 1):
                    raise DataError(f"EXIF-rotated image requires explicit box-aware normalization: {name}")
                pixel_hash = hashlib.sha256(struct.pack("!II", width, height) + pixels.convert("RGB").tobytes()).hexdigest()
            records.append({"source_id": image_id, "source_name": name, "path": path,
                            "width": width, "height": height, "pixel_sha256": pixel_hash,
                            "annotations": normalized})
        except (DataError, OSError, ValueError, Image.DecompressionBombError) as exc:
            if invalid_images == "error":
                raise DataError(f"{split} image {image_id}: {exc}") from exc
            # Dropping only a bad box would falsely label that object as background.
            report["dropped_invalid_images"].append({"split": split, "image_id": image_id, "reason": str(exc)})
    orphans = set(by_image) - seen_ids
    if orphans:
        raise DataError(f"Annotations reference missing image IDs in {split}: {sorted(map(str, orphans))[:10]}")
    return records


def _subset(records, limit, seed):
    """Reproducible class-coverage-first subset; fill remaining quota at random."""
    if limit is None or limit >= len(records):
        return records
    if limit < 1:
        raise DataError("--max-train must be positive")
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    class_sets = [{a["class_name"] for a in record["annotations"] if not a["iscrowd"]} for record in shuffled]
    uncovered = set().union(*class_sets)
    selected = set()
    while uncovered:
        best = max((i for i in range(len(shuffled)) if i not in selected),
                   key=lambda i: len(class_sets[i] & uncovered))
        selected.add(best)
        uncovered -= class_sets[best]
        if len(selected) > limit:
            raise DataError(f"--max-train {limit} cannot cover every observed class; increase it")
    for i in range(len(shuffled)):
        if len(selected) >= limit:
            break
        selected.add(i)
    return [record for i, record in enumerate(shuffled) if i in selected]


def prepare(source, output, *, max_train=None, seed=42, invalid_images="error",
            source_name="local COCO export", source_url=None, license_name="unverified"):
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_dir():
        raise DataError(f"Source directory does not exist: {source}")
    if output.exists():
        raise DataError(f"Refusing to overwrite existing output: {output}")
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise DataError("Source and output must be separate, non-nested directories")
    if invalid_images not in ("error", "drop"):
        raise DataError("invalid_images must be error or drop")
    report = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "source": {"name": source_name, "url": source_url, "license": license_name, "path": str(source)},
              "source_annotations": {}, "declared_categories": {}, "dropped_invalid_images": [],
              "duplicates_removed": [], "splits": {}, "seed": seed, "max_train": max_train,
              "warnings": ["Boxes identify objects, not hazards or severity.",
                           "Exact decoded-pixel duplicate checking does not establish route/session independence or detect near duplicates."]}
    records = {split: _load_split(source, split, invalid_images, report) for split in SPLITS}
    all_classes = sorted({a["class_name"] for group in records.values() for record in group for a in record["annotations"]})
    if not all_classes:
        raise DataError("No annotated classes found")
    seen = {}
    for split in ("test", "valid", "train"):
        kept = []
        for record in records[split]:
            digest = record["pixel_sha256"]
            if digest in seen:
                previous_split, previous = seen[digest]
                if _signature(previous["annotations"]) != _signature(record["annotations"]):
                    raise DataError(f"Conflicting labels for exact duplicate {split}/{record['source_name']} and {previous_split}/{previous['source_name']}")
                report["duplicates_removed"].append({"removed_split": split, "removed_file": record["source_name"],
                                                     "kept_split": previous_split, "kept_file": previous["source_name"], "pixel_sha256": digest})
            else:
                kept.append(record)
                seen[digest] = (split, record)
        records[split] = kept
    records["train"] = _subset(records["train"], max_train, seed)
    train_classes = {a["class_name"] for record in records["train"] for a in record["annotations"] if not a["iscrowd"]}
    if set(all_classes) - train_classes:
        raise DataError(f"No usable training labels for classes: {sorted(set(all_classes) - train_classes)}")
    for split in SPLITS:
        if not records[split]:
            raise DataError(f"No images remain in {split}")
    categories = [{"id": i + 1, "name": name, "supercategory": "object"} for i, name in enumerate(all_classes)]
    class_to_id = {c["name"]: c["id"] for c in categories}
    report["categories"] = categories
    report["category_id_policy"] = "COCO IDs 1..N; RF-DETR 1.11.2 remaps custom labels to prediction IDs 0..N-1."
    report["removed_unannotated_category_names"] = sorted(set().union(*(set(d.values()) for d in report["declared_categories"].values())) - set(all_classes))
    if max_train is not None:
        report["warnings"].append("Pilot subset: class coverage is enforced; remaining examples are seeded random, not a guaranteed balanced sample.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".rfdetr-prepare-", dir=output.parent) as temporary:
        staging = Path(temporary)
        for split in SPLITS:
            folder = staging / split
            folder.mkdir()
            images, annotations, manifest = [], [], []
            counts = Counter()
            for image_id, record in enumerate(records[split], 1):
                # Flatten nested source names and avoid collisions.
                filename = f"{image_id:07d}{record['path'].suffix.lower()}"
                shutil.copyfile(record["path"], folder / filename)
                images.append({"id": image_id, "file_name": filename, "width": record["width"], "height": record["height"]})
                manifest.append({"image_id": image_id, "file_name": filename, "source_id": record["source_id"],
                                 "source_name": record["source_name"], "pixel_sha256": record["pixel_sha256"]})
                for item in record["annotations"]:
                    box = item["bbox"]
                    annotations.append({"id": len(annotations) + 1, "image_id": image_id, "category_id": class_to_id[item["class_name"]],
                                        "bbox": box, "area": box[2] * box[3], "iscrowd": item["iscrowd"]})
                    counts[item["class_name"]] += 1
            _write(folder / "_annotations.coco.json", {"info": {"description": source_name}, "images": images,
                                                        "annotations": annotations, "categories": categories})
            _write(staging / f"{split}_manifest.json", manifest)
            report["splits"][split] = {"images": len(images), "annotations": len(annotations),
                                      "class_instance_counts": {name: counts[name] for name in all_classes},
                                      "negative_images": sum(not r["annotations"] for r in records[split]),
                                      "crowd_annotations": sum(a["iscrowd"] for a in annotations)}
        if any(s["crowd_annotations"] for s in report["splits"].values()):
            report["warnings"].append("COCO crowd labels are preserved; detector training/evaluation may ignore them. Check crowd counts.")
        _write(staging / "class_names.json", all_classes)
        _write(staging / "data_report.json", report)
        # Reserve a fresh destination. Never merge with an existing folder.
        output.mkdir(exist_ok=False)
        for item in staging.iterdir():
            shutil.move(str(item), str(output / item.name))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Local COCO root containing train/valid/test (val alias supported)")
    parser.add_argument("--output", required=True, type=Path, help="New output directory; never overwritten")
    parser.add_argument("--max-train", type=int, help="Optional pilot limit; default uses every valid, nonduplicate training image")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--invalid-images", choices=("error", "drop"), default="error", help="Default fail; drop removes entire invalid image and all its boxes")
    parser.add_argument("--source-name", default="local COCO export")
    parser.add_argument("--source-url")
    parser.add_argument("--license", default="unverified", dest="license_name")
    parser.add_argument("--ood-v1", action="store_true", help="Record OOD v1 source attribution (use only for that export)")
    args = parser.parse_args()
    if args.ood_v1:
        args.source_name, args.source_url, args.license_name = "OOD v1 by FPN", SOURCE_URL, "CC BY 4.0"
    try:
        report = prepare(args.source, args.output, max_train=args.max_train, seed=args.seed, invalid_images=args.invalid_images,
                         source_name=args.source_name, source_url=args.source_url, license_name=args.license_name)
    except (DataError, OSError, json.JSONDecodeError) as exc:
        parser.exit(2, f"Dataset preparation failed: {exc}\n")
    print(json.dumps({"output": str(args.output.resolve()), "splits": report["splits"],
                      "removed_duplicates": len(report["duplicates_removed"])}, indent=2))


if __name__ == "__main__":
    main()
