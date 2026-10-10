"""Prepare a bounded PAVE/SANPO accessibility-QA pilot, not invented hazard labels.

Only source-supported assessment prose is used as the assistant target. Official
PAVE evaluation and SANPO test sessions never become training examples.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import threading
import urllib.parse
import urllib.request

PAVE_REVISION = "945dae763031eadcd77917890d2ffaf68624131f"
PAVE_BASE = f"https://huggingface.co/datasets/rafiibnsultan/PAVE/resolve/{PAVE_REVISION}/"
SANPO_BASE = "https://storage.googleapis.com/gresearch/sanpo_dataset/v0/sanpo-real/"
METADATA_HASHES = {
    "PAVE_train85.jsonl": "974d317fdefda9acbb58dda714ae1f0271ff0529cafb3c0c743b7ce1a21c4ac9",
    "PAVE_val85.jsonl": "1a548c66fbd841e9402a9733d5940614c6f977ac7be7b2e7f411507d2d88b78d",
    "sanpo_test_session_ids.txt": "af2c6a5aa06cd1e8ec2cff76672966dac7288b3b5267a14f835d8595601a13d0",
    "sanpo_train_session_ids.txt": "1bad2efa02690f455d5f27b59319eb9ef31c0efb6e1a314b056b63d836fb3cec",
}
SOURCES = {
    "source_url": "https://huggingface.co/datasets/rafiibnsultan/PAVE",
    "annotations": "https://huggingface.co/datasets/rafiibnsultan/PAVE",
    "images": "https://github.com/google-research-datasets/sanpo_dataset",
    "annotation_license": "https://huggingface.co/datasets/rafiibnsultan/PAVE/blob/" + PAVE_REVISION + "/README.md",
    "image_license": "https://github.com/google-research-datasets/sanpo_dataset#license--contact",
    "license": "CC-BY-4.0",
    "license_verified": True,
    "annotation_generation": "Existing GPT-5-nano QA generated from SANPO scene attributes by PAVE authors",
    "annotation_method_url": "https://arxiv.org/html/2603.10703v1#S3.SS2",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ScopedRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(request, fp, code, msg, headers, newurl)
        if redirected and urllib.parse.urlparse(request.full_url).hostname != urllib.parse.urlparse(newurl).hostname:
            redirected.remove_header("Authorization")
        return redirected


class DownloadBudget:
    def __init__(self, limit: int):
        self.limit, self.used = limit, 0
        self.lock = threading.Lock()

    def claim(self, count: int):
        with self.lock:
            if self.used + count > self.limit:
                raise ValueError("Dataset download budget exhausted; reduce frame counts")
            self.used += count


def bounded_get(url: str, max_bytes: int, budget: DownloadBudget | None = None) -> bytes:
    """Read a public source with a hard per-request cap; never log credentials."""
    headers = {"User-Agent": "AGEII-dataset-audit/1.0"}
    if urllib.parse.urlparse(url).hostname == "huggingface.co" and os.getenv("HF_TOKEN"):
        headers["Authorization"] = "Bearer " + os.environ["HF_TOKEN"]
    opener = urllib.request.build_opener(ScopedRedirect())
    with opener.open(urllib.request.Request(url, headers=headers), timeout=60) as response:
        size = response.headers.get("Content-Length")
        if size and int(size) > max_bytes:
            raise ValueError(f"Source exceeds request cap of {max_bytes} bytes")
        chunks, total = [], 0
        while True:
            chunk = response.read(min(1024 * 1024, max_bytes + 1-total))
            if not chunk:
                break
            if budget is not None:
                budget.claim(len(chunk))
            chunks.append(chunk)
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(f"Source exceeds request cap of {max_bytes} bytes")
        data = b"".join(chunks)
    if len(data) > max_bytes:
        raise ValueError(f"Source exceeds request cap of {max_bytes} bytes")
    return data


def download_rows(rows: list[dict], workers: int, budget: DownloadBudget):
    """At most workers images are downloading/queued in RAM, in source order."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = deque()
        iterator = iter(rows)
        for _ in range(workers):
            row = next(iterator, None)
            if row is not None:
                pending.append((row, pool.submit(bounded_get, row["image_url"], 20_000_000, budget)))
        while pending:
            row, future = pending.popleft()
            raw = future.result()
            yield row, raw
            following = next(iterator, None)
            if following is not None:
                pending.append((following, pool.submit(bounded_get, following["image_url"], 20_000_000, budget)))


def cached_get(cache: Path, filename: str, url: str, limit: int) -> tuple[bytes, dict]:
    path = cache / filename
    if path.exists():
        if path.stat().st_size > limit:
            raise ValueError("Cached metadata exceeds size cap")
        data = path.read_bytes()
    else:
        data = bounded_get(url, limit)
        cache.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    if filename in METADATA_HASHES and sha256(data) != METADATA_HASHES[filename]:
        raise ValueError(f"Pinned metadata checksum mismatch: {filename}; inspect source changes before updating hashes")
    return data, {"url": url, "sha256": sha256(data), "bytes": len(data)}


def parse_row(row: dict) -> dict:
    session = row.get("session", "")
    match = re.fullmatch(r"PAVE/([A-Za-z0-9_-]+)/camera_(head|chest)/(left|right)/?", session)
    index = str(row.get("index", ""))
    if not match or not re.fullmatch(r"[0-9]{6}", index):
        raise ValueError("Unsupported or unsafe PAVE session/index path")
    question = row.get("question")
    assessment = row.get("assessment")
    if not isinstance(question, str) or not question.strip() or not isinstance(assessment, str):
        raise ValueError("PAVE row lacks source question/assessment")
    paragraphs = re.findall(r"\[assessment\]\s*(.*?)\s*\[/assessment\]", assessment, re.DOTALL)
    if len(paragraphs) != 1 or not paragraphs[0].strip():
        raise ValueError("PAVE row must have exactly one nonempty assessment paragraph")
    # Distances and segmentation tokens depend on modalities absent from this
    # pilot. Retain the source paragraph verbatim; do not manufacture substitutes.
    recording, camera, side = match.groups()
    relative = f"{recording}/camera_{camera}/{side}/video_frames/{index}.png"
    return {
        "source_id": str(row["id"]), "session_id": recording,
        "frame_key": relative, "image_url": SANPO_BASE + relative,
        "question": question.strip(), "target": paragraphs[0].strip(),
        "source_row_sha256": sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()),
    }


def parse_jsonl(data: bytes) -> list[dict]:
    return [parse_row(json.loads(line)) for line in data.decode("utf-8-sig").splitlines() if line.strip()]


def partition_rows(train: list[dict], official_eval: list[dict], sanpo_test: set[str], seed: int = 42):
    """Group-held validation from train sessions; official eval stays test-only."""
    eval_sessions = {row["session_id"] for row in official_eval}
    original_train_sessions = {row["session_id"] for row in train}
    if original_train_sessions & eval_sessions:
        raise ValueError("PAVE train/evaluation recording overlap: refusing leakage")
    excluded = [row for row in train if row["session_id"] in sanpo_test]
    eligible = [row for row in train if row["session_id"] not in sanpo_test]
    groups = sorted({row["session_id"] for row in eligible},
                    key=lambda group: sha256(f"{seed}:{group}".encode()))
    if len(groups) < 2:
        raise ValueError("Need at least two eligible recording sessions for grouped validation")
    heldout_count = max(1, round(len(groups) * 0.1))
    validation_groups = set(groups[:heldout_count])
    return {
        "train": [row for row in eligible if row["session_id"] not in validation_groups],
        "validation": [row for row in eligible if row["session_id"] in validation_groups],
        "test": official_eval,
    }, excluded


def select_frames(rows: list[dict], max_frames: int, seed: int = 42) -> list[dict]:
    """One QA per distinct frame, spread across sessions before taking more."""
    sessions: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in sorted(rows, key=lambda r: r["source_id"]):
        sessions[row["session_id"]].setdefault(row["frame_key"], row)
    groups = []
    for session in sorted(sessions, key=lambda s: sha256(f"{seed}:{s}".encode())):
        ordered = sorted(sessions[session].values(), key=lambda r: sha256(f"{seed}:{r['frame_key']}".encode()))
        groups.append(ordered)
    selected = []
    for index in range(max((len(group) for group in groups), default=0)):
        for group in groups:
            if len(selected) >= max_frames:
                return selected
            if index < len(group):
                selected.append(group[index])
    return selected


def safe_path(root: Path, relative: str) -> Path:
    root = root.resolve()
    result = (root / relative).resolve()
    if not result.is_relative_to(root):
        raise ValueError("Media path escapes output root")
    return result


def image_asset(raw: bytes, path: Path, max_edge: int) -> dict:
    from PIL import Image, ImageOps

    with Image.open(io.BytesIO(raw)) as source:
        if source.width * source.height > 40_000_000:
            raise ValueError("Image exceeds pixel limit")
        source.load()
        im = ImageOps.exif_transpose(source).convert("RGB")
    original = [im.width, im.height]
    im.thumbnail((max_edge, max_edge))
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, "JPEG", quality=92)
    return {"source_sha256": sha256(raw), "source_bytes": len(raw),
            "prepared_sha256": sha256(path.read_bytes()), "original_size": original,
            "prepared_size": [im.width, im.height]}


def make_example(row: dict, split: str, image: Path, media: dict) -> dict:
    return {
        "id": "pave-" + sha256((row["frame_key"] + ":" + row["source_id"]).encode())[:20],
        "task": "source_qa", "source": "PAVE", "source_id": row["source_id"],
        "sequence_id": "sanpo:" + row["session_id"], "split": split,
        "frames": [{"path": str(image.resolve()), "timestamp_ms": 0}],
        "objects": [], "action_mode": "walking", "question": row["question"],
        "target": row["target"],
        "provenance": {**SOURCES, "revision": PAVE_REVISION,
                       "source_row_sha256": row["source_row_sha256"],
                       "image_url": row["image_url"], "media": media,
                       "target_transform": "verbatim [assessment] paragraph only",
                       "supervision": "auxiliary accessibility QA; no hazard/ID/severity labels"},
    }


def load_resume_rows(output: Path) -> dict[str, dict]:
    cached = {}
    for split in ["train", "validation", "test"]:
        path = output / f"{split}.jsonl"
        if not path.exists():
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if index == len(lines)-1:
                    continue  # An interrupted final write is not a cached row.
                raise ValueError("Corrupt non-final resume manifest line") from None
            cached[row["id"]] = row
    return cached


def reusable_media(row: dict, split: str, image: Path, cached: dict[str, dict], max_edge: int):
    identifier = "pave-" + sha256((row["frame_key"] + ":" + row["source_id"]).encode())[:20]
    previous = cached.get(identifier, {})
    provenance = previous.get("provenance", {})
    media = provenance.get("media", {})
    if (previous.get("split") != split or provenance.get("revision") != PAVE_REVISION
            or provenance.get("source_row_sha256") != row["source_row_sha256"]
            or provenance.get("image_url") != row["image_url"] or not image.is_file()
            or sha256(image.read_bytes()) != media.get("prepared_sha256")):
        return None
    from PIL import Image
    try:
        original = media["original_size"]
        with Image.open(image) as im:
            im.load()
            if list(im.size) != media["prepared_size"] or max(im.size) != min(max(original), max_edge):
                return None
    except (KeyError, TypeError, ValueError, OSError):
        return None
    return media


def prepare(args) -> dict:
    cache = args.cache.resolve()
    train_raw, train_meta = cached_get(cache, "PAVE_train85.jsonl", PAVE_BASE + "PAVE_train85.jsonl", 20_000_000)
    eval_raw, eval_meta = cached_get(cache, "PAVE_val85.jsonl", PAVE_BASE + "PAVE_val85.jsonl", 2_000_000)
    test_raw, test_meta = cached_get(cache, "sanpo_test_session_ids.txt", SANPO_BASE + "splits/test_session_ids.txt", 100_000)
    train_ids, ids_meta = cached_get(cache, "sanpo_train_session_ids.txt", SANPO_BASE + "splits/train_session_ids.txt", 100_000)
    train, evaluation = parse_jsonl(train_raw), parse_jsonl(eval_raw)
    native_train = set(train_ids.decode().split())
    native_test = set(test_raw.decode().split())
    all_sessions = {r["session_id"] for r in train + evaluation}
    unknown = all_sessions - native_train - native_test
    if unknown:
        raise ValueError(f"PAVE references {len(unknown)} sessions absent from SANPO official splits")
    partitions, excluded = partition_rows(train, evaluation, native_test, args.seed)
    selected = {split: select_frames(rows, getattr(args, f"max_{split}_frames"), args.seed)
                for split, rows in partitions.items()}
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(), "source": SOURCES,
        "pave_revision": PAVE_REVISION, "metadata": [train_meta, eval_meta, test_meta, ids_meta],
        "task": "source_qa", "status": "metadata_only", "seed": args.seed,
        "native_train_rows": len(train), "native_eval_rows": len(evaluation),
        "excluded_sanpo_test_rows": len(excluded),
        "excluded_sanpo_test_sessions": sorted({row["session_id"] for row in excluded}),
        "available": {split: {"rows": len(rows), "sessions": len({r['session_id'] for r in rows})}
                      for split, rows in partitions.items()},
        "selected": {split: {"rows": len(rows), "sessions": len({r['session_id'] for r in rows})}
                     for split, rows in selected.items()},
        "limitations": ["Single-frame accessibility QA, not motion or candidate-action risk supervision.",
                        "PAVE QA text is existing model-generated supervision, not human-verified hazard ground truth.",
                        "No object IDs, exact danger types, unknown labels or calibrated severity targets.",
                        "Native PAVE distances and segmentation tokens are excluded.",
                        "PAVE official evaluation sessions are test-only, not training or model selection.",
                        "PAVE QA includes questions about several mobility needs, not exclusively blindness."],
    }
    cached_rows = {}
    if args.output.exists():
        if not args.resume:
            raise FileExistsError("Use a new output directory, or --resume after the prior preparation process has stopped")
        prior = json.loads((args.output/"data_report.json").read_text(encoding="utf-8"))
        if any(prior.get(key) != report[key] for key in ["pave_revision", "seed", "selected", "metadata"]):
            raise ValueError("Resume source revision, split selection or metadata differs")
        cached_rows = load_resume_rows(args.output)
    else:
        args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "data_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not args.download_media:
        return report
    total_bytes = 0
    budget = DownloadBudget(args.max_download_bytes)
    reused_images = 0
    pixel_splits: dict[str, str] = {}
    from PIL import Image
    try:
        for split, rows in selected.items():
            cache_for_split = {}
            pending_rows = []
            for row in rows:
                image = safe_path(args.output / "media", row["frame_key"].removesuffix(".png") + ".jpg")
                media = reusable_media(row, split, image, cached_rows, args.max_edge)
                if media is None:
                    pending_rows.append(row)
                else:
                    cache_for_split[row["frame_key"]] = media
            downloads = download_rows(pending_rows, args.workers, budget)
            with closing(downloads), (args.output / f"{split}.jsonl").open("w", encoding="utf-8") as destination:
                for row_index, row in enumerate(rows, 1):
                    image = safe_path(args.output / "media", row["frame_key"].removesuffix(".png") + ".jpg")
                    media = cache_for_split.get(row["frame_key"])
                    if media is None:
                        downloaded_row, raw = next(downloads)
                        if downloaded_row != row:
                            raise ValueError("Download/source row order mismatch")
                        total_bytes += len(raw)
                        media = image_asset(raw, image, args.max_edge)
                    else:
                        reused_images += 1
                    with Image.open(image) as im:
                        pixel_hash = sha256(str(im.size).encode() + im.convert("RGB").tobytes())
                    previous = pixel_splits.setdefault(pixel_hash, split)
                    if previous != split:
                        raise ValueError("Identical prepared pixels cross dataset splits")
                    destination.write(json.dumps(make_example(row, split, image, media), ensure_ascii=False) + "\n")
                    if row_index % 20 == 0 or row_index == len(rows):
                        print(f"Prepared {split} {row_index}/{len(rows)}; downloaded {total_bytes} bytes", flush=True)
        report.update(status="prepared", downloaded_bytes=total_bytes,
                      reused_images=reused_images,
                      prepared_images=sum(v["rows"] for v in report["selected"].values()),
                      image_policy={"max_edge": args.max_edge, "format": "JPEG", "quality": 92})
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, downloaded_bytes=budget.used)
        raise
    finally:
        (args.output / "data_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True, help="Pinned metadata cache (not a source bundle)")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; existing directories are refused")
    parser.add_argument("--download-media", action="store_true", help="Fetch only selected PNGs and save resized JPEGs")
    parser.add_argument("--max-train-frames", type=int, default=2000)
    parser.add_argument("--max-validation-frames", type=int, default=200)
    parser.add_argument("--max-test-frames", type=int, default=0, help="Default leaves official evaluation untouched")
    parser.add_argument("--max-download-bytes", type=int, default=12_000_000_000)
    parser.add_argument("--max-edge", type=int, default=896)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=6, help="Bounded parallel image downloads (1-12)")
    parser.add_argument("--resume", action="store_true", help="Reuse hash-verified media from a stopped prior attempt with identical selection")
    args = parser.parse_args()
    if min(args.max_train_frames, args.max_validation_frames, args.max_test_frames) < 0:
        parser.error("Frame limits must be nonnegative")
    if args.max_download_bytes <= 0 or not 224 <= args.max_edge <= 2048:
        parser.error("Positive download budget and max-edge between 224 and 2048 required")
    if not 1 <= args.workers <= 12:
        parser.error("workers must be between 1 and 12")
    if args.resume and not args.download_media:
        parser.error("--resume requires --download-media")
    print(json.dumps(prepare(args), indent=2))


if __name__ == "__main__":
    main()
