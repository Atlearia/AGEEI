"""Compare base Gemma and an adapter on held-out teacher labels, never human accuracy.

The base weights load once. Base generation temporarily disables the loaded LoRA.
Outputs and running proxy metrics are saved after every inference, without images.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import statistics
import sys
import threading
import time

WORKER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKER_DIR))
from contract import Assessment, MODEL_ID, MODEL_REVISION, load_json, validate_assessment


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def deadline_timestamp(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("--deadline-utc requires an explicit timezone")
    return parsed.timestamp()


def actionable(assessment, frame_count):
    if assessment.status != "ok" or assessment.scene_uncertain:
        return None
    return [hazard for hazard in assessment.hazards
            if not hazard.uncertain and hazard.frame_index == frame_count - 1]


def categories_by_id(hazards):
    result = {}
    for hazard in hazards or []:
        for track_id in hazard.track_ids:
            result.setdefault(track_id, set()).add(hazard.danger_type)
    return result


def score_prediction(raw, target, *, frame_count, track_ids):
    """Count abstention/invalid output separately, including false negatives for known targets."""
    reference = validate_assessment(target, frame_count=frame_count, track_ids=track_ids)
    stats = {"valid_json": False, "valid_schema": False, "valid_references": False,
             "valid_contract": False, "determinate": False, "target_determinate": False,
             "hazard_presence_agrees": None, "id_true_positive": 0, "id_false_positive": 0,
             "id_false_negative": 0, "category_matched_ids": 0, "category_agreeing_ids": 0}
    predicted = None
    try:
        value = load_json(raw.strip())
        stats["valid_json"] = True
        candidate = Assessment.model_validate(value, strict=True)
        stats["valid_schema"] = True
        stats["valid_references"] = all(
            hazard.frame_index < frame_count and len(hazard.track_ids) == len(set(hazard.track_ids))
            and set(hazard.track_ids).issubset(track_ids)
            and (hazard.frame_index == frame_count - 1 or not hazard.track_ids)
            for hazard in candidate.hazards)
        predicted = validate_assessment(value, frame_count=frame_count, track_ids=track_ids)
        stats["valid_contract"] = True
    except (ValueError, TypeError):
        pass
    expected_hazards = actionable(reference, frame_count)
    predicted_hazards = actionable(predicted, frame_count) if predicted else None
    stats["determinate"] = predicted_hazards is not None
    stats["target_determinate"] = expected_hazards is not None
    if expected_hazards is not None:
        # An abstention does not count as agreement with a determinate teacher label.
        stats["hazard_presence_agrees"] = predicted_hazards is not None and bool(predicted_hazards) == bool(expected_hazards)
        expected, actual = categories_by_id(expected_hazards), categories_by_id(predicted_hazards)
        wanted, found = set(expected), set(actual)
        stats.update(id_true_positive=len(wanted & found), id_false_positive=len(found - wanted),
                     id_false_negative=len(wanted - found), category_matched_ids=len(wanted & found),
                     category_agreeing_ids=sum(expected[i] == actual[i] for i in wanted & found))
    return stats


def normalize_exact_json_fence(raw):
    """Offline content comparison only; never scrape JSON from prose or prefixes."""
    fenced = re.fullmatch(r"\s*```(?:json)?\s*\n([\s\S]*?)\n```\s*", raw)
    return fenced.group(1) if fenced else raw


def score_content_proxy(raw, target, *, frame_count, track_ids):
    """Separate teacher-content proxy after removing only one complete JSON fence."""
    return score_prediction(normalize_exact_json_fence(raw), target,
                            frame_count=frame_count, track_ids=track_ids)


CONTENT_PROXY_KEY = "content_proxy_after_exact_fence_normalization"


def summarize(records, *, metrics_key="metrics"):
    reports = {}
    for mode in ("base", "adapter"):
        rows = [row for row in records if row["mode"] == mode]
        if not rows:
            continue
        count = len(rows)
        sums = {key: sum(int(row[metrics_key].get(key) or 0) for row in rows)
                for key in ("valid_json", "valid_schema", "valid_references", "valid_contract", "determinate",
                            "target_determinate", "hazard_presence_agrees", "id_true_positive", "id_false_positive",
                            "id_false_negative", "category_matched_ids", "category_agreeing_ids")}
        tp, fp, fn = (sums[key] for key in ("id_true_positive", "id_false_positive", "id_false_negative"))
        latency = [row["generation_seconds"] for row in rows if row.get("generation_seconds") is not None]
        reports[mode] = {
            "examples": count, "counts": sums,
            **{key + "_rate": sums[key] / count for key in ("valid_json", "valid_schema", "valid_references", "valid_contract", "determinate")},
            "hazard_presence_agreement": sums["hazard_presence_agrees"] / sums["target_determinate"] if sums["target_determinate"] else None,
            "track_id_precision": tp / (tp + fp) if tp + fp else None,
            "track_id_recall": tp / (tp + fn) if tp + fn else None,
            "category_set_agreement_on_matched_ids": sums["category_agreeing_ids"] / sums["category_matched_ids"] if sums["category_matched_ids"] else None,
            "generation_seconds_mean": statistics.mean(latency) if latency else None,
            "generation_seconds_median": statistics.median(latency) if latency else None,
            "generation_seconds_max": max(latency) if latency else None,
            "latency_examples": len(latency),
        }
    return reports


def write_report(output, metadata, records, status):
    paired = {row["example_id"] for row in records if row["mode"] == "base"} & {
        row["example_id"] for row in records if row["mode"] == "adapter"}
    report = {**metadata, "status": status, "updated_utc": datetime.now(timezone.utc).isoformat(),
              "paired_comparison_examples": len(paired), "results": summarize(records),
              CONTENT_PROXY_KEY: summarize(records, metrics_key=CONTENT_PROXY_KEY)}
    temporary = output / "report.tmp.json"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output / "report.json")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--max-examples", type=int, default=12)
    parser.add_argument("--deadline-utc", required=True, help="Absolute hard deadline including data checks, model loading and evaluation")
    parser.add_argument("--only-adapter", action="store_true", help="Skip base generation when the remaining budget is short")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    deadline = deadline_timestamp(args.deadline_utc)
    if deadline <= time.time() or not 1 <= args.max_examples <= 100:
        parser.error("Deadline must be future and max-examples must be 1-100")
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "kind": "gemma-teacher-label-proxy-evaluation-v1", "model_id": MODEL_ID,
        "model_revision": args.revision, "adapter": str(args.adapter.resolve()),
        "created_utc": datetime.now(timezone.utc).isoformat(), "deadline_utc": args.deadline_utc,
        "requested_examples": args.max_examples, "only_adapter": args.only_adapter, "seed": args.seed,
        "maximum_new_tokens": 384,
        "interpretation": "Agreement with held-out teacher pseudo-labels, not human ground truth, navigation safety or real-world hazard accuracy.",
        "metric_scope": "Current, non-uncertain hazards only. Invalid/unknown predictions count as non-agreement and missing target IDs for determinate targets. Unknown targets are excluded from agreement denominators.",
        "content_proxy_scope": "Separate teacher-agreement scores after removing only one complete JSON code fence. The strict raw JSON/contract results remain unchanged. No numeric prefixes, prose or arbitrary JSON extraction are accepted. Content proxy improvement is distinct from formatting improvement and is not human hazard accuracy.",
        "latency_scope": "Generation and decoding for all completed outputs, including valid unknown and invalid output; excludes processor encoding and model loading.",
    }
    records = []
    write_report(args.output, metadata, records, "initializing")

    def hard_stop():
        try:
            (args.output / "deadline_status.json").write_text(json.dumps({
                "status": "hard_deadline_reached", "partial_report": "report.json",
                "message": "Last completed results are retained; no score is claimed for interrupted inference."}), encoding="utf-8")
        finally:
            os._exit(124)

    timer = threading.Timer(max(0, deadline - time.time()), hard_stop)
    timer.daemon = True
    timer.start()
    try:
        train = load_module("gemma_evaluation_training_helpers", Path(__file__).with_name("train.py"))
        rows = []
        identifiers = set()
        manifest_bytes = args.validation.read_bytes()
        metadata["validation_manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
        for line in manifest_bytes.decode("utf-8-sig").splitlines():
            if not line.strip():
                continue
            row = train.validate_row(load_json(line), args.validation.resolve(), "validation", allow_weak_labels=True)
            if row["id"] in identifiers:
                raise ValueError("Duplicate validation example ID")
            identifiers.add(row["id"])
            if row["task"] == "hazard_assessment":
                rows.append(row)
        random.Random(args.seed).shuffle(rows)
        rows = rows[:args.max_examples]
        if not rows:
            raise ValueError("Validation manifest has no hazard assessment targets")
        metadata["selected_examples"] = len(rows)
        metadata["selected_ids"] = [row["id"] for row in rows]
        write_report(args.output, metadata, records, "loading_model")
        worker = load_module("gemma_evaluation_worker_helpers", WORKER_DIR / "worker.py")
        runtime = worker.GemmaRuntime(args.model, args.revision, args.adapter)
        setup_started = time.perf_counter()
        runtime.load()
        if not args.only_adapter and not hasattr(runtime.model, "disable_adapter"):
            raise ValueError("Base comparison requires the unmerged LoRA adapter directory/file; use --only-adapter for a merged full checkpoint")
        metadata["model_load_seconds"] = time.perf_counter() - setup_started
        metadata["supervision_quality"] = runtime.manifest.get("supervision_quality", "unknown")
        import torch
        modes = ("adapter",) if args.only_adapter else ("base", "adapter")
        with (args.output / "predictions.jsonl").open("x", encoding="utf-8") as audit:
            for row in rows:
                if deadline - time.time() < 5:
                    write_report(args.output, metadata, records, "deadline_partial")
                    return 124
                _, messages = train.messages_for_row(row)
                inputs = runtime.processor.apply_chat_template(messages, add_generation_prompt=True,
                    enable_thinking=False, tokenize=True, return_dict=True, return_tensors="pt")
                input_length = inputs["input_ids"].shape[-1]
                if input_length > runtime.max_input_tokens:
                    raise ValueError("Validation example exceeds the serving token budget")
                inputs = inputs.to(runtime.model.device)
                for mode in modes:
                    context = runtime.model.disable_adapter() if mode == "base" else nullcontext()
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    with context, torch.inference_mode():
                        generated = runtime.model.generate(**inputs, max_new_tokens=384, do_sample=False, use_cache=True)
                    raw = runtime.processor.decode(generated[0, input_length:], skip_special_tokens=True)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - started
                    metrics = score_prediction(raw, row["target"], frame_count=len(row["frames"]),
                                               track_ids={box["track_id"] for box in row["objects"]})
                    content_metrics = score_content_proxy(raw, row["target"], frame_count=len(row["frames"]),
                                                          track_ids={box["track_id"] for box in row["objects"]})
                    record = {"example_id": row["id"], "source": row["source"], "source_id": row["source_id"],
                              "mode": mode, "raw_prediction": raw, "teacher_target": row["target"],
                              "metrics": metrics, CONTENT_PROXY_KEY: content_metrics,
                              "content_proxy_exact_fence_removed": normalize_exact_json_fence(raw) != raw,
                              "generation_seconds": elapsed}
                    records.append(record)
                    audit.write(json.dumps(record, allow_nan=False) + "\n")
                    audit.flush()
                    write_report(args.output, metadata, records, "running")
                    print(json.dumps({"example_id": row["id"], "mode": mode,
                                      "valid_contract": metrics["valid_contract"], "generation_seconds": round(elapsed, 3)}), flush=True)
        write_report(args.output, metadata, records, "completed")
        return 0
    except Exception:
        write_report(args.output, metadata, records, "failed")
        raise
    finally:
        timer.cancel()


if __name__ == "__main__":
    raise SystemExit(main())
