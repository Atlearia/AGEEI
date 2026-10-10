#!/usr/bin/env python3
"""Read HF model metadata and Dataset Viewer schemas, without downloading weights/images."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


def fetch_json(url: str, timeout: float = 25.0) -> dict:
    headers = {"User-Agent": "AGEII-training-source-check/1.0"}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        with urlopen(Request(url, headers=headers), timeout=timeout) as response:
            return json.load(response)
    except HTTPError as exc:
        return {"error": f"HTTP {exc.code}"}
    except (URLError, TimeoutError, ValueError, OSError) as exc:
        # Do not print request headers or potentially credential-bearing exception text.
        return {"error": type(exc).__name__}


def inspect_model(repo_id: str) -> dict:
    raw = fetch_json("https://huggingface.co/api/models/" + quote(repo_id, safe="/"))
    if "error" in raw:
        return {"id": repo_id, **raw}
    card = raw.get("cardData") or {}
    return {
        "id": raw.get("id", repo_id), "revision": raw.get("sha"),
        "license": card.get("license"), "gated": raw.get("gated"),
        "pipeline_tag": raw.get("pipeline_tag"), "safetensors": raw.get("safetensors"),
    }


def inspect_dataset(repo_id: str) -> dict:
    raw = fetch_json("https://huggingface.co/api/datasets/" + quote(repo_id, safe="/"))
    if "error" in raw:
        return {"id": repo_id, **raw}
    card = raw.get("cardData") or {}
    result = {
        "id": repo_id, "revision": raw.get("sha"), "license": card.get("license"),
        "gated": raw.get("gated"),
        "note": "License metadata alone does not replace dataset access conditions or upstream licenses.",
    }
    base = "https://datasets-server.huggingface.co/"
    splits = fetch_json(base + "splits?" + urlencode({"dataset": repo_id}))
    result["splits"] = splits
    result["size"] = fetch_json(base + "size?" + urlencode({"dataset": repo_id}))
    if splits.get("splits"):
        split = splits["splits"][0]
        params = {"dataset": repo_id, "config": split["config"], "split": split["split"],
                  "offset": 0, "length": 1}
        rows = fetch_json(base + "rows?" + urlencode(params))
        result["preview_schema"] = rows.get("features", [])
        result["num_rows_total"] = rows.get("num_rows_total")
        if "error" in rows:
            result["preview_error"] = rows["error"]
        # Summarize column types without printing images, encoded bytes or long model-generated prose.
        if rows.get("rows"):
            result["first_row_value_types"] = {
                key: type(value).__name__ for key, value in rows["rows"][0].get("row", {}).items()
            }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", help="HF model ID; may be repeated")
    parser.add_argument("--dataset", action="append", help="HF dataset ID; may be repeated")
    parser.add_argument("--output", type=Path, help="Save JSON to a new file instead of stdout")
    args = parser.parse_args()
    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "models": [inspect_model(x) for x in (args.model or ["google/gemma-4-12B-it"])],
        "datasets": [inspect_dataset(x) for x in (args.dataset or [])],
    }
    encoded = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
        print(f"Saved source metadata: {args.output}")
    else:
        print(encoded, end="")
    # Viewer failures are surfaced in the report; a failed top-level repo check is a failing exit.
    return int(any("error" in x for x in report["models"] + report["datasets"]))


if __name__ == "__main__":
    raise SystemExit(main())
