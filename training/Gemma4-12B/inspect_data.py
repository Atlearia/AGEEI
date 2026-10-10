"""Inspect the primary WAD/PAVE sources without downloading dataset archives."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import urllib.request

from prepare_data import bounded_get, PAVE_BASE, sha256

WAD_ARCHIVE = "https://sprproxy-1258344707.cos.ap-shanghai.myqcloud.com/seraphyuan/ilabel/blind_vlm/wad_dataset_v2.tar"
WAD_TREE = "https://api.github.com/repos/xiaoyuan1996/walkvlm/git/trees/main?recursive=1"


def inspect() -> dict:
    report = {"checked_at": datetime.now(timezone.utc).isoformat(), "sources": {}}
    for name, url, cap in [
        ("pave_card", PAVE_BASE + "README.md", 100_000),
        ("sanpo_card", "https://raw.githubusercontent.com/google-research-datasets/sanpo_dataset/main/README.md", 100_000),
        ("wad_readme", "https://raw.githubusercontent.com/xiaoyuan1996/walkvlm/main/wad_dataset/README.md", 100_000),
    ]:
        raw = bounded_get(url, cap)
        report["sources"][name] = {"url": url, "sha256": sha256(raw), "text": raw.decode()}
    with urllib.request.urlopen(urllib.request.Request(WAD_ARCHIVE, method="HEAD"), timeout=30) as response:
        metadata = {key: response.headers.get(key) for key in ["Content-Length", "Content-Type", "ETag", "Last-Modified", "Accept-Ranges"]}
    with urllib.request.urlopen(urllib.request.Request(WAD_ARCHIVE, headers={"Range": "bytes=0-1023"}), timeout=30) as response:
        # Bounded even if a server ignores Range: close after at most 1024 bytes.
        signature = response.read(1024)
        metadata["range_status"] = response.status
        metadata["range_header"] = response.headers.get("Content-Range")
    metadata.update(url=WAD_ARCHIVE, inspected_bytes=len(signature), gzip_magic=signature[:2] == b"\x1f\x8b")
    report["sources"]["wad_archive"] = metadata
    tree = json.loads(bounded_get(WAD_TREE, 1_000_000))
    report["sources"]["wad_repository"] = {
        "url": WAD_TREE, "revision": tree["sha"],
        "license_named_paths": [item["path"] for item in tree["tree"]
                                if "license" in item["path"].lower() or "terms" in item["path"].lower()],
    }
    report["decisions"] = {
        "WAD": "Relevant video/reminder data; no dataset terms verified in the inspected sources. Large gzip archive is not a selective-media source. No archive downloaded.",
        "PAVE": "Licensed auxiliary image accessibility QA; source paths separately verified by prepare_data.py. Does not supply the product hazard/ID/severity schema.",
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New JSON audit file")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite existing audit")
    report = inspect()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "decisions": report["decisions"]}, indent=2))


if __name__ == "__main__":
    main()
