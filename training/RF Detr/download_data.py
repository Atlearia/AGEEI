#!/usr/bin/env python3
"""Download the pinned OOD v1 COCO export. Requires roboflow and ROBOFLOW_API_KEY.

No download happens when imported or with --help. Alternatively manually export
COCO JSON at https://universe.roboflow.com/fpn/ood-pbnro/dataset/1 and use
prepare_data.py --source /path/to/export --output /path/to/prepared --ood-v1.
"""
import argparse
import contextlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="New download directory; refuses an existing path")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.exit(2, "Refusing to overwrite an existing download directory.\n")
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        parser.exit(2, "Set ROBOFLOW_API_KEY in the environment or manually export COCO JSON from the dataset page.\n")
    try:
        from roboflow import Roboflow
    except ImportError:
        parser.exit(2, "Install the optional downloader: python -m pip install roboflow\n")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        # SDK exceptions/progress can include signed URLs or credentials. Do not
        # echo raw SDK output, tokens, or exception bodies into public logs.
        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            project = Roboflow(api_key=key).workspace("fpn").project("ood-pbnro")
            dataset = project.version(1).download("coco", location=str(output), overwrite=False)
    except Exception as exc:
        parser.exit(2, f"OOD download failed ({type(exc).__name__}); check dataset access and network. A partial output may remain; use a new path to retry.\n")
    print(json.dumps({"downloaded_to": str(dataset.location), "dataset": "FPN/OOD", "version": 1,
                      "format": "COCO JSON", "license": "CC BY 4.0",
                      "source": "https://universe.roboflow.com/fpn/ood-pbnro/dataset/1"}, indent=2))


if __name__ == "__main__":
    main()
