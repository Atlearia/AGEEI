# RF-DETR Medium training on one H100

This folder fine-tunes **official RF-DETR Medium** to locate outdoor walking obstacles for blind and low-vision pedestrians. It includes an online-data downloader, COCO validation and preparation, training, timing, and checkpoint evaluation. Target: Linux, Python 3.11, **one H100 80 GB**, with data and checkpoints on persistent storage. See the [outdoor project scope](../OUTDOOR_SCOPE.md).

RF-DETR supplies boxes to the Stride Ahead Gemma 4 pipeline. Gemma handles observations, hazard reasoning, object-ID association and warning text; see [Gemma LoRA training](../Gemma4-12B/README.md).

## What trains, and on which data

| Component | Training decision | Supervision and amount |
|---|---|---|
| RF-DETR Medium | Fine-tune pretrained detector, including backbone at a smaller learning rate | [OOD v1 by FPN](https://universe.roboflow.com/fpn/ood-pbnro/dataset/1): published **8,000 train / 1,000 validation / 1,000 test**, 22 obstacle classes, 29,779 boxes overall; publisher license **CC BY 4.0**. |
| Tracker | No training in this package | Uses detections across frames; implement/configure in the application. |
| Gemma 4 12B | Task-specific LoRA in a separate folder | See its data audit; auxiliary QA does not supply full hazard supervision. |
| Severity score / speech policy | No supervised severity model is trained here | Selected data has **no validated per-object severity labels**. Risk reasoning remains a separately evaluated application behavior. |

OOD covers useful classes such as stairs, curbs, poles, benches, people, bicycles and vehicles. The full training split is a sensible starting point for transfer learning under this deadline; the number alone does not establish adequacy. Use per-class validation metrics to identify weak classes. This is outdoor coverage, not a complete indoor/outdoor hazard vocabulary. See [DATA_SOURCES.md](DATA_SOURCES.md) for sources, alternatives and limitations.

Preparation keeps the publisher's split membership except for removing pixel-identical duplicates, prioritizing test over validation over train. Invalid images fail by default. Actual retained counts and class distribution are recorded in `data_report.json`; they may be smaller than the advertised counts. No manual annotation is needed. Near-duplicate frames and route independence cannot be established from the provided metadata.

## H100 quick start

Clone the repository onto persistent storage on the GPU host. The following commands are for **Linux Bash on that host**. Start at the repository root, then enter this folder as shown below. Relative data and output paths resolve to the ignored `training/data` and `training/runs` directories.

```bash
cd "training/RF Detr" # from the cloned repository root
python3.11 -m venv ../.venv-rfdetr
source ../.venv-rfdetr/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip check
CUDA_VISIBLE_DEVICES=0 python check_environment.py --require-h100
```

Use a provider image with a driver compatible with the selected CUDA wheel. This creates a separate environment; do not replace the application's environment. The environment check validates package/config compatibility and performs tiny CUDA operations without downloading weights. Dependency resolution and GPU execution still need to succeed on the actual host; this is not a fully GPU-tested lockfile. After a successful run, save `python -m pip freeze` alongside the checkpoints.

### 1. Download and audit OOD

Set `ROBOFLOW_API_KEY` through your provider's secret environment settings, or export COCO JSON manually from the [version-1 dataset page](https://universe.roboflow.com/fpn/ood-pbnro/dataset/1). Do not put a key in source code or commit it.

```bash
python download_data.py --output ../data/ood-raw
python prepare_data.py \
  --source ../data/ood-raw \
  --output ../data/ood --ood-v1
```

For a manual export, point `--source` at the extracted directory containing `train`, `valid` (or `val`) and `test`, each with `_annotations.coco.json`. Both scripts refuse to overwrite existing outputs. Review `../data/ood/data_report.json`. The preparation script audits decoded dimensions, box bounds, category mappings, duplicates and class coverage, then copies images to a clean COCO layout. Keep the raw export and attribution.

Use all retained training images. `--max-train` is available for a small troubleshooting subset; do not use a random tiny subset as the final training set. `--invalid-images drop` explicitly removes whole invalid images and records them; use only after reviewing an audit failure.

### 2. Measure one epoch, then train

```bash
python train.py --dry-run
CUDA_VISIBLE_DEVICES=0 python train.py \
  --data ../data/ood \
  --output ../runs/medium-ood --benchmark
```

The benchmark performs one **real full training epoch plus validation** in `../runs/medium-ood-benchmark`. Read its `timing.json` for elapsed time, peak allocated GPU memory and a clearly labeled projection. It leaves the main run directory free and is not automatically resumed as the main run. `--dry-run` only checks/prints configuration; it does not audit data, import ML packages or measure throughput.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --data ../data/ood \
  --output ../runs/medium-ood
```

The default configuration in `configs/h100_medium.json` uses:

- Native **576px** resolution; BF16; microbatch **8**, accumulation **2**, effective batch **16**.
- **30 epochs maximum**, validation every epoch, EMA weights and early stopping after five stagnant validation checks. This is a starting budget, not a claim that 30 epochs is optimal.
- Learning rate **1e-4** for the rest of the detector and **1e-5** for its backbone, with one warmup epoch and cosine decay. Training starts from pretrained weights.
- Fixed resolution and standard torchvision augmentation to keep the first experiment predictable. Test evaluation is disabled during fitting.

Use `--epochs 15` for a shorter initial run if the measured projection exceeds your budget. Keep all data and the best validation checkpoint. If GPU memory is insufficient, try `--batch-size 4 --grad-accum-steps 4`. If the container has shared-memory/data-worker errors, try `--num-workers 0`; expect lower throughput.

To resume an interrupted run with optimizer state:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --data ../data/ood --output ../runs/medium-ood \
  --resume ../runs/medium-ood/last.ckpt
```

Use your own trusted checkpoint and the original data/configuration. An inference `.pth` is not the resume checkpoint. Do not load a different label schema into an interrupted run.

### 3. Evaluate and retain the selected model

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
  --data ../data/ood \
  --checkpoint ../runs/medium-ood/checkpoint_best_total.pth \
  --split valid

# After checkpoint and deployment-threshold selection, report the held-out test once.
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
  --data ../data/ood \
  --checkpoint ../runs/medium-ood/checkpoint_best_total.pth \
  --split test
```

Evaluation writes COCO AP50, AP50:95, AR and per-class metrics to `evaluation-valid/metrics.json` or `evaluation-test/metrics.json`. It uses predictions without a deployment confidence cutoff. These are object detection metrics; AR@100 is not the recall of alerts at your chosen UI threshold. Inspect missed stairs/curbs and false detections on validation, select alert thresholds there, and measure live-video latency separately.

Keep `checkpoint_best_total.pth`, `last.ckpt`, `class_names.json`, `environment.json`, `run_plan.json`, `data_fingerprint.json`, the data report, metrics and source attribution. The native trainer saves additional logs/configs. For inference with this pinned release:

```python
from rfdetr import RFDETRMedium

detector = RFDETRMedium.from_checkpoint(
    "../runs/medium-ood/checkpoint_best_total.pth", device="cuda"
)
detections = detector.predict("frame.jpg", threshold=0.25)
# 0.25 is an example only; tune it on validation for your alert behavior.
```

Use checkpoint class names when integrating; the prepared COCO IDs are 1..N while custom RF-DETR prediction IDs are 0..N-1. Detection confidence measures confidence in object detection, **not danger severity**. This folder does not implement live capture, tracking, risk reasoning or speech.

## Gemma inference

See [Gemma setup](../Gemma4-12B/README.md). The detector's labels and normalized boxes are inputs to Gemma; the model does not replace detector training or generate new box coordinates. The shared Gemma worker is included with its training tools for evaluation and export checks.

## Time budget on one H100

These are **unmeasured planning estimates**, assuming an 80 GB H100, local images, working CUDA libraries and the configurations here:

| Work | Planning allowance |
|---|---|
| RF environment, export download and data audit | Approximately 30-60 minutes; account access, network and installation failures can add time. |
| RF-DETR Medium, up to 30 epochs over roughly 8,000 training images | Reserve roughly **1-3 hours**, then replace this estimate with the included one-epoch measurement. Validation and storage can materially change it. |

Use the measured RF full-epoch time multiplied by the remaining epoch budget, allowing for early stopping and first-epoch overhead. Pretrained model download and inference integration need their own time allowance.

With six hours total, reserve the last two hours for integration, error review and demo rehearsal. Train RF-DETR separately from serving the two inference models on the same GPU.

## Local verification and files

The detector package has CPU dataset, configuration and HTTP worker tests using synthetic fixtures and stubbed models. In the original H100 experiment, a full one-epoch benchmark completed in 137.28 seconds of fit time and the main fitting stage completed in 1,822.98 seconds. A NumPy JSON-serialization issue in standalone evaluation was fixed before recovering validation and test metrics. See the included [evaluation report](reports/EVALUATION.md) and full metrics beside it.

Run lightweight tests without either model installed:

```bash
python -m pip install -r ../requirements-test.txt
python -m pytest -q --import-mode=importlib -p no:cacheprovider tests
```

`inspect_sources.py` can inspect current Hugging Face model/dataset metadata and Dataset Viewer schemas using public APIs. `--help` lists its options. It records access failures rather than silently substituting a different dataset; it does not download image archives.

The implementation was checked against [RF-DETR 1.11.2 source](https://github.com/roboflow/rf-detr/tree/1.11.2) and uses RF-DETR's own trainer.
