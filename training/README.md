# Stride Ahead model training

Training, data preparation, evaluation, and export scripts for Stride Ahead's
two models. This directory is self-contained: it includes the shared Gemma
response contract and checkpoint-loading code needed by the scripts.

| Model | Purpose | Training |
| --- | --- | --- |
| [RF-DETR Medium](RF%20Detr/README.md) | Detect and locate outdoor objects using labels and bounding boxes. | Fine-tuned on OOD v1: 7,999 training, 1,000 validation, and 1,000 test images across 22 classes. |
| [Gemma 4 12B](Gemma4-12B/README.md) | Interpret camera images together with RF detections and describe potential pedestrian hazards. | Rank-16 language-attention LoRA, 40 optimizer steps on 155 training and 22 validation examples from disjoint recording sessions. |

Gemma's targets are weak, AI-generated labels, not human-verified hazard ground
truth. Detection metrics and agreement with those labels do not establish
real-world warning accuracy. Alert priority and urgency are application rules,
not a trained injury-severity model. See [outdoor scope](OUTDOOR_SCOPE.md) and
[Gemma data provenance](Gemma4-12B/HAZARD_DATA.md).

## Contents

- `RF Detr/`: dataset download/audit, COCO preparation, training, evaluation,
  detector worker, configuration, CPU tests, and recorded experiment reports.
- `Gemma4-12B/`: source preparation, RF context attachment, weak-label generation
  and filtering, LoRA training, evaluation, merged/quantized export, shared
  model helpers, configuration, CPU tests, and recorded experiment reports.

The reports describe the original experiments; their absolute paths are
historical provenance. Datasets, trained weights, caches, and run outputs are
excluded from this Git directory. Download the completed model artifacts from
[Atlearia/Gemma_4_finetuned on Hugging Face](https://huggingface.co/Atlearia/Gemma_4_finetuned).
Keep each checkpoint's metadata, label names, and processor files together.

## Running training

Use Linux with Python 3.11 and a compatible CUDA GPU. The supplied recipes were
used on an H100; they are not CPU training recipes. Use separate Python
environments for the models, install their pinned requirements, and follow the
model-specific README files linked above. Paths in commands are examples:
choose your own data/output locations and a future deadline when required.

Training is independent of the Next.js application. These scripts do not start
training during an application build or publish model artifacts automatically.

## Lightweight verification

From the repository root, in a separate test environment:

```bash
python -m pip install -r training/requirements-test.txt
python -m pytest -q --import-mode=importlib -p no:cacheprovider \
  "training/RF Detr/tests" training/Gemma4-12B/tests
```

Importlib mode keeps the two models' similarly named test files separate. The
CPU suite uses synthetic fixtures and model stubs; it does not download weights,
train models, or establish GPU performance. The Gemma native checkpoint
round-trip script is a separate optional GPU check.
