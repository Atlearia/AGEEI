# Gemma 4 outdoor data: what is ready and what is missing

Audited 2026-10-09. The executable data path is **PAVE annotations + the original SANPO images**, both CC BY 4.0. Preparation completed for **300 training images and 60 validation images**, including the finished RF detector's predictions. Source QA is an intermediate input to privileged teacher labeling, not the student training objective. See [HAZARD_DATA.md](HAZARD_DATA.md) for the active task-contract workflow and its weak-label limitations. Completed source preparation does not establish completed LoRA training.

After teacher generation and a separate AI critic pass, the actual prepared LoRA dataset contains **155 train / 22 validation examples from 54 / six disjoint recording sessions**. Training contains 94 potential-hazard scenes, 56 with no confirmed hazard reported, and five unknown; validation contains 16, three and three. These are weak generated labels, not human hazard truth. The historical dataset was named `gemma-hazards-filtered-v2`; the 300/60 counts below describe the original source pool, not final optimizer inputs. Critic prompt SHA-256: `a16ba6c5ed9aecd4ad95345f205c1925a4dca48bc1b871a2afa075e582259404`.

## What these examples teach

Input: one real outdoor pedestrian-camera image and the existing PAVE question. Target: the existing prose inside the source's `[assessment] ... [/assessment]` span, unchanged. The JSONL task is `source_qa`. We remove the separate segmentation tokens and numerical distance section because those outputs depend on additional source modalities; we do not replace them with invented labels.

These paragraphs provide outdoor accessibility vocabulary and context to the teacher. They do **not** supply motion supervision, reliable proposed-action risk labels, RF tracking IDs, the exact product danger-category schema, or calibrated severity. Some questions concern wheelchairs and other mobility needs. It is not exclusively blind-person hazard data. Some questions reveal scene content, so good QA loss alone would not establish unprompted hazard detection quality.

The [primary paper, section 3.2](https://arxiv.org/html/2603.10703v1#S3.SS2), says the authors generated PAVE QA text with **GPT-5-nano from structured SANPO scene attributes**. The images are real and the source segmentation includes human annotations, but these QA paragraphs are existing synthetic supervision, not human-verified danger ground truth. The separate authorized teacher stage runs the base Gemma model on the H100 and generates additional weak hazard labels; it does not call a paid external teacher API.

The runtime must still enforce its response schema and validate object IDs. More data with real per-hazard labels and object associations is needed before claiming those abilities were learned through supervised fine-tuning. Do not convert every non-accessible semantic class into a hazard: PAVE can include sky among non-accessible classes.

## Exact source and splits

- [PAVE dataset card](https://huggingface.co/datasets/rafiibnsultan/PAVE), pinned revision `945dae763031eadcd77917890d2ffaf68624131f`.
- `PAVE_train85.jsonl`: **8,500 rows, 85 recording sessions**.
- `PAVE_val85.jsonl`: **600 rows, 6 separate sessions**. Reserved as **test**, never training or checkpoint selection.
- [SANPO images and license](https://github.com/google-research-datasets/sanpo_dataset#license--contact): original head-camera PNGs, each independently downloadable from the public Google Cloud bucket.

The real audit found **16 PAVE training sessions / 1,600 rows in SANPO's official test list**. The converter excludes these rows from training and validation. Of the remaining 69 recording sessions, a seeded group split assigns:

| Partition | Available frames / QA rows | Sessions | Active pilot |
|---|---:|---:|---:|
| Train | 6,200 | 62 | 300 |
| Validation | 700 | 7 | 60 |
| Test, official PAVE evaluation | 600 | 6 | 0 downloaded by default |

The pilot samples across sessions before adding more frames from one session and takes at most one QA per distinct frame. This is a budgeted domain-adaptation experiment, not an empirically proven optimum. Expand toward the 6,200 eligible frames only after comparing with the untouched base model. Group separation prevents the same recording from appearing in training and validation; it cannot prove different recordings contain no repeated route. A few validation scenes were reviewed during teacher-prompt development, so validation is not an untouched final test.

The converter verifies pinned metadata hashes, path boundaries, image decoding, and exact prepared-pixel duplication across splits. It emits per-image hashes, source URLs, source row hashes, transformations, terms, and split counts. It stops on a failure and marks `data_report.json` as `failed`; do not train on a partial failed output directory.

## Commands on the GPU machine

Use the Gemma environment containing Pillow. Run from the repository root and set `STRIDE_DATA` as in [README.md](README.md). Keep generated data excluded from Git. The first command inspects metadata only:

```bash
python training/Gemma4-12B/prepare_data.py \
  --cache "$STRIDE_DATA/gemma-pave-cache" \
  --output "$STRIDE_DATA/gemma-pave-plan" \
  --max-train-frames 300 --max-validation-frames 60
```

Prepare the pilot into a **new** directory:

```bash
python training/Gemma4-12B/prepare_data.py \
  --cache "$STRIDE_DATA/gemma-pave-cache" \
  --output "$STRIDE_DATA/gemma-pave-raw-source" \
  --download-media \
  --max-train-frames 300 --max-validation-frames 60 \
  --max-test-frames 0 --max-download-bytes 2500000000 --workers 6
```

The original 2,208 x 1,242 PNGs are decoded in memory and saved as JPEGs with maximum edge 896; source PNGs are not kept. The active 360-image preparation transferred **1,212,521,937 source bytes** and produced **43,983,900 bytes of JPEGs**. The budget is a hard upper bound, not an estimate. No full SANPO archive is needed. Reserve additional disk space for model weights, optimizer state and checkpoints.

Outputs: `train.jsonl`, `validation.jsonl`, empty `test.jsonl` by default, `media/`, and `data_report.json`. Frame paths are absolute paths on the machine where preparation runs. Re-run preparation on the training machine, or explicitly relocate those paths when transferring data. A single image has timestamp 0; no video timing is invented.

The original run retained a private backup of the prepared source dataset and pinned annotations. Those archives, source media and caches are not included in this source package. Use the preparation commands to obtain data and create manifests for the current machine.

## Why WAD is not the default download

[WAD / WalkVLM](https://walkvlm2024.github.io/) is a stronger match for short walking clips, descriptions and reminders. Its public source does not provide a ready copy of our RF-ID/per-hazard product contract. The official project links `wad_dataset_v2.tar`; inspection showed **129,317,781,901 bytes** (about 120.4 GiB), gzip-compressed despite the `.tar` suffix. The [author repository](https://github.com/xiaoyuan1996/walkvlm/tree/main/wad_dataset) points to an older archive of similar size. HTTP ranges are supported, but an ordinary gzip stream does not provide random access to selected inner frames.

No explicit dataset license was verified in the inspected project page, dataset README, or repository tree. A public URL and a paper citation do not establish a data license. We downloaded only archive headers and 1,024 signature bytes. No WAD archive, training media or annotation body was downloaded, and no adapter was trained on WAD.

To use WAD later, obtain the author's dataset terms and a practical media/annotation subset or enough storage. Inspect the real annotation schema and preserve recording-level splits before writing a source-specific converter. Do not invent a parser for uninspected annotation files or use published test examples as training data.

Reproduce the source audit without downloading archives:

```bash
python training/Gemma4-12B/inspect_data.py --output "$STRIDE_DATA/gemma-source-audit.json"
```

## Attribution and evaluation

Cite **Sultan et al., WalkGPT: Grounded Vision-Language Conversation with Depth-Aware Segmentation for Pedestrian Navigation (2026)** for PAVE and **Waghmare et al., SANPO: A Scene Understanding, Accessibility, and Human Navigation Dataset (WACV 2025)** for the images. The converter records both sources and the resize/target extraction changes under CC BY 4.0.

Use grouped validation for pilot selection. Run the frozen base model on the same evaluation prompts before comparison. QA loss or fluency is not a safety metric; separately evaluate missed hazards, benign-scene false alerts, correct object-ID grounding, output validity and latency with properly labeled product examples. No existing PAVE score establishes a calibrated 0-100 danger score.
