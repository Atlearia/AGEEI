# OOD v1 RF-DETR Medium evaluation

Completed 2026-10-09. The full fine-tuning fit finished successfully in **1,822.98 seconds (30.38 minutes)**, with 6.30 GiB peak allocated GPU memory. The log reaches epoch 16 (zero based), so the fit finished before its 30-epoch cap. The requested-epoch field in `timing.json` is a cap, not the number completed.

The original workflow stopped after validation evaluation because NumPy `int64` category IDs could not be written to JSON. The model checkpoint was already saved. We fixed only metric serialization, verified a regression test, and ran validation again followed by the first held-out test evaluation. No retraining, checkpoint editing, threshold tuning or test-based model selection occurred.

| Metric | Validation, 1,000 images | Test, 1,000 images |
|---|---:|---:|
| COCO AP at IoU 0.50:0.95 | 0.53415 | 0.54938 |
| AP at IoU 0.50 | 0.72258 | 0.73967 |
| Average recall, max 100 detections | 0.72954 | 0.73873 |
| Small-object AP | 0.18971 | 0.14851 |
| Medium-object AP | 0.37603 | 0.29015 |
| Large-object AP | 0.61547 | 0.62833 |

Selected test classes:

| Class | Test boxes | AP 0.50:0.95 | AP 0.50 | AR 100 |
|---|---:|---:|---:|---:|
| Curb | 68 | 0.35724 | 0.47364 | 0.67353 |
| Pole | 104 | 0.26955 | 0.47224 | 0.62212 |
| Stairs | 87 | 0.45728 | 0.66141 | 0.74483 |
| Person | 387 | 0.25602 | 0.38986 | 0.64419 |
| Warning column | 167 | 0.59336 | 0.90055 | 0.67784 |

These are detection/localization metrics, **not danger accuracy or the probability an alert is correct**. Small objects, poles and people remain weaknesses. There is no measured matched-label-space baseline here, so these results do not quantify improvement over the pretrained detector. OOD lacks entirely unannotated images and verified recording-disjoint splits; these scores do not establish clear-path false-alert rates or unseen-route performance.

Evaluation used the fixed selected checkpoint, RF-DETR 1.11.2, BF16 inference, batch size 8, confidence cutoff 0 and COCO maxDets 100. Prediction loops took 38.42 seconds for validation and 40.14 seconds for test on the H100; this offline batch throughput is not live-video end-to-end latency.

## Artifacts

- Checkpoint SHA-256: `da22cd0f3c3389716a2a2f402c3bb54dc16eb84ecc821c445b9bd8518359cabf`; 134,105,701 bytes.
- Original run: `/workspace/runs/medium-ood`.
- Corrected validation report: `evaluation-valid-recovered/metrics.json`.
- Test report: `evaluation-test/metrics.json`.
- Durable recovery job: `/workspace/ageii-jobs/rf-evaluation-recovery`; completed with return code 0 at **2026-10-09 20:56:46 UTC**, total 110.16 seconds.
- Local full metrics: `ood-v1-validation-metrics.json` and `ood-v1-test-metrics.json` beside this document.
- Completed model artifacts are published separately on [Hugging Face](https://huggingface.co/Atlearia/Gemma_4_finetuned). Keep the detector checkpoint, ordered class names and training metadata together. The original deployment artifact is not included in this source package.

The evaluation regression test exercises the actual NumPy category-ID type and preserves missing-class metrics as JSON null. All 46 RF CPU tests passed after the fix. The only warning was an upstream Starlette/httpx deprecation.
