# Training data and what it can teach

Research recorded on 2026-10-09. Datasets and model weights are not bundled in this Git directory.

Stride Ahead uses trained RF-DETR Medium for object detection and fine-tuned Gemma 4 12B for contextual hazard descriptions. See the [training overview](../README.md) and [Gemma data provenance](../Gemma4-12B/HAZARD_DATA.md).

## RF-DETR Medium: use OOD v1

- Dataset: **Outdoor Obstacle Detection (OOD), FPN, version 1**.
- Source: https://universe.roboflow.com/fpn/ood-pbnro
- COCO export: https://universe.roboflow.com/fpn/ood-pbnro/dataset/1
- Publisher-listed license: **CC BY 4.0**. Keep attribution with the experiment and any redistributed data.
- Published size: **10,000 images, 29,779 annotated instances, 22 classes**.
- Published split: **8,000 train / 1,000 validation / 1,000 test**.
- Relevant labels include stairs, curb, pole, bench, tree, bicycle, person, vehicles, waste container and roadblocks. Load the exact names and IDs from the downloaded export; the webpage's prose and label list use slightly different names for one class.
- Export **COCO JSON**, not an unannotated image download. A Roboflow account/API key may be required. The local-file preparation route also works with a manually exported copy.

Use all available training images by default. At this scale, reducing the number of epochs is preferable to randomly discarding rare obstacle classes merely to save time. An optional class-covering subset is only for a smoke test or a constrained first experiment.

`prepare_data.py` checks images, dimensions, category consistency and COCO boxes; handles exact duplicate leakage; and writes the retained counts and per-class distribution to `data_report.json`. The advertised counts above are BEFORE these checks. Do not advertise them as the final training counts without reading that report. Pixel-identical duplicate detection does not prove independence between neighboring video frames or near-duplicates.

The existing test split is held out. Validation selects checkpoints. Test is used for a final report after selecting a checkpoint. Source route/session metadata is not established, so describe this as the publisher's image split, not an independently verified route-held-out evaluation.

### Is this enough data?

It is a reasonable starting dataset for adapting an already pretrained detector to the listed outdoor obstacle classes. It is not evidence of coverage of all indoor environments, overhead branches, open pits, water, glass or every dangerous configuration. Class-specific recall and false detections matter more than the total image count. Inspect the per-class report; a class with few training/validation instances cannot support a strong performance claim. This dataset does not label whether every detected object is hazardous for the current action.

## Gemma 4 12B: contextual hazard supervision

Gemma uses PAVE/SANPO images with RF predictions and weak teacher-generated hazard targets. Its separate [training folder](../Gemma4-12B/README.md) contains preparation, filtering, training and evaluation scripts. The original accessibility answers inform the teacher; the student does not receive those answers as input.

## Other online data inspected

| Source | Existing labels | Decision |
|---|---|---|
| [SANPO](https://github.com/google-research-datasets/sanpo_dataset) | Egocentric images/video, panoptic masks, depth, pose; CC BY 4.0 data | Good follow-up source. Full data is about 6 TB, needs selective download and mask/region conversion. No per-object injury-severity labels. Avoid making it the six-hour critical path. |
| [PAVE](https://huggingface.co/datasets/rafiibnsultan/PAVE) | Accessibility QA and spatial/segmentation references over SANPO; annotation license CC BY 4.0 | Used for Gemma's source-informed weak hazard labels. Images are NOT included; obtain and match SANPO images separately. Example answers mix general accessibility with physical hazards and need quality checks. Not a verified per-object severity source. |
| [GuideDog](https://huggingface.co/datasets/kjunh/GuideDog) | 19,978 generated training descriptions; 2,106 human-verified evaluation descriptions | Not a default dependency: academic-research-only gate and noncommercial/no-redistribution conditions. Keep the human-verified set for evaluation if an eligible project uses it. |
| [WAD / WalkVLM](https://walkvlm2024.github.io/) | Walking clips, descriptions, boxes and scene danger labels | No explicit dataset license verified. Scene danger labels are not per-object consequence severity. |
| [EMBHazard](https://huggingface.co/datasets/EMBGuard/EMBHazard) | Image + proposed action, binary risk, hazard explanation | No dataset license verified; model/code licenses do not establish dataset permissions. No severity labels. |
| [Warehouse Safety](https://huggingface.co/datasets/Jsohal174/warehouse-safety-hazard-dataset) | Synthetic overhead warehouse images with severity/descriptions; CC BY 4.0 metadata | Different camera/domain and no matching per-object boxes. Do not use as proof of blind walking hazard accuracy. |
| [Abtinzandi obstacle collection](https://huggingface.co/datasets/Abtinzandi/Obstacle-Detection-Dataset-YOLO) | 24,326 images, 25 object classes and YOLO boxes | Public download is about 1.54 GB. MIT metadata does not cover every integrated source: the card preserves upstream licenses. Avoid using the merged collection as an unqualified permissive source. |

## Severity in the application

No selected training data provides validated per-object injury severity. Application priority scores are explicitly heuristic, allow `unknown`, and differ from detection confidence. Do not create evaluation truth by having the same model label its own outputs. A "no hazard" prediction is not a guarantee that a route is clear.

## Model/training references

- RF-DETR source and exact release: https://github.com/roboflow/rf-detr/tree/1.11.2
- RF-DETR Medium weights/code license and size table: https://github.com/roboflow/rf-detr
- RF-DETR package: https://pypi.org/project/rfdetr/1.11.2/
- Gemma model: https://huggingface.co/google/gemma-4-12B-it

The scripts use RF-DETR's own trainer. Checkpoints are written to the supplied local/persistent output directory; no automatic cloud job, Hub upload or publication occurs.
