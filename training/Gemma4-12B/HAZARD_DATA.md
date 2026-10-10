# Task-contract hazard training data

The completed prototype used **PAVE images + native accessibility descriptions + actual frozen RF detections as privileged teacher inputs**. It did not train the source-QA task alone. The teacher produces the same structured hazard response that the local worker expects.

## Sources and preparation

1. Use the licensed, pinned PAVE/SANPO source pipeline in [DATA.md](DATA.md), with **300 training frames and 60 validation frames**. Recording sessions remain disjoint and the original PAVE evaluation split is untouched. PAVE training rows drawn from SANPO's native test list are excluded.
2. `attach_rf_boxes.py` runs the finished RF-DETR Medium checkpoint on these images, at confidence threshold **0.3**, with up to 100 boxes. Each box carries the actual detection confidence, normalized coordinates, a label and a frame-local ID. These detections are predictions, not new human labels.
3. The teacher sees the image, those detections and the source assessment text. The source text provides outdoor context, but its own provenance is existing GPT-5-nano-generated PAVE QA. Teacher outputs must pass the shared response and object-ID validators. Invalid generations are rejected rather than silently converted into safe scenes.
4. The student trains on the **image + RF context → hazard JSON** pairs. The privileged PAVE answer is excluded from student inputs. This is weak supervised task adaptation, not a claim that human annotators verified the resulting hazards.

Required generated-label provenance:

```json
{
  "hazard_labels_verified": false,
  "supervision": "weak",
  "supervision_kind": "weak_teacher",
  "label_policy_id": "pave-rf-privileged-teacher-v2",
  "label_policy_version": 1
}
```

Keep original source/license/image hashes and detector checkpoint SHA-256 alongside those fields. A successful schema check proves format and reference validity, not factual correctness.

## Executable preparation

Run from the repository root with the paths and environments set in [README.md](README.md). `STRIDE_RF_CHECKPOINT` must identify a trained RF-DETR Medium checkpoint; substitute the actual RF environment path if it differs.

```bash
python training/Gemma4-12B/prepare_data.py \
  --cache "$STRIDE_DATA/gemma-pave-cache" \
  --output "$STRIDE_DATA/gemma-pave-raw-source" \
  --download-media --max-train-frames 300 --max-validation-frames 60 \
  --max-test-frames 0 --max-download-bytes 2500000000 --workers 6

training/.venv-rfdetr/bin/python training/Gemma4-12B/attach_rf_boxes.py \
  --source "$STRIDE_DATA/gemma-pave-raw-source" \
  --output "$STRIDE_DATA/gemma-pave-teacher-source" \
  --checkpoint "$STRIDE_RF_CHECKPOINT" \
  --threshold 0.3 --batch-size 8
```

Both commands require new output directories and leave source images/checkpoints unchanged. The downloader keeps resized JPEGs, not the original PNGs. `data_report.json` must say `prepared` before teacher generation starts. `train.jsonl` and `validation.jsonl` in `gemma-pave-teacher-source` are still **teacher input**, not finished hazard training labels; their `target` fields contain the native assessment paragraph until teacher generation creates a separate dataset.

## Completed source preparation

On 2026-10-09, the H100 preparation produced the following actual teacher inputs. See [the saved preparation report](reports/pave-teacher-input-report.json) for hashes and provenance.

| Split | Images | Recording sessions | RF boxes | Images without boxes |
| --- | ---: | ---: | ---: | ---: |
| Train | 300 | 62 | 1,808 | 10 |
| Validation | 60 | 7 | 338 | 2 |

The recording sessions do not overlap. Downloaded source image bytes totalled 1,212,521,937; local JPEGs are smaller. The detector used checkpoint SHA-256 `da22cd0f3c3389716a2a2f402c3bb54dc16eb84ecc821c445b9bd8518359cabf`. RF preprocessing finished at 21:20:03 UTC. These counts establish source availability, not final accepted hazard-label counts or completed training.

## Teacher quality audit

The first teacher prompt produced valid schema and ID references but over-reported path-edge objects as hazards. In a nonrandom first-12 validation audit, all 12 examples emitted two hazards, only one hazard was uncertain, and no scene had an empty hazard list. Visual inspection of three images confirmed an unobstructed broad park path labelled hazardous because of an adjacent pole and hedge, and a path-edge bench labelled as an obstruction despite open walking space. These are reasons to revise the teacher, not verified hazard annotations.

The revised teacher must require visible evidence of a current forward-path obstruction, contact risk, drop or surface issue. Objects beside the route, distant people and hypothetical veering or crossing do not by themselves establish danger. Empty hazard lists are valid when the image supports them; the generator must not assign negatives to satisfy a quota. Record the actual prompt hash and policy ID per output and audit accepted targets before training. Format validity and teacher agreement alone do not show practical warning quality.

The `pave-rf-privileged-teacher-v2` smoke generated 20 training and 12 validation examples in 84.18 seconds, with zero schema rejections. Training contained 15 potential-hazard scenes, four scenes without confirmed hazards, and one unknown; validation contained nine, one, and two respectively. Rechecking the same images showed the empty park path no longer warned about its adjacent pole/hedge, the plaza warning identified the visible bin instead of the edge bench/birds, and the sidewalk warning dropped its planter tree in favor of the person ahead. This supports proceeding with weak prototype generation, not a factual accuracy claim. Some descriptions still omit a clear consequence, and still-image movement descriptions may be overinterpreted. See [the bounded v2 audit](reports/teacher-v2-smoke-audit.json).

A later targeted audit of training rows 31–40 found remaining contextual errors: bins beside a wide route and a person at a fence edge were called obstructions, and a sky-dominated image with parked car roofs was labelled `moving_traffic` without acknowledging that the walking surface was barely visible. These observations limit any quality claim and may warrant a separate curation stage. Four early full-run validation rejections were formatting prefixes; one was an empty-hazard example, with no observed route-clear wording rejection. See [the later audit](reports/teacher-v2-later-audit.json). No active generation output was edited by the audit.

The full v2 teacher pass completed with **296 training / 56 validation examples**, rejecting four malformed outputs from each source split. These are raw weak labels before critic filtering. The completed [generation report](reports/teacher-v2-generation-report.json) is included for provenance; raw manifests and dataset backups are excluded from this source package.

## Separate critic pass

`filter_hazard_labels.py` uses the same base model in a separate image-based accept/reject role. It judges the existing image, detections, source context and proposed response; it never writes replacement target labels. Retained records preserve their target JSON and receive explicit critic provenance. This can reduce inconsistent supervision but remains correlated AI filtering, not independent human verification.

A 32-example smoke subset includes the three known noisy positives, three previously improved examples, and three empty scenes in each original split. The initial critic's raw decisions rejected all three noisy positives and retained the visually checked benign train scenes. Its first execution failed because valid decisions were usually enclosed in one complete JSON fence. The parser fix permits that exact offline wrapper while still rejecting numeric prefixes, prose and multiple fences. Serving remains strict. Final training counts must come from the completed filtered dataset report, with retained hazard/empty/unknown coverage inspected before training. Do not use raw source or smoke counts as the final training size.

## Final prepared training set

The completed critic pass retained **155 training examples and 22 validation examples**, from **54 and six disjoint recording sessions**, respectively. It took 155.9 seconds and made exactly one decision for each of the 352 raw teacher examples. The historical dataset was named `gemma-hazards-filtered-v2`. Its media and manifests are excluded from this source package; regenerate them using the documented preparation workflow.

| Split | Potential-hazard scenes | No confirmed hazard reported | Unknown scenes | Total |
| --- | ---: | ---: | ---: | ---: |
| Train | 94 | 56 | 5 | 155 |
| Validation | 16 | 3 | 3 | 22 |

The critic policy is `pave-forward-area-critic-v1`, with prompt SHA-256 `a16ba6c5ed9aecd4ad95345f205c1925a4dca48bc1b871a2afa075e582259404`. It preserved every retained target and input unchanged, preserved original provenance, and kept `hazard_labels_verified: false`. The [final critic report](reports/critic-final-data-report.json) and [integrity audit](reports/critic-final-integrity-audit.json) record the actual counts and hashes.

Retained training targets contain 39 `surface_hazard`, 27 `trip_fall`, and 37 `obstacle_collision` hazard records. Only 29 of the 103 training hazard records have supplied RF IDs; many walking-surface conditions are unboxed. Validation has 21 hazard records, eight with IDs. This run does not supply retained supervised examples of moving traffic, fire, electrical or overhead hazards. The base model may produce other categories, but this adaptation does not demonstrate learning them. The dataset is a small prototype, not an adequate production validation set.

Targeted visual checks found visible stairs, a path puddle and cracked pavement among retained positives. All three previously identified noisy positive cases were rejected. For combined inference smoke testing, filtered validation row 18 (zero-based; `pave-3a32ac2a37e2f9ea74c8`) shows an RF-boxed bin ahead on a constrained sidewalk. Row 2 (`pave-b6df3f38d4bc324f3d92`) shows an open park path with adjacent poles and greenery. These examples support checking response wiring and box association; they are not an untouched accuracy benchmark.

## What this can and cannot establish

The resulting LoRA can learn task formatting, short warnings, the outdoor vocabulary, and association with supplied detector IDs. Self-distillation from the base Gemma model is not evidence of new underlying reasoning ability. A comparison against that base model is necessary, and agreement with generated validation labels is **teacher agreement**, not human hazard accuracy.

Single images do not supply measured walking speed, moving-object trajectories, injury severity, true depth or guaranteed passable routes. The application keeps these uncertainties explicit and computes any heuristic alert priority separately. Neither this preparation nor the teacher introduces human-verified severity scores.

Geometry-only labels were rejected: a box's image position does not establish a real walking hazard, and forcing every target to unknown would not teach useful contextual assessment. The active strategy uses source descriptions as privileged teacher context and preserves explicit weak-label provenance.
