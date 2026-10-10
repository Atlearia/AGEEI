# Outdoor obstacle awareness for blind and low-vision pedestrians

Updated 2026-10-09. Input is sampled live video from a pedestrian camera. RF-DETR Medium supplies object boxes; Gemma 4 supplies activity observations and separate hazard descriptions. Application code handles ID validation, freshness and heuristic alert priority.

Focus on sidewalks, pedestrian and park paths, outdoor stairs and curbs. Prioritize evaluation of stairs/curbs, poles, roadblocks, benches, bins, people, bicycles and vehicles. Object presence alone does not make an object hazardous. Sparse frames and a vehicle box do not establish that the vehicle is approaching.

OOD has no explicit classes for potholes, pavement cracks, uncovered manholes, low branches, ice, puddles or drop-offs. A tree box does not establish head-height branch detection. Gemma may describe an unboxed region, but that possibility is not measured coverage. Exact distance, crossing permission, safe alternative routes and calibrated injury severity are not established outputs.

| Source | Outdoor accessibility fit | Current use |
|---|---|---|
| [OOD v1](https://universe.roboflow.com/fpn/ood-pbnro/dataset/1) | Publisher describes obstacles affecting blind pedestrians; 22 classes with boxes | RF training. Audited 7,999 train / 1,000 validation / 1,000 test; CC BY 4.0. |
| [PAVE](https://huggingface.co/datasets/rafiibnsultan/PAVE) + [SANPO](https://github.com/google-research-datasets/sanpo_dataset) | Broader pedestrian accessibility, not exclusively blindness; real images with existing synthetic QA prose | Original pool: 300 train / 60 validation images from 62/7 disjoint sessions. Final teacher-plus-critic dataset: **155 train / 22 validation examples from 54/6 sessions**. Weak targets informed by source prose and real RF predictions; no human hazard, severity or speed ground truth. |
| [WAD / WalkVLM](https://walkvlm2024.github.io/) | Specifically walking assistance for blind people; clips, descriptions and reminders | Further research. Official archive about 120.4 GiB; dataset terms unverified. No bulk download or training. |
| [VIP-Mobility360](https://opendata.ljmu.ac.uk/id/eprint/226/) | Outdoor mobility for people with visual impairments | Possible later detector extension, pending archive/schema audit. Not used in the current run. |

Dataset intent is distinct from capture by blind participants or validation with blind users. The current training and tests do not establish the latter. Read [Gemma's source audit](Gemma4-12B/DATA.md) for precise provenance and excluded test sessions.

The final Gemma training mix is 94 potential-hazard, 56 no-confirmed-hazard and five unknown scenes; validation is 16/3/3. Retained hazard categories are surface hazards, trips/falls and obstacle collisions. Many surface hazards have no RF box; the application must not invent one. The critic retained original targets without rewriting and remained explicitly unverified (`pave-forward-area-critic-v1`, prompt SHA-256 `a16ba6c5ed9aecd4ad95345f205c1925a4dca48bc1b871a2afa075e582259404`). This small prototype set and AI-assisted consistency filtering do not establish practical warning accuracy.

Warnings should be brief and factual, for example "Pole ahead, slightly left," only when supported by the image. Earlier-frame-only or uncertain hazards have no current box score or spoken warning. No detection or no reported hazard must not be rendered as "safe."
