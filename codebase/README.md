# UAV Pedestrian Attribute Pipeline

Implements the methodology doc section-by-section: video audit → frame
sampling → RT-DETRv2 detection → ByteTrack tracking → diverse crop
selection → Swin+DFSM+S-ACRM attribute recognition with multi-observation
relation fusion → final per-person CSV/JSON.

## Part 1 — Gender / Attribute Recognition

## Where to run this

This needs a GPU (or a patient CPU) with `torch`, `timm`, `ultralytics`,
`supervision` installed — run it in Colab, your own machine, or any box
with a real Python/CUDA setup. It won't run inside a plain chat sandbox.

```bash
pip install -r requirements.txt
```

## 1. One command for stages 1–5 (no labels needed yet)

```bash
python run_pipeline.py --video /path/to/your_5-6min_drone_video.mp4 --video_id run01
```

This produces, under `work/`:
- `audit_report.json` — footage stats + sampling recommendation (§3.1)
- `frames/` + `frames_manifest.json` — sampled frames, split by video
  segment into train/test so no leakage (§3.2/3.3)
- `detections/detections.json` + `detections/cvat_preannotations.xml` —
  RT-DETRv2 D0 baseline boxes, exportable straight into CVAT for review (§4)
- `tracks/tracks.json` — ByteTrack trajectories (§5)
- `crops/person_<id>/` + `crops/crop_index.json` — 5–10 diverse,
  farthest-point-sampled crops per identity (§5 close / §6.1)

## 2. Label the crops

Section 6.1: each identity gets 5–10 representative crops, and you label
the **identity once** (all its crops share the label). Open
`work/crops/person_<id>/` for each id and fill in `labels.json`:

```json
{
  "1": {"gender": 0, "hat": 1, "backpack": 0, "upper_colour": 3, "upper_style": 1, "lower_colour": 2, "lower_style": 0},
  "2": {"gender": -1, "hat": 0, ...}
}
```

Use `-1` for anything genuinely unjudgeable — "unknown", not assumed
negative, per the doc. `gender` is 0/1 (adjust the class map in
`config.py: ATTRIBUTE.attribute_num_classes` to your own label scheme,
e.g. if you're using UAV-Human's original category indices).

## 3. Train the attribute model (Swin + DFSM + S-ACRM, §6.2)

```bash
python stage6_attribute_model.py train --crop_index work/crops/crop_index.json --labels labels.json
```

## 4. Fused inference + final output

```bash
python stage6_attribute_model.py infer --ckpt checkpoints/attribute_model.pt \
    --crop_index work/crops/crop_index.json

python stage7_output_and_ablation.py build --video_id run01 \
    --tracks work/tracks/tracks.json --attr_preds work/attribute_predictions.json
```

Produces `work/output/per_person_attributes.csv` with the **exact**
field set from §8.1:

```
video_id | frame_id | person_id | x1 | y1 | x2 | y2 | detector_confidence
| backpack | hat | upper_colour | upper_style | lower_colour | lower_style
| attribute_confidence
```

plus `per_person_summary.json`, which carries `gender` at the identity
level (it's not repeated per detection row, since it's a per-person
result, not a per-box one).

## 5. The core ablation (§7): does fusing K observations actually help?

```bash
python stage7_output_and_ablation.py ablate --crop_index work/crops/crop_index.json \
    --labels labels.json --ckpt checkpoints/attribute_model.pt
```

Reports per-attribute accuracy at K = 1, 2, 3, 4, 5 — the single
comparison the doc calls out as central: K=1 (best single frame) vs
K>1 (fused). This is exactly where your gender-prediction question
gets answered empirically, not assumed.

## Detector fine-tuning ladder (§4.3), if D0 isn't good enough on your footage

```bash
# after reviewing detections.json in CVAT and exporting YOLO-format labels:
python stage3_detection.py finetune --data uav_dataset.yaml --weights rtdetr-l.pt --name D1
python stage3_detection.py eval --weights runs/detect/D1/weights/best.pt --data uav_dataset.yaml
```

## Part 2 — Crossing-Intention Prediction (Zhou et al. framework)

Built on top of everything above (reuses stage3's detector and stage4's
tracks), this adds pedestrian-centric environment graphs (Section III-C),
a GCN-based environment encoder (III-D), a pedestrian-state encoder
(III-E), and an LSTM intention decoder (III-F) — implementing the paper's
equations directly (Eq. 1-18).

**Sampling-rate note:** stage2's frames are sparse (1/1-2s) for attribute
annotation *coverage*. The intention module needs dense, closely-spaced
frames within short 1-second windows instead — it re-extracts those
directly from the raw video per sample (stage9's `extract_dense_window`),
it does not reuse stage2's frames.

### New setup steps

**1. Zebra crossing annotation** (crosswalk isn't a COCO class):
```bash
python stage8_zebra_crossing.py --frame work/frames/frame_0000000_t00000.00.jpg --out_dir work/zebra
```
Review `work/zebra/candidates_visualization.png`, hand-edit
`work/zebra/zebra_crossings.json` if the heuristic missed/misdrew a
crossing. One-time annotation for the whole video (assumes a mostly-fixed
drone hover — re-run per distinct camera position if it moves a lot).

**2. Label crossing events** — this is the human-in-the-loop step Section
IV-A itself requires (there's no way around watching footage and noting
when people cross). Create `crossing_labels.json`:
```json
{
  "17": {"crosses": true, "cross_start_time_sec": 184.5},
  "23": {"crosses": false}
}
```
Track IDs come from `work/tracks/tracks.json` (stage4's output — the SAME
track IDs the gender/attribute pipeline uses, so one round of tracking
serves both modules).

**3. Train + infer**:
```bash
python stage14_train_intention.py --video drone.mp4 \
    --tracks work/tracks/tracks.json --zebra work/zebra/zebra_crossings.json \
    --crossing_labels crossing_labels.json --out_dir work/intention
```
This runs both training phases (unsupervised pretraining of the GCN
autoencoder + trajectory LSTM-AE, then supervised end-to-end training of
the rest) and writes `work/intention/intention_predictions.json`.

**4. Combine with the gender/attribute output**:
```bash
python stage13_combined_output.py --video_id run01 \
    --intention_preds work/intention/intention_predictions.json \
    --attr_summary work/output/per_person_summary.json
```

### Honest engineering substitutions (read before you evaluate results)

- **Pose estimation**: the paper uses HRNet; this uses Ultralytics
  YOLO-pose (already a dependency here, easier to run on Colab). Neither
  will produce reliable keypoints on the highest-altitude passes of your
  footage — pedestrians are just too few pixels. The skeleton branch is
  confidence-gated (Section III-E.3's own occlusion-handling rule: low-
  confidence joints get zeroed rather than fed in as noise), so expect it
  to contribute little on your straight-down establishing shots and more
  on any closer/oblique passes.
- **Skeleton extractor pretraining**: the paper pretrains this branch on
  Kinetics-400 before fine-tuning. Not replicated here — it trains from
  random init as part of the supervised phase, which will need more
  labelled samples to pull its weight than the paper's setup assumed.
- **Vehicle/pedestrian motion tracking within the environment graph**:
  the paper doesn't fully specify how surrounding entities are matched
  frame-to-frame; this uses simple greedy nearest-centroid matching
  within each K1-second window (not full ByteTrack) — adequate for
  velocity/acceleration over 3 consecutive dense frames, not meant for
  long-term identity.
- **No `torch_geometric`**: the GCN and ST-GCNN layers are hand-implemented
  matrix ops (Eq. 7 directly), not a graph library. Functionally
  equivalent, easier to install on Colab/Kaggle without CUDA-version
  headaches.

### Realistic timing

The paper itself reports **5 days 16 hours** to converge on a single
RTX 3090 (Section IV-B) for a framework of comparable scope. On a Colab
T4/L4 with a few hundred labelled samples (realistic yield from one
30-minute video), expect the supervised phase to run for **hours**, not
minutes — each training sample needs its own multi-class detector pass +
pose estimation across ~8 dense frames, which dominates cost, not the
decoder itself. Colab's free-tier session limits (~12hrs) will likely
require checkpointing (`torch.save`) partway through and resuming; the
script already saves the decoder checkpoint at the end, but for a long
run you may want to add periodic mid-training checkpoints if you're on
the free tier specifically.

### Output schema

`stage13`'s combined CSV is keyed by **(person, time window)**, not just
person, since crossing intention is time-varying:

```
video_id | person_id | window_start_sec | window_end_sec | K2_sec
| crossing_intention_prob | crossing_predicted | gender | hat | backpack
| upper_colour | upper_style | lower_colour | lower_style | attribute_confidence
```

## Notes on Part 1 — what's a placeholder vs. what's real

- **Detection (RT-DETRv2)**: real, uses pretrained COCO weights out of
  the box via `ultralytics`. Works immediately on your footage.
- **Tracking (ByteTrack)**: real, via `supervision`.
- **Attribute model architecture**: real, matches §6.2's description
  (Swin backbone, PRS+RRS suppression, S-ACRM relation vectors,
  max-confidence fusion across K). What it **can't** do out of the box
  is predict correctly with zero training — attribute heads are randomly
  initialized until you train them on your labelled crops (step 2–3
  above). There is no publicly available pretrained checkpoint for this
  exact architecture + your exact attribute set, so some labeling is
  unavoidable — the doc's own §6.1 says as much ("each identity initially
  receives five to ten representative annotated crops").
