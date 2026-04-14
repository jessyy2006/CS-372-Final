# Plan: Rubik's Cube Vision Analyzer — CS 372 Final Project

## Context
Build a post-hoc video analysis system for 3×3 Rubik's cube solves. Given a locked-camera video of someone solving a cube, the system outputs:
1. **Move log** — sequence of primitive moves (R, R', R2, U, U', U2, …, 18 classes)
2. **Efficiency critique** — points in the solve where the user took N extra moves vs. optimal continuation
3. **Optimal solution** — the shortest solve for the detected scramble

Motivation: speedcubers rely on expensive coaches or Bluetooth smart cubes. A purely vision-based coach running on any phone camera democratizes improvement.

Deadline: **Sunday, April 26, 2026 (13 days from 2026-04-13)**. Team of 2, 10–15 hr/wk each, Duke GPU cluster access.

---

## System architecture (5 stages)

```
video ─▶ [1] cube detection ─▶ [2] move segmentation ─▶ [3] move classification ─▶ [4] state reconstruction ─▶ [5] analysis/UI
        YOLOv8 bbox+crop     motion energy (optical    VideoMAE fine-tuned       apply moves to initial     kociemba comparison
                             flow / frame diff)        (18 moves + no-op)        scramble; track state      + Streamlit web app
```

### Stage 1 — Cube detection (YOLOv8)
- Fine-tune YOLOv8-small from a Roboflow pretrained Rubik's cube checkpoint on ~100 of our own frames.
- Output: per-frame bounding box → cropped 224×224 cube region.
- Rubric hits: object detection fine-tuning (7), transfer learning (5).

### Stage 2 — Move segmentation
- Frame-difference / Farnebäck optical flow → motion energy signal.
- Threshold + peak detection identifies move boundaries (~0.5–2s per move).
- Output: list of `(start_frame, end_frame)` move clips.
- Rule-based — fast, interpretable, no training needed.

### Stage 3 — Move classification (VideoMAE)
- Fine-tune `MCG-NJU/videomae-base` (~86M params) on 16-frame clips → 19 classes (18 moves + no-op).
- ~8–12 GB GPU memory; fits on Duke cluster A100/A6000.
- Rubric hits: pretrained ViT (5), fine-tuning (5), ViT for video (7–10).

### Stage 4 — State reconstruction
- Detect initial scramble state from first static frame (sticker-color classifier per face).
- Apply predicted move sequence using `pycuber` → timeline `[(t, state, move)]`.

### Stage 5 — Analysis + UI
- At each timeline state, `kociemba.solve(state)` → compare optimal remaining vs. actual remaining.
- Flag suboptimal segments ("you took 9 moves here, optimal was 4").
- **Streamlit web app**: upload → processing bar → annotated move log, efficiency timeline, playback.
- Rubric hits: web app + UI deployment (10).

---

## Data creation — detailed pipeline

No public dataset covers all 18 move primitives with video-level labels. We build our own via **auto-labeled scripted scrambles**, supplemented by public data for detection.

### Step 1 — Scramble generation
**File:** `src/data/scramble_generator.py`

```python
# Pseudocode
MOVES = ['R','R\'','R2','L','L\'','L2','U','U\'','U2',
         'D','D\'','D2','F','F\'','F2','B','B\'','B2']  # 18 classes

def generate_scramble(n_moves=25, seed=None):
    """WCA-style: no consecutive moves on the same face; no three moves on same axis."""
    rng = random.Random(seed)
    seq, last_face, last_axis = [], None, None
    while len(seq) < n_moves:
        move = rng.choice(MOVES)
        face, axis = move[0], FACE_TO_AXIS[move[0]]
        if face == last_face: continue
        if axis == last_axis and len(seq) >= 1 and FACE_TO_AXIS[seq[-1][0]] == axis:
            continue  # prevent 3-in-a-row on same axis
        seq.append(move); last_face, last_axis = face, axis
    return seq
```

**Output:** 200 scrambles → `data/scrambles/scramble_XXXX.json` with `{id, moves, seed, created_at}`.

Class balance: across 200 × 25 = 5,000 moves, each of 18 classes gets ~275 samples on average — enough per class.

### Step 2 — Recording protocol
**Why rigid:** auto-labeling only works if recording conditions are consistent enough for motion segmentation to reliably detect move boundaries.

**View:** high-angle bird's-eye, **camera tilted ~15° off vertical toward the solver**. U face dominates the frame; a sliver of the front face is visible. Deliberate compromise — keeps the consistent overhead framing that makes motion segmentation trivial while giving the classifier enough side-face signal to disambiguate F/B and L/R.

Setup checklist per session:
- Phone mounted above the cube, tilted 15° off vertical toward the solver. Height ~30–35 cm above the cube. Measure the tilt once with a phone level and mark the rig position with tape for reproducibility.
- Ambient room lighting, **fixed for the entire session** (no light switches, no blinds mid-recording). Position self so hand shadows fall off to the side of the cube, not onto it.
- Same physical 3×3 cube (matte stickers preferred; we'll note brand in ATTRIBUTION.md).
- Camera: iPhone at 60fps, 1080p, locked exposure + focus (tap-hold to lock — critical, since hands entering the frame otherwise trigger re-exposure).
- Plain dark mat under cube; tape cross to mark the cube's resting position.
- Solver's hands start out of frame, cube in neutral orientation (white up, green toward solver) on the cross.

Per scramble:
1. **Initial state capture (~6 s)** — solver holds the cube ~15–20 cm below the lens and presents each face flat **to the lens** (not to the mat) for ~1 s each in order U, D, F, B, L, R. Full stillness per face — motion blur kills color classification. Then sets cube down on the cross.
2. **Clap frame** — solver snaps fingers once over the cube → bright motion spike = sync anchor between state-capture and scramble execution.
3. Execute the scramble moves at ~1 move/sec with a **deliberate 500 ms pause between every move**. No rotations (x/y/z) allowed. Keep cube on the cross; minimize translation between moves. Pinch-grip for side turns (no palm-cupping). Consistent hand-approach direction per face (R from right, L from left, F from near, B from far, U/D from above). Both hands always visible.
4. Hold still 2 seconds at end → natural end marker.

**File:** `src/data/recording_utils.py`
- `SessionManager` class writes each recording to `data/raw/{recorder_id}/{scramble_id}/video.mp4` alongside `metadata.json` (known moves, recorder, timestamp, cube orientation).

**Time budget:** 25 moves × 200 scrambles × ~1.2s/move + 4s overhead per scramble = ~120 min of video, ~4–6 hr total with resets across team.

### Step 3 — Frame extraction + cube cropping
**File:** `src/data/frame_pipeline.py`

1. Decode video with **decord** (faster than OpenCV for random access) or `torchvision.io.read_video`.
2. Downsample to 30fps (from 60) → `frames.npy` per video, shape `(T, H, W, 3)` uint8.
3. Run YOLOv8 detector on every 10th frame; linearly interpolate bbox between keyframes → smooth per-frame `bbox[t]`.
4. Crop each frame to `bbox[t]` with 20% margin, resize to 224×224 → `frames_cropped.npy`.

### Step 4 — Motion segmentation → auto-labels
**File:** `src/processing/motion_segmentation.py`

```python
# Pseudocode
def motion_energy(frames_cropped):
    """Dense optical flow magnitude per frame, summed over cube region."""
    energy = []
    for t in range(1, len(frames_cropped)):
        flow = cv2.calcOpticalFlowFarneback(
            prev=to_gray(frames_cropped[t-1]),
            next=to_gray(frames_cropped[t]),
            flow=None, pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        mag = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
        energy.append(mag.sum())
    return np.array(energy)

def segment_moves(energy, fps=30, known_move_count=25):
    """Smooth, threshold, find peaks; return (start,end) for each move."""
    smooth = uniform_filter1d(energy, size=int(0.2*fps))   # 200ms box
    thresh = np.percentile(smooth, 30)                      # rest level
    active = smooth > thresh
    # Find contiguous active runs, merge runs closer than 100ms
    segments = runs_of_true(active, min_gap=int(0.1*fps))
    # Sanity check: segment count must equal known_move_count (±1 for clap)
    return segments
```

**Alignment:** ordered segments → assigned sequentially to the known scramble move list.

**Validation gate:** after first 5 scrambles recorded, manually verify alignment. If segment count ≠ move count on >10% of scrambles, fallback is to re-record with explicit 500ms pause between moves (trades realism for label cleanliness — acceptable since we also evaluate on real un-paced solves held out at the end).

### Step 5 — Clip extraction for transformer
**File:** `src/data/clip_extractor.py`

For each `(start_frame, end_frame, move_label)` from segmentation:
- **Target clip length = 16 frames** (VideoMAE standard).
- If `end - start >= 16`: sample 16 frames uniformly across the segment.
- If `end - start < 16`: symmetric padding with adjacent rest frames (important — gives the model "setup" + "execution" + "settle" context).
- Save each clip as a single `.mp4` (H.264, 224×224) + row in `manifest.csv`:

```csv
clip_id,video_path,start_frame,end_frame,move_label,scramble_id,recorder,split
0001_00,data/clips/0001_00.mp4,30,52,R,0001,alice,train
0001_01,data/clips/0001_01.mp4,52,78,U',0001,alice,train
...
```

**Expected yield:** ~200 scrambles × 25 moves ≈ **5,000 labeled clips**. Plus a "no-op" class sampled from rest periods between moves (~500 clips) → 19 classes total.

### Step 6 — Splits
- 70/15/15 train/val/test, split **at scramble level** (not clip level) to prevent leakage.
- `split_manifest.py` writes `manifest_train.csv`, `manifest_val.csv`, `manifest_test.csv`.
- **Held-out end-to-end test set** (separate): ~20 *real un-paced solves* (not scripted scrambles) recorded at the very end, with hand-labeled move sequences. This is the honest benchmark for the final system.

### Step 7 — Public data integration
- **Roboflow cube-detection images** (~500 total across 2–3 public datasets): merge into YOLOv8 training set. Provides viewpoint/lighting diversity.
- **felikemath R/U/F clips**: 200 frames/40 sequences. Extract same 16-frame clip format; mix into train set for R/U/F classes only (clearly noted in ATTRIBUTION.md).
- **LAION strategic_game_cube**: symbolic-only, used to unit-test solver/state code. Not vision training.

---

## Preparing data for the transformer (VideoMAE)

VideoMAE expects specific tensor shapes, normalization, and augmentation patterns. Here's exactly how clips become model inputs.

### Input specification
- **Shape:** `(batch, num_frames=16, channels=3, H=224, W=224)` float32.
- **Normalization:** ImageNet stats — `mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]` per channel.
- **Value range:** normalized floats (not 0–255 ints).
- **Frame order:** temporal, earliest frame first.

VideoMAE internally tokenizes into **3D tubelets of size 2×16×16** (2 frames × 16×16 pixels per patch), producing `(16/2) × (224/16) × (224/16) = 8 × 14 × 14 = 1,568` tokens per clip. We don't implement this — the HuggingFace model does — but knowing the shape helps debug OOM and tune batch size.

### Dataset class
**File:** `src/models/videomae_dataset.py`

```python
# Pseudocode — actual implementation uses HF VideoMAEImageProcessor
from transformers import VideoMAEImageProcessor
import decord, torch

class MoveClipDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_csv, split, augment=False):
        self.rows = pd.read_csv(manifest_csv).query(f"split == '{split}'")
        self.processor = VideoMAEImageProcessor.from_pretrained(
            "MCG-NJU/videomae-base")
        self.augment = augment
        self.label_to_id = {m: i for i, m in enumerate(ALL_CLASSES)}  # 19 classes

    def __getitem__(self, idx):
        row = self.rows.iloc[idx]
        vr = decord.VideoReader(row.video_path)
        # Step A: temporal jitter (augmentation)
        n = len(vr)
        if self.augment:
            offset = np.random.randint(-2, 3)  # ±2 frames
            start = max(0, offset)
        else:
            start = 0
        indices = np.linspace(start, n-1, 16).astype(int)
        frames = vr.get_batch(indices).asnumpy()  # (16, H, W, 3) uint8

        # Step B: spatial augmentation
        if self.augment:
            frames = self._spatial_augment(frames)  # rotation, color jitter, flip

        # Step C: HF processor handles resize(224) + normalize + to tensor
        pixel_values = self.processor(list(frames), return_tensors="pt").pixel_values
        # pixel_values shape: (1, 16, 3, 224, 224) → squeeze batch
        label = self.label_to_id[row.move_label]
        return {"pixel_values": pixel_values.squeeze(0),
                "labels": torch.tensor(label)}
```

### Augmentation details (train split only)
Each matters for generalization given only ~3,500 train clips:

| Augmentation | Detail | Why |
|---|---|---|
| **Temporal jitter** | Shift sampled frame indices by ±2 frames | Robustness to imperfect segmentation boundaries |
| **Random rotation** | ±10° around image center, applied consistently to all 16 frames | Tolerate slight tripod / cube-orientation drift |
| **Color jitter** | brightness ±0.2, contrast ±0.2, saturation ±0.1 | Lighting variation at inference time |
| **Horizontal flip** | With 50% prob, flip all frames **AND swap move label** (R↔L, F↔B, etc.; U/D unchanged; prime/2 preserved) | Doubles effective data for L-R pairs |
| **No vertical flip** | — | Would produce physically impossible cube states |
| **No MixUp/CutMix** | — | Mixing two move clips destroys the label |

The horizontal-flip label swap is important and subtle — implement as a lookup table `FLIP_MAP = {'R':'L', 'L':'R', 'R\'':'L\'', ...}`.

Rubric hits: data augmentation with impact evidence (5pt) — we'll run ablation with/without each augmentation.

### DataLoader + batching
```python
train_loader = DataLoader(
    MoveClipDataset("manifest.csv", "train", augment=True),
    batch_size=8,          # fits in ~10 GB with fp16
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
)
```

### Model head + fine-tuning setup
```python
from transformers import VideoMAEForVideoClassification
model = VideoMAEForVideoClassification.from_pretrained(
    "MCG-NJU/videomae-base",
    num_labels=19,
    ignore_mismatched_sizes=True,  # replaces pretrained head
    id2label=ID_TO_LABEL, label2id=LABEL_TO_ID,
)
# Freeze backbone for first 2 epochs, then unfreeze all
for p in model.videomae.parameters(): p.requires_grad = False
```

### Training recipe (starting point, to tune)
- **Optimizer:** AdamW, lr=5e-5 (head), 5e-6 (backbone when unfrozen).
- **Scheduler:** cosine with 10% warmup → rubric hit: LR scheduling (3pt).
- **Loss:** cross-entropy; optional class-balanced weighting if any class <150 samples.
- **Epochs:** 15 (2 frozen + 13 unfrozen). Early stop on val F1.
- **Mixed precision:** `fp16` via HF Trainer `--fp16`.
- **Hyperparameter sweep:** 3 configs — {lr ∈ 1e-5, 5e-5, 1e-4} × fixed everything else → rubric hit: HP tuning (3pt).
- **Metrics:** per-class precision/recall/F1, top-1 accuracy, confusion matrix. Track with Weights & Biases or just CSV + matplotlib.

### Evaluation on held-out test
1. Per-clip accuracy on `manifest_test.csv` (scripted scrambles, held-out).
2. End-to-end move-log accuracy on real un-paced solves (hand-labeled).
3. Solve reconstruction rate: fraction of videos where `apply_moves(initial_state, predicted_moves) == solved_state`.

---

## 13-day schedule

| Day | Milestone | Owner |
|-----|-----------|-------|
| 1 (Mon 4/14) | Repo structure, scramble generator, recording setup validated on 5 test scrambles | Both |
| 2 (Tue 4/15) | Record all 200 scrambles; run segmentation + alignment validation | Partner A |
| 2–3 | YOLOv8 fine-tune on Roboflow + ~100 of our frames | Partner B |
| 3–4 (Thu 4/17) | Dataset v1 frozen (manifest.csv committed); clip extraction done | Both |
| 5–7 (Sun 4/19) | VideoMAE fine-tune; first end-to-end pipeline run | B lead, A support |
| 6–7 | Sticker-color scramble-state detector | Partner A |
| 8–9 (Tue 4/21) | kociemba integration + suboptimality analysis | Partner A |
| 8–10 | Streamlit web app UI | Partner B |
| 11 (Fri 4/24) | Ablations, error analysis, training curves, held-out eval | Both |
| 12 (Sat 4/25) | Demo + technical videos; README/SETUP/ATTRIBUTION | Both |
| 13 (Sun 4/26) | Buffer / polish / self-assessment | Both |

---

## Rubric item shortlist (~15 items, 90+ pts)

1. Modular code design (3)
2. Train/val/test split at scramble level (3)
3. Training curves tracked (3)
4. Proper batching (3)
5. Data augmentation with impact evidence (5)
6. HP tuning ≥3 configs (3)
7. **Original dataset collection (10)** ⭐
8. Pretrained vision model (VideoMAE) (5)
9. Fine-tuning / transfer learning (5)
10. Vision Transformer for video (7)
11. Object detection fine-tuning (YOLOv8) (7)
12. Learning rate scheduling (3)
13. ≥3 eval metrics (3)
14. Error analysis with visualization (5)
15. **Web app + UI deployment (10)** ⭐

**ML category subtotal:** ~79pt (73 cap) + Category 2 + 3 → target 103+.

---

## Partner split

- **Partner A (data + analysis):** data collection, segmentation, sticker classifier, kociemba analysis, eval scripts.
- **Partner B (models + UI):** YOLOv8, VideoMAE, Streamlit, end-to-end pipeline.
- Both: recording, docs, demo videos.

---

## Key files

```
src/
├── data/
│   ├── scramble_generator.py
│   ├── recording_utils.py
│   ├── frame_pipeline.py
│   └── clip_extractor.py
├── processing/
│   ├── motion_segmentation.py
│   └── sticker_classifier.py
├── models/
│   ├── yolo_cube_detector.py
│   ├── videomae_dataset.py
│   └── videomae_classifier.py
├── analysis/
│   └── solver.py
└── pipeline.py
app/
└── streamlit_app.py
notebooks/
├── 01_data_exploration.ipynb
├── 02_training_curves.ipynb
└── 03_error_analysis.ipynb
```

---

## Verification (end-to-end)

1. `python -m src.pipeline --video held_out_solve.mp4 --out results.json`
2. Check predicted move log produces valid cube path (final state = solved).
3. Review suboptimality highlights on 5 hand-labeled solves.
4. `streamlit run app/streamlit_app.py` → upload video → annotated output in <1 min.
5. Benchmarks:
   - Per-move classification accuracy ≥90% on scripted held-out.
   - End-to-end solve reconstruction ≥80% on real un-paced held-out.
   - Pipeline latency <45s for 1-min video on laptop GPU.
