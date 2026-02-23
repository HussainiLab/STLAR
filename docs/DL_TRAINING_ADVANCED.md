# Advanced DL Training Guide

Back to quick workflow: [README DL Workflow](../README.md#complete-deep-learning-training-workflow)

This guide covers advanced deep learning training configuration, CWT usage, monitoring, and troubleshooting.

## End-to-End Pipeline

1. `prepare-dl` — build training manifests and segments
2. `train-dl` — train model
3. `export-dl` — export TorchScript/ONNX model
4. `dl-batch` — run model on new recordings

---

## Step 1: Prepare Training Data (`prepare-dl`)

Single-session:

```bash
python -m stlar prepare-dl --eoi-file detections.txt --egf-file recording.egf --output training_data --split-train-val --val-fraction 0.2
```

Batch mode:

```bash
python -m stlar prepare-dl --batch-dir study_data --region Hippocampus --split-train-val -v
```

Useful options:
- `--region`
- `--set-file`
- `--pos-file`, `--ppm` (behavior-aware prep)
- `--split-train-val`, `--val-fraction`, `--random-seed`

---

## Step 2: Train (`train-dl`)

Basic:

```bash
python -m stlar train-dl --train training_data/manifest_train.csv --val training_data/manifest_val.csv --epochs 15 --batch-size 64 --lr 1e-3 --weight-decay 1e-4 --out-dir models
```

With CWT (2D models):

```bash
python -m stlar train-dl --train training_data/manifest_train.csv --val training_data/manifest_val.csv --model-type 6 --use-cwt --fs 4800 --epochs 15 --out-dir models
```

Batch training:

```bash
python -m stlar train-dl --batch-dir study_data --epochs 15 --batch-size 64 -v
```

Monitoring options:
- `--gui` for live diagnostics
- `--no-plot` for headless runs
- `--debug-cwt <dir>` to save scalograms for validation

---

## Step 3: Export (`export-dl`)

Single-session:

```bash
python -m stlar export-dl --ckpt models/best.pt --onnx models/model.onnx --ts models/model.pt --example-len 2000
```

Batch export:

```bash
python -m stlar export-dl --batch-dir study_data -v
```

Notes:
- Ensure `--model-type` / `--use-cwt` match training settings.
- Install ONNX dependencies if needed: `pip install onnx onnxruntime`.

---

## Step 4: Detection (`dl-batch`)

**Standard 1D models:**
```bash
python -m stlar dl-batch -f recordings/ --model-path models/model.pt --threshold 0.5 --batch-size 64 -v
```

**CWT 2D models (CRITICAL):**
If your model was trained with `--use-cwt`, you **must** enable CWT preprocessing during inference:
```bash
python -m stlar dl-batch -f recordings/ --model-path models/cwt_model.pt --use-cwt --fs 4800 --threshold 0.5 --batch-size 64 -v
```

**Important:** `--use-cwt` and `--fs` must match your training configuration, otherwise inference will fail with shape mismatch errors.

Additional options:
- `--dump-probs`: Print probability statistics for debugging
- `--debug-cwt <dir>`: Save CWT scalogram images for visual inspection
- `--window-size <secs>`: Window size in seconds (default: 0.1)
- `--overlap <fraction>`: Window overlap fraction (default: 0.5)
- `--gap-threshold <secs>`: Gap for merging nearby detections (default: 0.05 = 50ms for ripples; increase to 0.5 for longer epileptic events)

---

## Tuning Recommendations

- Start defaults: `--lr 1e-3`, `--batch-size 64`, `--weight-decay 1e-4`
- If validation plateaus: reduce learning rate (2–5x)
- If overfitting appears: increase weight decay
- If unstable: reduce batch size

Related docs:
- [Training Visualization](TRAINING_VISUALIZATION.md)
- [CWT Debug Guide](CWT_DEBUG_GUIDE.md)

---

## Troubleshooting

- CUDA OOM → reduce `--batch-size`
- No convergence → lower `--lr`, verify labels
- Export failure → verify checkpoint path and model type
- Missing plots → install `matplotlib`

---

## Cross-Links

- Quick commands: [README](../README.md)
- Advanced CLI operations (metrics/filtering/spatial): [Advanced CLI Guide](CLI_ADVANCED.md)
