# HFO Deep Learning Classifier (Minimal Scaffold)

This folder provides a small, self-contained pipeline to train and export a 1D CNN classifier for HFO segments.

## What it expects
- A manifest CSV with columns: `segment_path,label`
  - `segment_path`: path to a `.npy` file containing a 1D waveform (float32), already trimmed to the EOI.
  - `label`: `1` for HFO, `0` for non-HFO.
- All segments in the manifest should share the same sampling rate; per-segment z-scoring is applied inside the loader.

## Quick start
1) Install deps (in `pyhfogui` env):
```powershell
conda run -n pyhfogui pip install torch onnxruntime
```

2) Detect EOIs in GUI:
- Run Hilbert, STE, or MNI detection to populate the EOI list.

3) Export EOIs for training:
- Click "Export EOIs for DL Training" button in the Score window.
- Select an output directory (e.g., `training_data`).
- Segments will be saved as `.npy` files and a `manifest.csv` will be created.

  Alternatively, if you have manually scored events in the Score tab (e.g., Ripple / Fast Ripple / Sharp Wave Ripple vs Artifact), click "Create labels for DL training" to export labeled segments directly (Ripple-family=1, Artifact=0).

4) Convert to Train/Val Splits:
- Switch to the "Convert" tab in the Score window.
- Click "Add Manifest(s)" and select one or more manifest.csv files (from different sessions/subjects).
- Set validation fraction (default 0.2 = 20%).
- Optionally enable stratified split to balance label distributions.
- Select output directory and click "Create Train/Val Split".
- This creates `train.csv`, `val.csv`, and metadata files with proper subject-wise splitting.

   Alternatively (manual split): Split `manifest.csv` into train/val using the command-line tool:
```powershell
conda run -n pyhfogui python -m hfoGUI.dl_training.manifest_splitter manifest1.csv manifest2.csv --val-frac 0.2 --output splits/
```

5) Train:
```powershell
conda run -n pyhfogui python -m hfoGUI.dl_training.train --train splits/train.csv --val splits/val.csv --epochs 15 --batch-size 64 --lr 1e-3 --out-dir models
```

6) Export TorchScript and ONNX:
```powershell
conda run -n pyhfogui python -m hfoGUI.dl_training.export --ckpt models/best.pt --onnx models/hfo_classifier.onnx --ts models/hfo_classifier.pt
```

8) Use in GUI:
- In the Deep Learning window, set Model Path to the exported `.onnx` or `.pt`.
- Detect EOIs with Hilbert/STE/MNI again (on new data).
- Click "Classify EOIs" to classify the new EOIs using your trained model.
- Classifications appear in the Scores tree with probabilities.

## Old manual approach (not required if using GUI export)

## Files
- `model.py` – Simple 1D CNN for binary classification.
- `data.py` – Dataset and dataloader helpers; per-segment z-scoring.
- `train.py` – Training loop with early stopping on validation loss.
- `export.py` – Load a checkpoint and export TorchScript and ONNX.
- `prepare_segments.py` – Extract EOIs to `.npy` segments and build a manifest.

## Notes
- This is intentionally minimal; tune architecture and preprocessing to your data.
- If you prefer time-frequency inputs, replace the dataset transform to produce spectrograms and switch to a 2D CNN.
