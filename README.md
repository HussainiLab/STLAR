# STLAR: Spatio-Temporal LFP Analyzer

![STLAR Overview Banner](docs/images/banner.png)

STLAR (or **Stellar**) combines temporal HFO detection, spatial spectral mapping, and optional deep learning workflows.

---

<a id="quickstart"></a>
## Quickstart

- Preferred Python: **3.12** (Conda recommended)

```bash
conda create -n stlar python=3.12
conda activate stlar
pip install -r requirements.txt
```

Run GUI:

```bash
python -m stlar gui
```

Run a simple CLI batch:

```bash
python -m stlar hilbert-batch -f path/to/data/
```

Windows reset tip (DLL / `_ctypes` issues after Python changes):

```bash
conda env remove -n stlar
conda create -n stlar python=3.12
conda activate stlar
pip install -r requirements.txt
```

---

## 📑 Table of Contents

### Getting Started
- [Quickstart](#quickstart)
- [Installation](#installation)
- [Features](#features)
- [Choose Your Workflow](#choose-your-workflow)

### Usage
- [GUI Workflow](#gui-workflow)
- [CLI Reference](#cli-reference)
  - [Detection Methods](#detection-methods)
- [HFO Metrics & Score Filtering](#hfo-metrics--score-filtering)
- [Spatial Mapping (batch-ssm)](#spatial-mapping-batch-ssm)
- [Complete Deep Learning Training Workflow](#complete-deep-learning-training-workflow)

### Advanced Docs
- [Advanced CLI Guide](docs/CLI_ADVANCED.md)
- [Advanced DL Training Guide](docs/DL_TRAINING_ADVANCED.md)
- [Consensus Detection](docs/CONSENSUS_DETECTION.md)
- [Detection Tuning](docs/DETECTION_TUNING.md)
- [Technical Reference](docs/TECHNICAL_REFERENCE.md)

### Support
- [Troubleshooting Installation](#troubleshooting-installation)
- [Getting Help](#getting-help)
- [Recent Changes](#recent-changes)

---

<a id="features"></a>
## Features

### Temporal Analysis
- HFO detection (Hilbert, STE, MNI, Consensus, Deep Learning)
- Scoring and event review workflow
- Time-frequency analysis (Stockwell transform)

### Spatial Analysis
- Arena heatmaps and trajectory-aware mapping
- PSD across positions and chunks
- Optional binned exports for downstream analysis

### Deep Learning
- Prepare training segments from EOIs
- Train and export custom models
- Use trained models in `dl-batch`

---

<a id="installation"></a>
## Installation

### Requirements
- Python 3.10–3.13 supported (3.12 preferred)
- `pip`
- ~2-3 GB free disk

### Install

```bash
git clone https://github.com/HussainiLab/STLAR.git
cd STLAR

conda create -n stlar python=3.12
conda activate stlar
pip install -r requirements.txt
```

Optional editable install:

```bash
pip install -e .
```

#### Deep Learning optional dependencies

CPU:

```bash
pip install torch onnxruntime
```

GPU (CUDA 11.8):

```bash
pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio
pip install onnxruntime-gpu
```

<a id="troubleshooting-installation"></a>
### Troubleshooting Installation

- `python` not found: try `python3`
- Conda not activating: run `conda init`, restart terminal
- Import errors: verify env active, rerun `pip install -r requirements.txt --upgrade`

---

## Choose Your Workflow

1. **GUI-first (recommended):** `python -m stlar gui`
2. **CLI batch processing:** `python -m stlar <command> ...`
3. **DL workflow:** `prepare-dl` → `train-dl` → `export-dl` → `dl-batch`

Recommended order for new users:
- Start with [GUI Workflow](#gui-workflow)
- Then use [CLI Reference](#cli-reference)
- Use advanced docs only when needed:
  - [Advanced CLI Guide](docs/CLI_ADVANCED.md)
  - [Advanced DL Training Guide](docs/DL_TRAINING_ADVANCED.md)

---

<a id="gui-workflow"></a>
## GUI Workflow

Launch:

```bash
python -m stlar gui
```

Basic flow:
1. Import `.set` / session folder
2. Open **Graph Settings** and choose source `.eeg` / `.egf`
3. Open **HFO Detection** → **Automatic Detection** tab
4. Run detection (Hilbert / STE / MNI / Consensus / DL)
5. Move selected EOIs to **Score** tab
6. Label and save scores

Spatial GUI:

```bash
python -m stlar spatial-gui
```

---

<a id="cli-reference"></a>
## CLI Reference

STLAR CLI format:

```bash
python -m stlar <command> [options]
```

Quick command groups:
- Detection: `hilbert-batch`, `ste-batch`, `mni-batch`, `consensus-batch`, `dl-batch`
- Analysis: `metrics-batch`, `filter-scores`, `batch-ssm`
- DL pipeline: `prepare-dl`, `train-dl`, `export-dl`

Basic examples:

```bash
python -m stlar hilbert-batch -f data/
python -m stlar consensus-batch -f data/ -v
python -m stlar dl-batch -f data/ --model-path models/hfo_detector.pt
```

### Detection Methods

- **Hilbert:** fast baseline detector for common ripple workflows
- **STE:** RMS-window energy detector
- **MNI:** percentile/baseline-driven detector
- **Consensus:** combines Hilbert + STE + MNI voting
- **DL:** model-based detection from exported `.pt` / `.onnx`

For full parameter tables and advanced recipes, see [Advanced CLI Guide](docs/CLI_ADVANCED.md).

---

## HFO Metrics & Score Filtering

Basic examples:

```bash
python -m stlar metrics-batch -f HFOScores/ -v
python -m stlar filter-scores -f HFOScores/session_HIL.txt --min-duration-ms 15 --max-duration-ms 120
```

Use this after detection to summarize event rates/durations and clean score files.

Advanced options (presets, behavior gating, custom speed thresholds):
- [Advanced CLI Guide](docs/CLI_ADVANCED.md#metrics-and-filtering)

---

## Spatial Mapping (batch-ssm)

Basic examples:

```bash
python -m stlar batch-ssm data/session.egf --ppm 595
python -m stlar batch-ssm data/ --ppm 595 --chunk-size 60
```

Optional exports:

```bash
python -m stlar batch-ssm data/ --ppm 595 --export-binned-csvs --plot-trajectory
```

Advanced spatial mapping usage and output details:
- [Advanced CLI Guide](docs/CLI_ADVANCED.md#spatial-mapping-batch-ssm)

---

## Complete Deep Learning Training Workflow

Minimal 4-step flow:

1. Prepare segments/manifests
2. Train model
3. Export model
4. Run `dl-batch` on new data

Basic commands:

```bash
python -m stlar prepare-dl --eoi-file detections.txt --egf-file recording.egf --output training_data --split-train-val
python -m stlar train-dl --train training_data/manifest_train.csv --val training_data/manifest_val.csv --epochs 15 --out-dir models
python -m stlar export-dl --ckpt models/best.pt --ts models/model.pt --onnx models/model.onnx
python -m stlar dl-batch -f new_recordings/ --model-path models/model.pt --threshold 0.5
```

Advanced training topics (CWT, batch training, GUI monitoring, tuning, troubleshooting):
- [Advanced DL Training Guide](docs/DL_TRAINING_ADVANCED.md)

Cross-link: advanced guide points back to this quick workflow.

---

<a id="module-structure"></a>
## Module Structure

### Temporal Analysis (HFO Detection)
- Location: `hfoGUI/` and `stlar/`
- Entry: `python -m stlar`
- Docs: [Consensus Quickstart](docs/CONSENSUS_QUICKSTART.md), [Consensus Detection](docs/CONSENSUS_DETECTION.md)

### Spatial Analysis
- Location: `spatial_mapper/`
- Entry: `python -m stlar spatial-gui` or `python -m stlar batch-ssm`
- Docs: [Advanced CLI Guide (Spatial Mapping)](docs/CLI_ADVANCED.md#spatial-mapping-batch-ssm), [Technical Reference](docs/TECHNICAL_REFERENCE.md)

### Deep Learning
- Location: `hfoGUI/dl_training/`
- Entry: `prepare-dl`, `train-dl`, `export-dl`, `dl-batch`
- Docs: [Advanced DL Training Guide](docs/DL_TRAINING_ADVANCED.md), [Training Visualization](docs/TRAINING_VISUALIZATION.md)

## Original Tools

This project unifies:
- [hfoGUI](https://github.com/HussainiLab/hfoGUI)
- [Spatial_Spectral_Mapper](https://github.com/HussainiLab/Spatial_Spectral_Mapper)

## Documentation

### Core Guides
- [Advanced CLI Guide](docs/CLI_ADVANCED.md)
- [Advanced DL Training Guide](docs/DL_TRAINING_ADVANCED.md)
- [Technical Reference](docs/TECHNICAL_REFERENCE.md)
- [Consensus Detection](docs/CONSENSUS_DETECTION.md)
- [Consensus Quickstart](docs/CONSENSUS_QUICKSTART.md)
- [Detection Tuning](docs/DETECTION_TUNING.md)
- [Containerization Guide](docs/CONTAINERIZATION_GUIDE.md)

### GUI / DL / CWT
- [GUI Quickstart](docs/GUI_QUICKSTART.md)
- [Training Visualization](docs/TRAINING_VISUALIZATION.md)
- [CWT Debug Guide](docs/CWT_DEBUG_GUIDE.md)
- [Preset Gating Guide](docs/PRESET_GATING_GUIDE.md)

<a id="api-documentation"></a>
## API Documentation

For developer-oriented APIs and internals:
- [Technical Reference](docs/TECHNICAL_REFERENCE.md)
- `hfoGUI/core/`
- `hfoGUI/dl_training/`
- `spatial_mapper/src/`

<a id="getting-help"></a>
## Getting Help

- Check [Troubleshooting Installation](#troubleshooting-installation)
- Use [GUI Quickstart](docs/GUI_QUICKSTART.md) for first-run issues
- Use [Advanced CLI Guide](docs/CLI_ADVANCED.md) for command options
- Use [Advanced DL Training Guide](docs/DL_TRAINING_ADVANCED.md) for model pipeline issues

<a id="recent-changes"></a>
## Recent Changes

- README simplified for quicker onboarding
- Advanced CLI details moved to [docs/CLI_ADVANCED.md](docs/CLI_ADVANCED.md)
- Advanced DL workflow details moved to [docs/DL_TRAINING_ADVANCED.md](docs/DL_TRAINING_ADVANCED.md)

---

GPL-3.0 License - see [LICENSE](LICENSE) for details.