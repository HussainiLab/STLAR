# STLAR: Spatio-Temporal LFP Analyzer

STLAR (pronounced Stellar) is a Spatio-Temporal LFP analysis tool combining hfoGUI and Spatial Spectral Mapper.

## Features

### 🕐 Temporal Analysis
- HFO detection (ripples 80-250 Hz, fast ripples 250-500 Hz)
- Multi-band filtering and visualization
- Time-frequency analysis (Stockwell transform)
- Event scoring and annotation
- Multiple automated detection algorithms (Hilbert, STE, MNI, Consensus, Deep Learning)

### 🗺️ Spatial Analysis
- Frequency heatmaps across arena
- Position tracking overlay
- Spatial power distribution
- Arena coverage visualization
- Multi-channel spatial mapping

### 🔗 Spatio-Temporal Integration
- Synchronized temporal-spatial views
- HFO location mapping
- Behavioral state-dependent analysis
- Cross-region coordination metrics

### 🤖 Deep Learning
- Automated HFO classification
- Train custom models on your data
- PyTorch (.pt) and ONNX export
- Pre-trained models available

### ⚡ Batch Processing
- Multi-file batch CLI processing with 5 detection methods
- Recursive directory scanning
- Configurable detection thresholds
- Progress tracking and summary statistics

## Installation
```bash
# Clone repository
git clone https://github.com/HussainiLab/STLAR.git
cd STLAR

# Install dependencies
pip install -r requirements.txt

# (Optional) install package locally
pip install -e .
```

## Project Structure
```
STLAR/
├── hfoGUI/           # Temporal analysis (HFO detection)
├── stlar/            # Main package wrapper
├── spatial_mapper/   # Spatial spectral mapping
├── docs/             # Documentation
├── tests/            # Test files
├── settings/         # User configuration files (gitignored)
├── HFOScores/        # User output data (gitignored)
└── main.py           # Entry point
```

**Note:** `settings/` and `HFOScores/` directories contain user-generated configuration and output data. These are created automatically on first run and should not be committed to version control.

## Quick Start

### HFO Analysis GUI (Temporal)
```bash
python -m stlar
```

### Spatial Spectral Mapper GUI
```bash
python spatial_mapper/src/main.py
```

### Batch Processing (Command Line)

See [Batch Processing](#batch-processing-cli) section below for detailed syntax.

### Python API
```python
from stlar.core.analysis import TemporalAnalyzer, SpatialAnalyzer

# Temporal analysis
temporal = TemporalAnalyzer()
ripples = temporal.detect_ripples(lfp_data)

# Spatial analysis
spatial = SpatialAnalyzer()
heatmaps = spatial.create_frequency_maps(lfp_data, position_data)
```

## Batch Processing (CLI)

STLAR provides a powerful command-line interface for batch processing HFO detection on multiple files. The CLI supports 5 different detection algorithms that can be run individually or combined via consensus voting.

### Invocation

All batch commands are invoked through:
```bash
python -m hfoGUI <command> [options]
```

### File Input Modes

All commands support both **single file** and **directory (batch)** modes:

- **Single file:** Process one .eeg or .egf file
- **Directory (recursive):** Automatically discovers all .eeg and .egf files in a directory tree
- **Smart file pairing:** If both .eeg and .egf exist with the same basename, only .egf is processed (since .eeg files typically don't contain HFOs)

### Output Structure

Detected events are saved as tab-separated values with the following structure:

```
ID#:          Start Time(ms):   Stop Time(ms):  Settings File:
HIL1          1234.56           1245.67         /path/to/session_HIL_settings.json
HIL2          2345.67           2356.78         /path/to/session_HIL_settings.json
```

**Locations:**
- Custom output: `--output` directory if specified
- Default: `HFOScores/<session_name>/<session_name>_<METHOD>.txt`

Each detection run saves corresponding settings in a JSON file for reproducibility.

### Detection Methods

#### 1. Hilbert Detection
Envelope-based detection using analytic signal (Hilbert transform).

**Command:**
```bash
python -m hfoGUI hilbert-batch -f <data_file_or_directory> [options]
```

**Examples:**

Single file:
```bash
python -m hfoGUI hilbert-batch \
    -f data/recording.eeg \
    --threshold-sd 3.0 \
    --min-freq 80 \
    --max-freq 250 \
    --epoch-sec 300 \
    -v
```

Directory batch with custom output:
```bash
python -m hfoGUI hilbert-batch \
    -f /data/recording_session/ \
    -s /data/recording_session/ \
    -o results/hilbert_detections/ \
    --threshold-sd 2.5 \
    --required-peaks 5 \
    -v
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-f, --file` | str | **required** | Path to .eeg/.egf file or directory |
| `-s, --set-file` | str | auto-detect | .set file or directory (for scaling calibration) |
| `-o, --output` | str | HFOScores/ | Output directory for results |
| `--epoch-sec` | float | 300 | Epoch length in seconds for analysis |
| `--threshold-sd` | float | 3.0 | Envelope threshold in SD above mean |
| `--min-duration-ms` | float | 10.0 | Minimum event duration (ms) |
| `--min-freq` | float | 80 | Minimum bandpass frequency (Hz) |
| `--max-freq` | float | 125 (EEG) / 500 (EGF) | Maximum bandpass frequency (Hz) |
| `--required-peaks` | int | 6 | Minimum peaks in rectified signal |
| `--required-peak-threshold-sd` | float | 2.0 | Peak threshold SD above mean |
| `--no-required-peak-threshold` | flag | off | Disable peak-threshold check |
| `--boundary-percent` | float | 30.0 | Percent of threshold to find boundaries |
| `--skip-bits2uv` | flag | off | Skip bits-to-uV conversion if .set missing |
| `-v, --verbose` | flag | off | Verbose progress logging |

---

#### 2. STE (Short-Term Energy / RMS) Detection
Fast detection based on RMS energy in sliding windows.

**Command:**
```bash
python -m hfoGUI ste-batch -f <data_file_or_directory> [options]
```

**Examples:**

```bash
python -m hfoGUI ste-batch \
    -f data/recording.eeg \
    --threshold 3.0 \
    --window-size 0.01 \
    --overlap 0.5 \
    --min-freq 80 \
    --max-freq 250
```

Directory batch:
```bash
python -m hfoGUI ste-batch \
    -f /data/recordings/ \
    -o results/ste_detections/ \
    --threshold 2.5 \
    --window-size 0.01 \
    --overlap 0.75 \
    -v
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-f, --file` | str | **required** | Path to .eeg/.egf file or directory |
| `-s, --set-file` | str | auto-detect | .set file or directory |
| `-o, --output` | str | HFOScores/ | Output directory |
| `--threshold` | float | 3.0 | RMS threshold (SD or absolute value) |
| `--window-size` | float | 0.01 | Window size in seconds |
| `--overlap` | float | 0.5 | Window overlap fraction (0-1) |
| `--min-freq` | float | 80 | Minimum frequency (Hz) |
| `--max-freq` | float | 500 | Maximum frequency (Hz) |
| `--skip-bits2uv` | flag | off | Skip scaling conversion |
| `-v, --verbose` | flag | off | Verbose logging |

---

#### 3. MNI Detection
Percentile-based detection using baseline power statistics.

**Command:**
```bash
python -m hfoGUI mni-batch -f <data_file_or_directory> [options]
```

**Examples:**

```bash
python -m hfoGUI mni-batch \
    -f data/recording.eeg \
    --baseline-window 10.0 \
    --threshold-percentile 99.0 \
    --min-freq 80
```

Directory batch:
```bash
python -m hfoGUI mni-batch \
    -f /data/recordings/ \
    -o results/mni_detections/ \
    --baseline-window 15.0 \
    --threshold-percentile 98.5 \
    -v
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-f, --file` | str | **required** | Path to .eeg/.egf file or directory |
| `-s, --set-file` | str | auto-detect | .set file or directory |
| `-o, --output` | str | HFOScores/ | Output directory |
| `--baseline-window` | float | 10.0 | Baseline window in seconds |
| `--threshold-percentile` | float | 99.0 | Threshold percentile (0-100) |
| `--min-freq` | float | 80 | Minimum frequency (Hz) |
| `--skip-bits2uv` | flag | off | Skip scaling conversion |
| `-v, --verbose` | flag | off | Verbose logging |

---

#### 4. Consensus Detection
Combines Hilbert, STE, and MNI detections using configurable voting strategy.

**Command:**
```bash
python -m hfoGUI consensus-batch -f <data_file_or_directory> [options]
```

**Examples:**

Basic consensus (majority voting):
```bash
python -m hfoGUI consensus-batch \
    -f data/recording.eeg \
    --voting-strategy majority \
    --overlap-threshold-ms 10.0
```

Strict consensus (all 3 methods must agree):
```bash
python -m hfoGUI consensus-batch \
    -f /data/recordings/ \
    -o results/consensus_detections/ \
    --voting-strategy strict \
    --overlap-threshold-ms 5.0 \
    --hilbert-threshold-sd 3.5 \
    --ste-threshold 2.5 \
    --mni-percentile 98.0 \
    -v
```

Lenient consensus (any method detection):
```bash
python -m hfoGUI consensus-batch \
    -f data/recording.eeg \
    --voting-strategy any \
    --overlap-threshold-ms 15.0
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-f, --file` | str | **required** | Path to .eeg/.egf file or directory |
| `-s, --set-file` | str | auto-detect | .set file or directory |
| `-o, --output` | str | HFOScores/ | Output directory |
| `--voting-strategy` | str | majority | `strict` (3/3), `majority` (2/3), or `any` (1/3) |
| `--overlap-threshold-ms` | float | 10.0 | Time window (ms) for overlapping detections |
| `--epoch-sec` | float | 300 | Hilbert epoch length (seconds) |
| `--hilbert-threshold-sd` | float | 3.5 | Hilbert envelope threshold (SD) |
| `--ste-threshold` | float | 2.5 | STE/RMS threshold |
| `--mni-percentile` | float | 98.0 | MNI threshold percentile |
| `--min-duration-ms` | float | 10.0 | Minimum event duration (ms) |
| `--min-freq` | float | 80 | Minimum frequency (Hz) |
| `--max-freq` | float | 125 (EEG) / 500 (EGF) | Maximum frequency (Hz) |
| `--required-peaks` | int | 6 | Hilbert minimum peaks |
| `--required-peak-sd` | float | 2.0 | Hilbert peak threshold (SD) |
| `--skip-bits2uv` | flag | off | Skip scaling conversion |
| `-v, --verbose` | flag | off | Verbose logging |

---

#### 5. Deep Learning Detection
Uses a pre-trained or custom neural network model for detection.

**Command:**
```bash
python -m hfoGUI dl-batch -f <data_file_or_directory> --model-path <model> [options]
```

**Examples:**

```bash
python -m hfoGUI dl-batch \
    -f data/recording.eeg \
    --model-path models/hfo_detector.pt \
    --threshold 0.5 \
    --batch-size 32
```

Directory batch with custom threshold:
```bash
python -m hfoGUI dl-batch \
    -f /data/recordings/ \
    -o results/dl_detections/ \
    --model-path models/hfo_detector.pt \
    --threshold 0.7 \
    --batch-size 64 \
    -v
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-f, --file` | str | **required** | Path to .eeg/.egf file or directory |
| `-s, --set-file` | str | auto-detect | .set file or directory |
| `-o, --output` | str | HFOScores/ | Output directory |
| `--model-path` | str | **required** | Path to trained model (.pt or .onnx) |
| `--threshold` | float | 0.5 | Detection probability threshold (0-1) |
| `--batch-size` | int | 32 | Inference batch size |
| `--skip-bits2uv` | flag | off | Skip scaling conversion |
| `-v, --verbose` | flag | off | Verbose logging |

---

### Batch Processing Workflow Examples

**Example 1: Quick single-file screening**
```bash
# Fast STE detection with verbose output
python -m hfoGUI ste-batch \
    -f data/session_1.eeg \
    --threshold 2.5 \
    -v
```

**Example 2: Batch directory with Hilbert (default settings)**
```bash
# Process entire directory, save to HFOScores/
python -m hfoGUI hilbert-batch \
    -f /data/rat_session/ \
    -s /data/rat_session/ \
    -v
```

**Example 3: High-confidence consensus detection**
```bash
# Strict consensus voting across directory
python -m hfoGUI consensus-batch \
    -f /data/recordings/ \
    -o /results/strict_consensus/ \
    --voting-strategy strict \
    --overlap-threshold-ms 5.0 \
    --hilbert-threshold-sd 3.5 \
    --ste-threshold 3.0 \
    --mni-percentile 99.0 \
    -v
```

**Example 4: Deep learning on pre-processed files**
```bash
# Use trained model on directory of files
python -m hfoGUI dl-batch \
    -f /data/preprocessed/ \
    -o /results/dl_predictions/ \
    --model-path /models/my_trained_detector.pt \
    --threshold 0.6 \
    --batch-size 128 \
    -v
```

### Output & Interpretation

After batch processing completes, you'll see a summary:

```
============================================================
BATCH PROCESSING SUMMARY
============================================================
Total files found:      5
Successfully processed: 5
Failed:                 0
Total HFOs detected:    1247
Average per file:       249.4
============================================================
```

**Output files:**
- **Scores:** `<session>_<METHOD>.txt` (tab-delimited, importable into Excel/analysis software)
- **Settings:** `<session>_<METHOD>_settings.json` (parameters used for reproducibility)

### Common Troubleshooting

**"No .set file found"**
- Use `--skip-bits2uv` to process without scaling calibration
- Or provide `--set-file` with the directory containing .set files

**"No .eeg or .egf files found in directory"**
- Verify file extensions are lowercase (.eeg, .egf)
- Check directory path is correct
- Use `-v` flag to see what files are discovered

**Sensitivity too high/low**
- **Too many false positives:** Increase threshold (e.g., `--threshold-sd 4.0` for Hilbert)
- **Too many false negatives:** Decrease threshold (e.g., `--threshold-sd 2.0` for Hilbert)
- Try **consensus** voting with different methods to find balanced detections

## Module Structure

### Temporal Analysis (HFO Detection)
- Location: `hfoGUI/` and `stlar/`
- Entry: `python -m stlar`
- See: [docs/CONSENSUS_QUICKSTART.md](docs/CONSENSUS_QUICKSTART.md), [docs/CONSENSUS_DETECTION.md](docs/CONSENSUS_DETECTION.md)

### Spatial Analysis (Spectral Mapper)
- Location: `spatial_mapper/`
- Entry: `python spatial_mapper/src/main.py` (GUI) or `python spatial_mapper/src/batch_ssm.py` (CLI)
- See: [spatial_mapper/README.md](spatial_mapper/README.md)

## Original Tools

This project unifies:
- [PyhfoGUI](https://github.com/HussainiLab/PyhfoGUI) - Temporal LFP analysis
- [Spatial_Spectral_Mapper](https://github.com/HussainiLab/Spatial_Spectral_Mapper) - Spatial LFP analysis

## Documentation

### HFO Detection (Temporal Analysis)
- [Consensus Detection Overview](docs/CONSENSUS_DETECTION.md)
- [Consensus Quick Start](docs/CONSENSUS_QUICKSTART.md)
- [Consensus Summary](docs/CONSENSUS_SUMMARY.md)

### Spatial Spectral Mapper
- [Spatial Mapper Guide](spatial_mapper/README.md)

## License

GPL-3.0 License - see [LICENSE](LICENSE) file for details.

## Contact

- Issues: https://github.com/HussainiLab/STLAR/issues

---

**STLAR** - Advancing spatio-temporal understanding of neural oscillations 🧠
