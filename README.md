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

### Requirements
- **Python 3.8+** (check with `python --version`)
- **pip** (Python package manager, usually included with Python)
- ~2-3 GB disk space for dependencies

### Step-by-Step Installation (For Beginners)

#### 1. Install Python (if needed)
If you don't have Python installed:
- **Windows:** Download from [python.org](https://www.python.org/downloads/) → Run installer → ✅ Check "Add Python to PATH"
- **macOS:** Use [homebrew](https://brew.sh/) → `brew install python3`
- **Linux:** `sudo apt-get install python3 python3-pip`

Verify installation:
```bash
python --version
pip --version
```

#### 2. Clone the Repository
```bash
# Download STLAR source code
git clone https://github.com/HussainiLab/STLAR.git
cd STLAR
```

**No git installed?** Download ZIP from GitHub → Extract → Open command prompt in the folder

#### 3. Create Virtual Environment (Recommended)

Using a virtual environment keeps STLAR's dependencies isolated from your system Python.

**Option A: Using Conda (Recommended for Data Science)**

```bash
# Create new environment named "stlar"
conda create -n stlar python=3.10

# Activate the environment
conda activate stlar

# You should see (stlar) at the start of your command prompt
```

To activate later: `conda activate stlar`  
To deactivate: `conda deactivate`

**Option B: Using venv (Built into Python)**

```bash
# Windows
python -m venv stlar
stlar\Scripts\activate

# macOS/Linux
python3 -m venv stlar
source stlar/bin/activate

# You should see (stlar) at the start of your command prompt
```

To activate later: 
- Windows: `stlar\Scripts\activate`
- macOS/Linux: `source stlar/bin/activate`

To deactivate: `deactivate`

**Why use an environment?**
- ✅ Prevents dependency conflicts with other Python projects
- ✅ Easy to reset if something breaks (`conda env remove -n stlar`)
- ✅ Reproducible setup across machines

#### 4. Install Dependencies
```bash
# Windows/macOS/Linux - same command
pip install -r requirements.txt
```

**Takes 2-5 minutes depending on internet speed.** You should see packages downloading.

#### 5. (Optional) Install as Editable Package
```bash
# Allows updating STLAR without reinstalling
pip install -e .
```

**Done!** You can now run STLAR.

### Troubleshooting Installation

**"Command not found: python"**
- Try `python3` instead of `python`
- Windows: Add Python to PATH (search "environment variables" → add Python installation folder)

**"Permission denied" (macOS/Linux)**
```bash
pip install --user -r requirements.txt
```

**Conda environment not activating**
- Make sure conda is initialized: `conda init` → restart terminal
- Check environment exists: `conda env list`

**ImportError when running STLAR**
- Ensure environment is activated: `conda activate stlar` or `source stlar/bin/activate`
- Ensure all dependencies installed: `pip install -r requirements.txt --upgrade`
- Check you're in the STLAR directory: `pwd` (macOS/Linux) or `cd` (Windows)

---

## Quick Start

### 3 Ways to Use STLAR

#### 1️⃣ GUI (Easiest - Point & Click)

**Launch the HFO Analysis GUI:**
```bash
python -m stlar gui
```
Then open a data file and adjust detection parameters with sliders.

**Launch the Spatial Mapper GUI:**
```bash
python -m stlar spatial-gui
```

#### 2️⃣ Command Line (Batch Processing)

Process multiple files automatically:

```bash
# Detect HFOs using Hilbert method
python -m stlar hilbert-batch -f mydata/recording.eeg

# Process entire directory of files
python -m stlar hilbert-batch -f mydata/

# Use consensus voting (more reliable)
python -m stlar consensus-batch -f mydata/ --voting-strategy strict

# Spatial spectral mapping
python -m stlar batch-ssm mydata/ --ppm 595
```

Results save to `HFOScores/` by default. See [CLI Reference](#cli-reference) for all commands and options.

#### 3️⃣ Python API (Advanced)

```python
from hfoGUI.core.Detector import Detector

detector = Detector('mydata.eeg')
ripples = detector.detect_ripples(method='hilbert')
print(f"Found {len(ripples)} ripples")
```

### File Format Support

- **.eeg** - Tint format (most common)
- **.egf** - Intan format with embedded tracking
- **.edf** - Standard EDF format

### Project Structure
```
STLAR/
├── hfoGUI/           # HFO detection (temporal analysis)
├── spatial_mapper/   # Spatial spectral mapping
├── stlar/            # Main command-line dispatcher
├── docs/             # Documentation & guides
│   ├── TECHNICAL_REFERENCE.md     # Algorithms & formulas (for scientists)
│   ├── CONSENSUS_DETECTION.md     # Consensus voting details
│   ├── CONSENSUS_QUICKSTART.md    # Quick guide
│   └── CONSENSUS_SUMMARY.md       # Summary table
├── tests/            # Unit tests
├── settings/         # User config (auto-created)
├── HFOScores/        # Output directory (auto-created)
└── requirements.txt  # Dependencies
```

## CLI Reference

### Overview

The command-line interface (CLI) allows batch processing of multiple files with consistent parameters. All commands use the format:

```bash
python -m stlar <command> [options]
```

**Supported commands:**
- **HFO Detection:** `hilbert-batch`, `ste-batch`, `mni-batch`, `consensus-batch`, `dl-batch`
- **Spatial Mapping:** `batch-ssm`

**Key features:**
- ✅ Single-file or directory (recursive) processing
- ✅ Auto-detects .eeg and .egf files
- ✅ Progress tracking with `-v` (verbose) flag
- ✅ Customizable output directory with `-o`
- ✅ Reproducible with saved settings JSON files

### Output Format

Each detection creates files in the output directory:

```
HFOScores/
├── recording_name/
│   ├── recording_name_HIL.txt          # Detected HFOs (tab-separated)
│   ├── recording_name_HIL_settings.json  # Settings used (for reproducibility)
│   ├── recording_name_HFO_scores.eoi   # EOI format (for Tint)
│   └── ...
```

**Output file format:**
```
ID#     Start(ms)    Stop(ms)     Peak(µV)   Duration(ms)
HIL1    1234.56      1245.67      125.3      11.11
HIL2    2345.67      2356.78      118.9      11.11
...
```

**Common options across all commands:**
- `-f, --file` - Input file or directory (required)
- `-o, --output` - Where to save results (default: `HFOScores/`)
- `-v, --verbose` - Show progress details
- `-s, --set-file` - Location of .set files for scaling calibration

---

### Detection Methods

#### 1. Hilbert Detection
Envelope-based detection using analytic signal (Hilbert transform).

**Command:**
```bash
python -m stlar hilbert-batch -f <data_file_or_directory> [options]
```

**Examples:**

Single file:
```bash
python -m stlar hilbert-batch \
    -f data/recording.eeg \
    --threshold-sd 3.0 \
    --min-freq 80 \
    --max-freq 250 \
    --epoch-sec 300 \
    -v
```

Directory batch with custom output:
```bash
python -m stlar hilbert-batch \
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
python -m stlar ste-batch -f <data_file_or_directory> [options]
```

**Examples:**

```bash
python -m stlar ste-batch \
    -f data/recording.eeg \
    --threshold 3.0 \
    --window-size 0.01 \
    --overlap 0.5 \
    --min-freq 80 \
    --max-freq 250
```

Directory batch:
```bash
python -m stlar ste-batch \
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
python -m stlar mni-batch -f <data_file_or_directory> [options]
```

**Examples:**

```bash
python -m stlar mni-batch \
    -f data/recording.eeg \
    --baseline-window 10.0 \
    --threshold-percentile 99.0 \
    --min-freq 80
```

Directory batch:
```bash
python -m stlar mni-batch \
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
python -m stlar consensus-batch -f <data_file_or_directory> [options]
```

**Examples:**

Basic consensus (majority voting):
```bash
python -m stlar consensus-batch \
    -f data/recording.eeg \
    --voting-strategy majority \
    --overlap-threshold-ms 10.0
```

Strict consensus (all 3 methods must agree):
```bash
python -m stlar consensus-batch \
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
python -m stlar consensus-batch \
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
python -m stlar dl-batch -f <data_file_or_directory> --model-path <model> [options]
```

**Examples:**

```bash
python -m stlar dl-batch \
    -f data/recording.eeg \
    --model-path models/hfo_detector.pt \
    --threshold 0.5 \
    --batch-size 32
```

Directory batch with custom threshold:
```bash
python -m stlar dl-batch \
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
python -m stlar ste-batch \
    -f data/session_1.eeg \
    --threshold 2.5 \
    -v
```

**Example 2: Batch directory with Hilbert (default settings)**
```bash
# Process entire directory, save to HFOScores/
python -m stlar hilbert-batch \
    -f /data/rat_session/ \
    -s /data/rat_session/ \
    -v
```

**Example 3: High-confidence consensus detection**
```bash
# Strict consensus voting across directory
python -m stlar consensus-batch \
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
python -m stlar dl-batch \
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

---

## Spatial Mapping (batch-ssm)

The `batch-ssm` command performs batch spatial spectral analysis on .egf files with animal tracking data. It computes power spectral density (PSD) across spatial positions and optionally exports binned analyses.

### Basic Usage

```bash
# Single file
python -m stlar batch-ssm data/session001.egf --ppm 595

# Directory batch processing
python -m stlar batch-ssm data/ --ppm 595 --chunk-size 60

# With binned exports (4×4 grid)
python -m stlar batch-ssm data/ --ppm 595 --export-binned-jpgs --export-binned-csvs
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_path` | str | *required* | Path to .egf file or directory containing .egf files |
| `--ppm` | int | *required* | Pixels per meter for position calibration |
| `--chunk-size` | int | 30 | Duration of each analysis chunk in seconds |
| `--speed-filter` | float | 0 | Minimum speed threshold (cm/s) for filtering stationary periods |
| `--window` | float | 1.0 | Spectral window duration in seconds |
| `--export-binned-jpgs` | flag | False | Export spatial bin visualizations as JPEG images |
| `--export-binned-csvs` | flag | False | Export binned spectral data as CSV files |

### Examples

**Process single session with default parameters:**
```bash
python -m stlar batch-ssm recordings/rat01_day1.egf --ppm 595
```

**Batch process directory with 60-second chunks:**
```bash
python -m stlar batch-ssm recordings/ --ppm 595 --chunk-size 60
```

**Apply speed filtering (exclude stationary periods):**
```bash
python -m stlar batch-ssm recordings/ --ppm 595 --speed-filter 5.0
```

**Export binned analyses for spatial correlation studies:**
```bash
python -m stlar batch-ssm recordings/ --ppm 595 \
  --export-binned-jpgs \
  --export-binned-csvs \
  --chunk-size 60
```

### Output Structure

batch-ssm creates a timestamped output directory for each session:

```
<session_name>_SSMoutput_<YYYYMMDD_HHMMSS>/
├── <session>_sessionAverage.csv          # Session-wide PSD averages
├── <session>_chunk_000_psd.csv           # Per-chunk PSD data
├── <session>_chunk_001_psd.csv
├── ...
├── binned_analysis/                      # (if --export-binned-* used)
│   ├── <session>_bin_0_0.csv            # Spatial bin PSDs
│   ├── <session>_bin_0_1.csv
│   ├── ...
│   └── <session>_bin_visualization.jpg  # (if --export-binned-jpgs)
```

**CSV format:**
- Columns: Frequency bins (e.g., 0.5 Hz, 1.0 Hz, ..., 250 Hz)
- Rows: PSD values in (µV²/Hz) for each chunk or spatial bin

### Troubleshooting

**"No .egf files found"**
- Verify directory contains .egf files (Tint format)
- Check file permissions and path correctness

**"Position data not found in .egf"**
- Ensure tracking data is embedded in .egf file
- Verify correct .pxyabw file was integrated during Intan conversion

**Binned analysis produces empty bins**
- Check if tracking covers full environment (bins may be outside tracked area)
- Adjust `--speed-filter` threshold if filtering out too much data
- Verify ppm calibration is correct (incorrect scaling affects spatial binning)

## Module Structure

### Temporal Analysis (HFO Detection)
- Location: `hfoGUI/` and `stlar/`
- Entry: `python -m stlar`
- See: [docs/CONSENSUS_QUICKSTART.md](docs/CONSENSUS_QUICKSTART.md), [docs/CONSENSUS_DETECTION.md](docs/CONSENSUS_DETECTION.md)

### Spatial Analysis (Spectral Mapper)
- Location: `spatial_mapper/`
- Entry: `python -m stlar spatial-gui` (GUI) or `python -m stlar batch-ssm` (CLI)
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

## Getting Help

### Common Issues

**Installation problems?**
- Re-run: `pip install -r requirements.txt --upgrade`
- Check Python version: `python --version` (should be 3.8+)
- See Installation > Troubleshooting above

**Detection not finding HFOs?**
- Check you're using correct frequency band for your data
- Lower the threshold (e.g., `--threshold-sd 2.5`)
- Try different detection methods (consensus is most reliable)
- See CLI Reference for parameter tuning guides

**File format errors?**
- Ensure files are .eeg or .egf format
- Check file isn't corrupted: try opening in Tint first
- Verify file path has no spaces or special characters

**Need more details?**
- 📖 **For scientists & engineers:** Read [docs/TECHNICAL_REFERENCE.md](docs/TECHNICAL_REFERENCE.md) for all algorithms, formulas, and implementation details
- 📚 **For consensus voting:** Check [docs/CONSENSUS_DETECTION.md](docs/CONSENSUS_DETECTION.md) for theory
- 📖 **Quick reference:** [docs/CONSENSUS_SUMMARY.md](docs/CONSENSUS_SUMMARY.md) has a quick comparison table
- 🔧 Run with `-v` flag for verbose output

### Report Issues

Found a bug? Have a feature request?
→ Open an issue: [github.com/HussainiLab/STLAR/issues](https://github.com/HussainiLab/STLAR/issues)

Include:
- What command you ran
- Error message (full traceback)
- Python version: `python --version`
- Operating system

## License

GPL-3.0 License - see [LICENSE](LICENSE) file for details.

## Contact

- Issues: https://github.com/HussainiLab/STLAR/issues

---

**STLAR** - Advancing spatio-temporal understanding of neural oscillations 🧠

