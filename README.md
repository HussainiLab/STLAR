# hfoGUI

This is an extended version of the original [hfoGUI](https://github.com/HussainiLab/hfoGUI) by Geoff Barrett and the HussainiLab. The original package was designed to visualize High Frequency Oscillations (HFOs) in LFP data recorded in the Tint format (from Axona's dacqUSB). This version adds additional automated detection methods (STE, MNI, Deep Learning) and a complete deep learning training pipeline.

# Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YourUsername/PyhfoGUI.git
   cd PyhfoGUI
   ```

2. **Create a conda environment (recommended):**
   ```bash
   conda create -n pyhfogui python=3.10
   conda activate pyhfogui
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the GUI:**
   ```bash
   python -m hfoGUI
   ```

# Dependencies

Core dependencies:
- PyQt5 - GUI framework
- pyqtgraph - Fast plotting
- numpy - Numerical computing
- scipy - Signal processing
- matplotlib - Plotting
- pandas - Data manipulation
- pyfftw - Fast Fourier transforms
- opencv-python - Image processing
- torch - Deep learning training/inference
- onnxruntime - ONNX model inference

# Usage

## GUI Mode (Default)
Launch the graphical interface:
```bash
python -m hfoGUI
```

### Intan RHD → Tint Conversion (GUI)

Convert Intan `.rhd` recordings into Tint `.set` + `.egf`/`.eeg` files directly from the GUI:

- Click "Intan Convert" next to "Import Set" on the main window
- Choose an `.rhd` file; canceling simply exits without conversion
- The converter auto-detects related chunked files in the same folder and concatenates them
- Output is saved next to the chosen file in a subfolder named after the session (e.g., `prefix_YYMMDD_HHMMSS/`)
- Creates `.egf` when sample rate ≥ 4.8 kHz, otherwise `.eeg`
- Converts a single amplifier channel (default: `A-000`) for quick testing

Example output:
```
Session name: sample_session
Files saved to: E:\DATA\sample_session
Files created: sample_session.set, sample_session.egf1, sample_session.egf2, ...
```

## CLI Mode - Automated Batch Processing

### Hilbert Detection Batch Command

Process HFO detection automatically using the Hilbert envelope method without launching the GUI. Supports both single-file and directory batch processing.

#### Basic Syntax
```bash
python -m hfoGUI hilbert-batch --file <path> [options]
```

#### Single File Processing
```bash
python -m hfoGUI hilbert-batch \
  --file /path/to/data.egf \
  --set-file /path/to/data.set \
  --epoch-sec 300 \
  --threshold-sd 4 \
  --min-duration-ms 12 \
  --min-freq 250 \
  --max-freq 600 \
  --required-peaks 6 \
  --required-peak-threshold-sd 3 \
  --boundary-percent 30
```

#### Directory Batch Processing
Recursively process all `.egf` and `.eeg` files in a directory (prioritizes `.egf` when both exist):
```bash
python -m hfoGUI hilbert-batch \
  --file /path/to/data/directory/ \
  --epoch-sec 180 \
  --threshold-sd 4 \
  --min-duration-ms 10 \
  --min-freq 80 \
  --max-freq 500 \
  --required-peaks 6 \
  --required-peak-threshold-sd 2 \
  --boundary-percent 25 \
  --verbose
```

#### Command-Line Options

**Required:**
- `--file PATH`: Path to `.eeg`/`.egf` file or directory to process recursively

**Optional Detection Parameters:**
- `--set-file PATH`: Path to `.set` calibration file (auto-detected if not specified)
- `--epoch-sec SECONDS`: Epoch window size in seconds (default: 300)
- `--threshold-sd SD`: Detection threshold in standard deviations (default: 3.0)
- `--min-duration-ms MS`: Minimum event duration in milliseconds (default: 10.0)
- `--min-freq HZ`: Minimum frequency for bandpass filter in Hz (default: 80 for EEG, 80 for EGF)
- `--max-freq HZ`: Maximum frequency for bandpass filter in Hz (default: 125 for EEG, 500 for EGF)
- `--required-peaks N`: Minimum number of peaks required in event (default: 6)
- `--required-peak-threshold-sd SD`: Peak detection threshold in SD (default: 2.0)
- `--no-required-peak-threshold`: Disable peak threshold (count all peaks)
- `--boundary-percent PERCENT`: Boundary detection threshold as % of main threshold (default: 30%)
- `--skip-bits2uv`: Skip bits-to-microvolts conversion if `.set` file missing
- `--output PATH`: Custom output directory (default: `HFOScores/<session>/`)
- `--verbose`, `-v`: Enable detailed progress logging

#### Output Files

For each processed session, the following files are created:
- `<session>_HIL.txt`: Tab-separated file with detected HFO events (ID, start time, stop time, settings)
- `<session>_settings.json`: JSON file with all detection parameters used

Default output location: `HFOScores/<session>/`

#### Examples

**Example 1: Process single file with custom parameters**
```bash
python -m hfoGUI hilbert-batch \
  --file E:\DATA\recording.egf \
  --epoch-sec 120 \
  --threshold-sd 5 \
  --min-freq 200 \
  --max-freq 600 \
  --verbose
```

**Example 2: Batch process directory (recursive)**
```bash
python -m hfoGUI hilbert-batch \
  --file E:\DATA\Experiments\ \
  --epoch-sec 300 \
  --threshold-sd 4 \
  --min-duration-ms 12 \
  --required-peaks 8 \
  --verbose
```
This will:
1. Scan all subdirectories for `.egf` and `.eeg` files
2. Auto-detect matching `.set` files
3. Process each file independently
4. Print a summary report showing total files, success/failure counts, and HFO statistics

**Example 3: Process without calibration file**
```bash
python -m hfoGUI hilbert-batch \
  --file recording.egf \
  --skip-bits2uv \
  --epoch-sec 180
```

#### Batch Processing Summary

When processing directories, a summary report is displayed:
```
============================================================
BATCH PROCESSING SUMMARY
============================================================
Total files found:     15
Successfully processed: 14
Failed:                 1
Total HFOs detected:    1247
Average per file:       89.1
============================================================
```

#### Notes
- Directory mode automatically matches `.set` files by basename
- When both `.eeg` and `.egf` exist with same basename, only `.egf` is processed
- Large epoch windows (e.g., 300s) work correctly with empty epochs
- Failed files don't stop batch processing (errors logged, processing continues)
- Use `--verbose` for per-epoch progress and detailed error traces

## Intan Conversion (CLI)

Run the converter without the GUI. If no file argument is provided, a file picker opens; canceling exits.

```bash
# As a module
python -m hfoGUI.intan_rhd_format

# Or direct script path
python hfoGUI/intan_rhd_format.py E:\DATA\recording_250k_240101_120000.rhd
```

Outputs are created in a session-named subfolder next to the input `.rhd`. The converter selects channel `A-000` by default and produces `.egf` or `.eeg` depending on input sample rate. A bundled sample file is available at `hfoGUI/core/load_intan_rhd_format/sampledata.rhd` for quick testing.

# Authors
* **Geoff Barrett** - [Geoff’s GitHub](https://github.com/GeoffBarrett)
* **HussainiLab** - [hfoGUI Repository](https://github.com/HussainiLab/hfoGUI)

**Updated (v3.0):** Added Intan RHD → Tint converter (GUI + CLI), global UI theme options, deep learning classification with training pipeline (original 1D CNN architecture), and automatic manifest splitting for DL workflows.

# Acknowledgments

This project incorporates HFO detection concepts and methodologies inspired by the following research:

- **Burnos et al. (2014)**: Foundational work on automated HFO detection using time-frequency analysis, introducing the Short-Term Energy (STE) and Mean-Normalized Integrated (MNI) detection methods. Our local scipy-based implementations are inspired by these concepts.
  - Burnos, S., Hilfiker, P., Sürücü, O., Scholkmann, F., Krayenbühl, N., Grundwald, T., et al. (2014). Human intracranial high frequency oscillations (HFOs) detected by automatic time-frequency analysis. *PLoS ONE* 9:e94381. doi: 10.1371/journal.pone.0094381  
    https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0094381

- **RippleLab (Navarrete et al., 2016)**: Comprehensive HFO detection application, whose Hilbert transform-based detection method inspired hfoGUI's original implementation by Geoff Barrett.
  - Navarrete M, Alvarado‐Rojas C, Le Van Quyen M, et al. RIPPLELAB: a comprehensive application for the detection, analysis and classification of high frequency oscillations in electroencephalographic signals. *PLoS ONE* 2016;11:e0158276.  
    https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0158276

The deep learning classification pipeline uses an original 1D CNN architecture designed for raw waveform classification, with custom training and export tooling developed for this project.

We thank the authors of the above foundational works for their contributions to automated HFO detection.

# License

This project is licensed under the GNU  General  Public  License - see the [LICENSE.md](../master/LICENSE) file for details
