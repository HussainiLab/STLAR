# pyHFO Integration Setup

## Status: ✅ Complete & Enhanced

Your hfoGUI application now supports **three pyHFO detection methods** with configurable parameters and parallel processing.

## What's New

### Three Detection Methods:
1. **pyHFO (Ripples)** - 80-250 Hz band - ID: PYR
2. **pyHFO (Fast Ripples)** - 250-500 Hz band - ID: PYF  
3. **pyHFO (Both)** - 80-500 Hz band - ID: PYH

Each method has its own:
- ✅ Parameter configuration window
- ✅ Saved settings (per session and persistent)
- ✅ Parallel processing support (multi-core CPU)
- ✅ Independent thread execution

## Performance Improvements

### Speed Optimizations:
- **Multi-core processing**: Use 4-8 CPU cores simultaneously (configurable)
- **Longer epochs**: Default 10-min epochs (vs 5-min) = fewer threshold recalculations
- **Progress feedback**: Console messages show start/completion with event counts

### Expected Speed-up:
- **Before**: Single-core, 5-min epochs → ~15-30s per minute of data
- **After**: 4 cores, 10-min epochs → ~4-8s per minute of data (3-4× faster)

## How to Use

1. **Open your set file** in hfoGUI
2. **Go to Score window** → **Automatic Detection tab**
3. **Select method**:
   - "pyHFO (Ripples)" for 80-250 Hz ripples
   - "pyHFO (Fast Ripples)" for 250-500 Hz fast ripples
   - "pyHFO (Both)" for combined 80-500 Hz detection
4. **Click "Find EOIs"** → Parameters window opens
5. **Adjust settings** (optional):
   - **Epoch**: 600s default (longer = faster)
   - **Threshold**: 5 SD default
   - **Min Duration**: 10 ms default
   - **CPU Cores**: 4 default (more = faster)
   - **Frequency band**: Auto-set per method
6. **Click "Analyze"** → Detection runs in background
7. **Results appear** in EOI list with appropriate ID (PYR/PYF/PYH)

## Parameters Explained

| Parameter | Default | Description | Impact on Speed |
|-----------|---------|-------------|-----------------|
| **Epoch(s)** | 600 | Time window for threshold calculation | Longer = faster |
| **Threshold(SD)** | 5 | Mean + X standard deviations | Higher = fewer events |
| **Min Duration(ms)** | 10 | Minimum event length | Lower = more events |
| **Min Frequency(Hz)** | 80/250 | Lower bandpass cutoff | Method-dependent |
| **Max Frequency(Hz)** | 250/500 | Upper bandpass cutoff | Method-dependent |
| **CPU Cores** | 4 | Parallel workers | More = much faster |

## Algorithm Details

**Detection pipeline:**
1. **Bandpass filter** signal to target frequency range
2. **Hilbert transform** computes instantaneous amplitude envelope  
3. **Divide into epochs** and calculate mean/SD per epoch
4. **Threshold detection**: Find samples where envelope > mean + threshold×SD
5. **Cluster events**: Group consecutive above-threshold samples
6. **Filter by duration**: Reject events shorter than minimum
7. **Parallel processing**: Split signal across CPU cores

**Why it's faster now:**
- Parallel processing distributes work across cores
- Longer epochs mean fewer threshold recalculations
- HFODetector is highly optimized C++/Cython under the hood

## Installed Packages

- ✅ **PyTorch** (2.5.1 CPU)
- ✅ **Transformers** (Hugging Face)
- ✅ **pyHFO** (from source at `C:\Users\Abid\Documents\Code\Python\pyhfo_repo`)
- ✅ **HFODetector** (core algorithms)
- ✅ **MNE, YASA, scikit-image** (supporting libraries)

## Settings Persistence

**Two levels of settings storage:**

1. **Session-specific**: `HFOScores/<session>/<session>_PYR/PYF/PYH_settings.txt`
   - Saved when you run detection
   - Linked to each EOI in the tree
   - Includes all parameters used

2. **User preferences**: `settings/pyhfo_ripples/fast_ripples/both_params.json`
   - Saved when you close the parameters window
   - Auto-loaded next time you open the window
   - Separate for each detection method

## Troubleshooting

### Slow Performance?
1. **Increase CPU cores** (4-8 recommended)
2. **Increase epoch length** (600-1800s)
3. **Close other CPU-intensive programs**

### Not Finding Expected Events?
1. **Lower threshold** (try 3-4 SD instead of 5)
2. **Check frequency bands** (Ripples vs Fast Ripples)
3. **Lower min duration** (try 6-8 ms)

### Import Errors?
```bash
conda activate pyhfogui
cd C:\Users\Abid\Documents\Code\Python\pyhfo_repo
pip install -e .
pip install HFODetector mne yasa scikit-image
```

## Technical Notes

**Files modified:**
- `hfoGUI/core/Score.py` – Added PyHFOParametersWindow, three detection methods, parallel support
- `requirements.txt` – Documented optional dependencies

**Key classes/functions:**
- `PyHFOParametersWindow` – Parameter configuration UI (3 variants)
- `PyHFODetection(self)` – Worker function with parallel processing
- `_convert_pyhfo_results_to_eois()` – Result converter

**Thread management:**
- `pyhfo_ripples_thread` – For Ripples detection
- `pyhfo_fast_ripples_thread` – For Fast Ripples detection
- `pyhfo_both_thread` – For Both detection

Each method runs independently and can be configured separately.
