# CWT Scalogram Debug Guide

This guide explains how to inspect CWT scalogram images to verify that the CWT preprocessing is working correctly.

## Quick Start

### 1. Training with CWT Debug Mode

Enable debug mode to save scalogram images during training:

```bash
python -m stlar train-dl \
  --train training_data/manifest_train.csv \
  --val training_data/manifest_val.csv \
  --model-type 6 \
  --use-cwt \
  --fs 4800 \
  --debug-cwt ./cwt_scalograms \
  --epochs 5
```

**Key argument:**
- `--debug-cwt ./cwt_scalograms` - Directory where scalogram PNG images will be saved

### 2. Inspect the Scalogram Images

The debug directory will contain PNG files like:
```
cwt_scalograms/
├── scalogram_000000_HFO.png          # True HFO sample
├── scalogram_000001_NonHFO.png       # Non-HFO sample
├── scalogram_000002_HFO.png
└── ...
```

**File naming convention:**
- `scalogram_XXXXXX_LABEL.png`
  - `XXXXXX` = Sample index (6-digit, zero-padded)
  - `LABEL` = HFO or NonHFO (based on training label)

### 3. Understanding the Scalogram Images

Each PNG image shows a 2D time-frequency representation:

```
┌─────────────────────────────────────────┐
│        CWT Scalogram - HFO              │
│                                         │
│  ▲                                      │
│  │  Freq  ████████████████████         │
│  │  63    ████░░████░░████░░████        │ Dark = High power
│  │        ████░░████░░████░░████        │ Light = Low power
│  │   :    :                :            │
│  │  32    ██░░░░░░░░░░░░░░░░██        │
│  │        ░░░░░░░░░░░░░░░░░░░░        │
│  │   0    ░░░░░░░░░░░░░░░░░░░░        │
│  └─────────────────────────────────────┤
│        0          Time           4800   │
│                                         │
│  Colorbar: Power (log scale)            │
└─────────────────────────────────────────┘
```

**Axes:**
- **X-axis**: Time samples (0 to signal length, ~4800 for 1 second at 4800 Hz)
- **Y-axis**: Frequency index (0-63 corresponding to 80-500 Hz)
- **Color intensity**: Power (logarithmic scale)
  - **Bright colors**: High power (strong signal)
  - **Dark colors**: Low power (weak signal)

## Interpreting Results

### Good CWT Scalogram (HFO)

A **healthy HFO scalogram** should show:
- ✅ Horizontal streaks in the frequency band (80-250 Hz for ripples)
- ✅ Bright colors indicating power concentration
- ✅ Temporal clustering (events grouped in time)
- ✅ Sharp boundaries (clear start/stop of event)

```
Example: Ripple (80-250 Hz)
High power concentrated in lower frequency band
```

### Poor CWT Scalogram (Noise)

A **non-HFO or noise scalogram** should show:
- ✅ Uniform/scattered coloring across all frequencies
- ✅ No clear horizontal streaks
- ✅ Low overall power
- ✅ Random patterns (no temporal clustering)

```
Example: Non-HFO (random noise)
Power spread uniformly, no clear pattern
```

### Common Issues

#### Issue 1: All images are completely dark
**Cause**: Signal amplitude too low or normalization broken
**Fix**: Check that your original signal has proper amplitude and z-score normalization is working

#### Issue 2: All images are completely bright
**Cause**: Signal is saturated or contains artifacts
**Fix**: Check for clipping in your original data, verify signal quality

#### Issue 3: Vertical stripes instead of horizontal
**Cause**: Transient noise or artifacts at specific frequencies
**Fix**: These are legitimate features! Verify they're not instrumental artifacts

#### Issue 4: Images are blank/white
**Cause**: CWT computation failed or matplotlib not installed
**Fix**: Install matplotlib with: `pip install matplotlib`

## Advanced Usage

### GUI Debug Mode (Development Only)

To enable debug mode in GUI DL detection, set environment variable before launching:

```bash
# Windows (PowerShell)
$env:STLAR_DEBUG_CWT = "C:\path\to\debug\directory"
python -m stlar gui

# Linux/macOS
export STLAR_DEBUG_CWT="/path/to/debug/directory"
python -m stlar gui
```

Then run DL detection. Scalogram images will be saved to the specified directory.

### Batch Training with Debug

Debug mode works in batch training mode too:

```bash
python -m stlar train-dl \
  --batch-dir ./sessions/ \
  --model-type 6 \
  --use-cwt \
  --debug-cwt ./all_scalograms \
  --epochs 5 \
  -v
```

This will save scalograms from all sessions in the `all_scalograms` directory.

## Technical Details

### CWT Parameters

The CWT scalogram uses these parameters (hardcoded):

| Parameter | Value | Description |
|-----------|-------|-------------|
| Wavelet | Morlet2 | Balanced time-frequency resolution |
| Frequencies | 80-500 Hz | Ripple and fast-ripple bands |
| Frequency Resolution | 64 levels | Divides 80-500 Hz into 64 evenly-spaced frequencies |
| Wavelet Parameter (w) | 6.0 | Standard Morlet; 6 = good balance |

### Power Computation

1. **Raw CWT**: Complex-valued matrix from Morlet wavelet
2. **Power**: Magnitude squared: $|CWT|^2$
3. **Log scaling**: $\log(1 + Power)$ for neural network stability
4. **Tensor**: Shape (1, 64, T) where T is time dimension

### Normalization

Each signal segment is independently normalized (z-score):
$$x_{norm} = \frac{x - \mu}{\sigma + 1e-8}$$

This ensures fair comparison across different recording sessions and amplitudes.

## Testing the Feature

A test script is provided:

```bash
cd STLAR
python test_cwt_debug.py
```

This creates synthetic sample data and generates scalograms to verify everything is working:
- ✅ Creates synthetic HFO and non-HFO signals
- ✅ Saves them as .npy segments
- ✅ Generates scalograms with debug mode
- ✅ Reports saved PNG files

## Troubleshooting

### Q: No PNG files are being saved
**A**: Check:
1. Directory path is correct and writable
2. matplotlib is installed: `pip install matplotlib`
3. Try test script: `python test_cwt_debug.py`

### Q: PNG files are created but all look the same
**A**: 
1. Verify your training labels are correct (check CSV file)
2. Check that signal values are reasonable (not all zeros)
3. Verify sampling frequency matches actual data

### Q: How many scalograms should be saved?
**A**: By default, only the first 3 samples from each dataset are logged to console. All samples are saved to disk. Check the directory for all PNG files.

## Using Scalograms for Quality Assurance

### Pre-Training Checklist

Before training, inspect a few scalograms:

1. **Label accuracy**: Do HFO scalograms look different from NonHFO?
2. **Signal quality**: Are there obvious artifacts or clipping?
3. **Frequency content**: Do ripples concentrate in expected frequency bands?

### Post-Training Analysis

After training with `--debug-cwt`:

1. **Success indicators**:
   - Training loss converges
   - HFO scalograms show distinctive patterns
   - Model achieves >80% validation accuracy

2. **Failure indicators**:
   - HFO and NonHFO scalograms look identical
   - Training loss doesn't improve
   - Model achieves <70% validation accuracy

## Questions or Issues?

If you encounter problems with CWT preprocessing:
1. Run the test script: `python test_cwt_debug.py`
2. Inspect generated scalograms in the output directory
3. Check console output for error messages
4. Verify matplotlib is installed: `pip install matplotlib`

---

**Last Updated**: January 2026
**Version**: 1.0
