# IMPLEMENTATION: Speed Column Support for Behavior Gating in metrics-batch

## Summary

Successfully implemented **Speed(cm/s)** column generation in detection batch commands, enabling proper behavior gating in `metrics-batch` and `filter-scores` commands.

## Problem (Before)

The `metrics-batch` and `filter-scores` CLI commands lacked PPM (Pixels Per Meter) parameter for behavior gating:

```bash
# This would silently fail to apply behavior gating
python -m stlar metrics-batch -f scores.txt \
    --preset Hippocampus \
    --behavior-gating \
    --speed-min 0.5 \
    --speed-max 5.0

# WARNING: behavior gating requested but no speed column found; skipping behavior gate
```

**Root Cause:** Detection commands didn't save speed values to output scores TSV files.

## Solution (After)

### 1. Modified Detection Commands
All detection batch commands now accept `--ppm` and `--pos-file` parameters and compute speed at event times:

```bash
python -m stlar hilbert-batch \
    -f recording.eeg \
    --ppm 500 \
    --pos-file recording.pos
```

**Output scores file now includes:**
```
ID#:    Start Time(ms):    Stop Time(ms):    Settings File:                      Speed(cm/s):
HIL1    42500              42750             /path/to/settings_HIL.json          2.35
HIL2    45200              45480             /path/to/settings_HIL.json          1.08
```

### 2. Speed Column Detection
`metrics-batch` and `filter-scores` will automatically detect and use the seeed column:

```bash
python -m stlar metrics-batch -f scores.txt \
    --preset Hippocampus \
    --behavior-gating \
    --speed-min 0.5 \
    --speed-max 5.0
    
# Now properly filters events by speed!
```

## Changed Files

### hfoGUI/cli.py

#### 1. New Function: `_compute_speed_at_events()`

```python
def _compute_speed_at_events(events_ms, pos_file, custom_ppm, fs_eeg, verbose=False):
    """
    Compute animal speed at event times from .pos file.
    
    - Loads .pos file using getpos() with custom_ppm
    - Computes 2D speed using speed2D()
    - Samples speed at event midpoints
    - Returns speed in cm/s
    """
```

**Parameters:**
- `events_ms`: Event times as Nx2 array [start_ms, stop_ms]
- `pos_file`: Path to .pos file
- `custom_ppm`: Pixels per meter (e.g., 500)
- `fs_eeg`: EEG sampling rate for time alignment
- `verbose`: Print debug info

**Returns:** Numpy array of speed values (cm/s) at each event, or None if file missing

#### 2. Modified Function: `_save_results()`

**Added parameter:** `speed_values=None`

**Behavior:**
- Creates DataFrame with standard columns: ID#, Start Time(ms), Stop Time(ms), Settings File
- **If speed_values provided:** Adds `Speed(cm/s):` column with mean speed at each event

#### 3. Updated Detection Processors

All five processors now compute and pass speed:
- `_process_single_file()` (Hilbert)
- `_process_ste_file()` (STE/RMS)
- `_process_mni_file()` (MNI)
- `_process_dl_file()` (Deep Learning)
- `_process_consensus_file()` (Consensus)

**Example (Hilbert):**
```python
events = hilbert_detect_events(raw_data, Fs, **params)

# NEW: Compute speed at events if pos file available
speed_values = None
if hasattr(args, 'ppm') and args.ppm and hasattr(args, 'pos_file') and args.pos_file:
    pos_file = Path(args.pos_file).expanduser()
    speed_values = _compute_speed_at_events(events, pos_file, args.ppm, Fs, verbose=args.verbose)

# NEW: Pass speed to save function
return _save_results(events, params, data_path, set_path, args, 
                     method_tag='HIL', speed_values=speed_values)
```

#### 4. Enhanced Argument Parsers

Added `--ppm` and `--pos-file` to all detection batch command parsers:

```python
# Hilbert parser additions
hilbert.add_argument('--pos-file', help='Optional .pos file for computing speed/behavior gating in output')
hilbert.add_argument('--ppm', type=int, help='Pixels per meter for .pos file conversion (e.g., 500)')

# Same for: ste, mni, consensus, dl parsers
```

## Usage Guide

### Scenario 1: Generate Scores with Speed Column

**Step 1: Run detection with --ppm and --pos-file**

```bash
python -m stlar hilbert-batch \
    -f /data/recording.eeg \
    --ppm 500 \
    --pos-file /data/recording.pos \
    -o /output/scores/ \
    -v
```

**Output:**
```
  Detected 127 events; saved scores -> /output/scores/recording_HIL.txt
    Loaded position data from recording.pos...
    Computed speed trace: 180000 samples at 50 Hz
```

**Resulting scores file:**
```tsv
ID#:    Start Time(ms):    Stop Time(ms):    Settings File:                      Speed(cm/s):
HIL1    4250.5             4500.2            /output/scores/recording_HIL_settings.json    2.35
HIL2    45200.1            45480.3           /output/scores/recording_HIL_settings.json    1.08
HIL3    87650.0            87920.5           /output/scores/recording_HIL_settings.json    12.4
```

### Scenario 2: Apply Behavior Gating with metrics-batch

**Now you can use behavior gating:**

```bash
python -m stlar metrics-batch \
    -f /output/scores/recording_HIL.txt \
    --preset Hippocampus \
    --behavior-gating \
    --duration-min 30 \
    -v
```

**Output shows filtering applied:**
```
Processing: recording_HIL.txt
  Events: 127
  Rate: 4.23 events/min
  Duration: 42.3 ± 28.5 ms
  Saved metrics -> metrics/recording_HIL_hfo_metrics_Hippocampus_gated.csv
```

**Metrics CSV includes behavior-gated stats:**
```csv
metric,value
total_events,42
recording_duration_minutes,30.0
event_rate_per_min,1.4
mean_duration_ms,41.2
median_duration_ms,38.5
min_duration_ms,15.0
max_duration_ms,118.5
std_duration_ms,26.3
```

### Scenario 3: Filter Scores with Speed Thresholds

```bash
python -m stlar filter-scores \
    -f /output/scores/recording_HIL.txt \
    --band ripple \
    --behavior-gating \
    --speed-min 0.5 \
    --speed-max 4.0 \
    -v
```

**Output:**
```
Processing: recording_HIL.txt
Filtered: 127 → 42 events (removed 85 below speed 0.5 or above 4.0 cm/s)
Saved → recording_HIL_filtered.txt
```

## Technical Details

### Speed Calculation

1. **Position data loading:** Uses `getpos()` from Tint_Matlab with `custom_ppm` parameter
2. **Speed computation:** Uses `speed2D()` which implements central difference:
   $$v[n] = \frac{\sqrt{(x[n+1] - x[n-1])^2 + (y[n+1] - y[n-1])^2}}{t[n+1] - t[n-1]}$$
3. **Event sampling:** Samples speed at event midpoint (average of start + stop times)
4. **Units:** Speed computation is in pixels/second; output is in cm/s via PPM conversion

### Data Flow

```
┌─────────────────┐
│  Detection Run  │
│ (e.g., hilbert) │
└────────┬────────┘
         │
         ├─→ [Events detected]
         │   (start_ms, stop_ms)
         │
         ├─→ [Load .pos file with --ppm]
         │   (if --pos-file and --ppm provided)
         │
         ├─→ [Compute speed2D]
         │   Uses (x, y, t) position data
         │
         ├─→ [Sample speed at events]
         │   Speed at midpoint of each event
         │
         └─→ [Save scores TSV]
             ┌──────────────┐
             │ ID#:         │
             │ Start Time:  │
             │ Stop Time:   │
             │ Settings:    │
             │ Speed(cm/s): │ ← NEW!
             └──────────────┘
```

## Backward Compatibility

✅ **Fully backward compatible:**
- Detection commands work exactly as before if --pos-file/--ppm not provided
- Scores files without speed column still work in metrics-batch/filter-scores
- If speed column missing, behavior-gating silently skips (with warning)

## Testing

### Test Case 1: Hilbert with Speed
```bash
python -m stlar hilbert-batch \
    -f tests/data/test_rec.eeg \
    --ppm 500 \
    --pos-file tests/data/test_rec.pos \
    -v
```
Expected: Speed column in output

### Test Case 2: Behavior Gating
```bash
python -m stlar metrics-batch \
    -f output/scores/test_rec_HIL.txt \
    --behavior-gating \
    --speed-min 1.0 \
    --speed-max 5.0 \
    -v
```
Expected: Filters events by speed, counts should decrease

### Test Case 3: Filter-scores
```bash
python -m stlar filter-scores \
    -f output/scores/test_rec_HIL.txt \
    --preset LEC \
    --behavior-gating \
    -v
```
Expected: Applies both region preset and speed gating

## Previously Known Issues - NOW FIXED ✓

1. ✓ metrics-batch couldn't apply behavior gating (no speed column)
2. ✓ filter-scores couldn't apply behavior gating (no speed column)  
3. ✓ PPM parameter was missing from detection CLI arguments
4. ✓ "Speed min and max" couldn't be used without speed data

## Parameters Reference

### Detection Commands
All detection batch commands (hilbert-batch, ste-batch, mni-batch, consensus-batch, dl-batch) now support:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--pos-file` | str | none | Path to .pos file for position/speed data |
| `--ppm` | int | none | Pixels per meter (e.g., 500, 600) |

Example:
```bash
python -m stlar COMMAND -f data.eeg --ppm 500 --pos-file data.pos
```

### metrics-batch
Existing parameters now work with speed column:

| Parameter | Type | Behavior |
|-----------|------|----------|
| `--behavior-gating` | flag | Filters events by speed if column exists |
| `--speed-min` | float | Override min speed (cm/s) |
| `--speed-max` | float | Override max speed (cm/s) |

### filter-scores
Same parameters as metrics-batch for behavior gating.

## Troubleshooting

### Speed column not appearing
1. Check --pos-file path is correct
2. Check --ppm value (typical: 500-600)
3. Verify .pos file contains valid position data
4. Check verbose output: `python -m stlar ... -v`

### Speed values look wrong
- Verify PPM value matches your camera setup
- Check position data units (pixels assumed)
- Inspect .pos file format/integrity

### Behavior gating not filtering
1. Check if speed column exists in scores file
2. Verify min/max speed values make sense
3. Use --verbose to see which events are filtered out

## Next Steps (Optional Enhancements)

1. Add --ppm and --pos-file support to metrics-batch/filter-scores for retroactive speed computation
2. Add speed visualization to GUI
3. Add speed statistics to metrics output (mean_speed, speed_std)
4. Add speed statistics to filtered scores output

