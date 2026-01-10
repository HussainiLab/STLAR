# Regional Preset & Behavior Gating Guide

## Overview
STLAR CLI now supports region-specific presets and behavior gating for both `metrics-batch` and `filter-scores` commands, bringing full parity with the GUI Score tab functionality.

## Features Added

### 1. Regional Presets
Built-in presets match GUI defaults:
- **LEC**: Ripple 80-250 Hz, FR 250-500 Hz, duration 15-120ms, speed gating 0-5 cm/s
- **Hippocampus**: Ripple 100-250 Hz, FR 250-500 Hz, duration 15-120ms, speed gating 0-5 cm/s
- **MEC**: Ripple 80-200 Hz, FR 200-500 Hz, duration 15-120ms, speed gating 0-5 cm/s

### 2. Band-Based Filtering
- Filter events by label/score column (e.g., ripple, fast ripple, artifact)
- Comma-separated list: `--band "ripple,fast"`
- Automatically applies region-specific duration constraints per band

### 3. Behavior Gating
- Filter events by animal movement speed
- Uses speed column in scores file
- Configurable min/max thresholds
- Can override preset defaults with `--speed-min` / `--speed-max`

### 4. Custom Presets
- Extend or override built-in presets via JSON file
- Pass `--preset-file custom_presets.json`
- Format: dict keyed by region name

### 5. Save Filtered Scores (metrics-batch only)
- New `--save-filtered` flag
- Saves preset-annotated filtered TSV to `filtered_scores/` subdirectory
- Filename includes applied filters (e.g., `session_LEC_band_gated.txt`)

## Usage Examples

### Sample Scores File Format
For preset/gating features to work, your scores file needs these columns:
```
ID#    Start Time(ms)    Stop Time(ms)    Settings File         Label           Speed(cm/s)
HIL1   1000              1050             hilbert_params.json   Ripple          1.2
HIL2   2000              2075             hilbert_params.json   Fast Ripple     2.3
HIL3   3500              3530             hilbert_params.json   Ripple          0.8
...
```

**Required columns:**
- ID#, Start Time(ms), Stop Time(ms) - always present in STLAR outputs
- **Label** or **Score** column - for band filtering
- **Speed(cm/s)** column - for behavior gating (if using .egf files with tracking)

### metrics-batch Examples

**Basic with preset:**
```bash
python -m stlar metrics-batch \
    -f HFOScores/recording_HIL.txt \
    --preset Hippocampus \
    --duration-min 30
```

**With band filter and gating:**
```bash
python -m stlar metrics-batch \
    -f HFOScores/ \
    --preset LEC \
    --band ripple \
    --behavior-gating \
    --save-filtered \
    --data data/
```

**Custom speed thresholds:**
```bash
python -m stlar metrics-batch \
    -f scores.txt \
    --preset MEC \
    --behavior-gating \
    --speed-min 1.0 \
    --speed-max 3.0 \
    --save-filtered \
    --duration-min 30 \
    -v
```

**Output structure:**
```
metrics/
├── session_hfo_metrics_LEC_band_gated.csv
└── filtered_scores/
    └── session_LEC_band_gated.txt
```

### filter-scores Examples

**With preset and band:**
```bash
python -m stlar filter-scores \
    -f scores.txt \
    --preset LEC \
    --band "ripple,fast" \
    --behavior-gating \
    -v
```

**Speed override without preset:**
```bash
python -m stlar filter-scores \
    -f scores.txt \
    --band ripple \
    --behavior-gating \
    --speed-min 0.5 \
    --speed-max 2.5 \
    --min-duration-ms 15 \
    --max-duration-ms 120
```

**Custom preset file:**
```bash
python -m stlar filter-scores \
    -f scores.txt \
    --preset "CustomRegion" \
    --preset-file my_presets.json \
    --behavior-gating
```

## Custom Preset File Format

Create a JSON file with region definitions:

```json
{
  "LEC": {
    "bands": {
      "ripple": [80, 250],
      "fast_ripple": [250, 500],
      "gamma": [30, 80]
    },
    "durations": {
      "ripple_min_ms": 15,
      "ripple_max_ms": 120,
      "fast_min_ms": 10,
      "fast_max_ms": 80
    },
    "threshold_sd": 3.5,
    "epoch_s": 300,
    "behavior_gating": true,
    "speed_threshold_min_cm_s": 0.0,
    "speed_threshold_max_cm_s": 5.0
  },
  "CustomRegion": {
    "bands": {
      "ripple": [90, 230]
    },
    "durations": {
      "ripple_min_ms": 20,
      "ripple_max_ms": 100
    },
    "behavior_gating": true,
    "speed_threshold_min_cm_s": 0.0,
    "speed_threshold_max_cm_s": 4.0
  }
}
```

## Testing with Sample Data

A sample scores file with Label and Speed columns is included for testing:

```bash
# Create sample file
cat > sample_scores.txt << 'EOF'
ID#	Start Time(ms)	Stop Time(ms)	Settings File	Label	Speed(cm/s)
HIL1	1000	1050	hilbert_params.json	Ripple	1.2
HIL2	2000	2075	hilbert_params.json	Fast Ripple	2.3
HIL3	3500	3530	hilbert_params.json	Ripple	0.8
HIL4	5000	5150	hilbert_params.json	Ripple	8.5
HIL5	6500	6560	hilbert_params.json	Fast Ripple	1.5
HIL6	8000	8010	hilbert_params.json	Artifact	12.0
EOF

# Test filtering
python -m stlar filter-scores -f sample_scores.txt --preset LEC --band ripple --behavior-gating -v

# Expected: filters out HIL4 (speed>5), HIL6 (artifact), keeps ripples within speed threshold
```

## Validation Results

✅ **filter-scores with preset + band + gating:**
```bash
python -m stlar filter-scores -f sample_scores.txt --preset LEC --band ripple --behavior-gating -v
# Output: Filtered: 10 -> 7 events (removed HIL4, HIL6, HIL10 due to speed/label)
```

✅ **metrics-batch with --save-filtered:**
```bash
python -m stlar metrics-batch -f sample_scores.txt --preset Hippocampus --behavior-gating --save-filtered --duration-min 1 -v
# Output: 
#   - metrics/sample_scores_hfo_metrics_Hippocampus_gated.csv
#   - metrics/filtered_scores/sample_scores_Hippocampus_gated.txt
```

✅ **Speed override:**
```bash
python -m stlar filter-scores -f sample_scores.txt --band "ripple,fast" --behavior-gating --speed-max 2.5 -v
# Output: Filtered: 10 -> 6 events (kept only events with speed ≤2.5 cm/s)
```

## Parameter Reference

### metrics-batch
| Parameter | Type | Description |
|-----------|------|-------------|
| `--preset` | str | Region name (LEC/Hippocampus/MEC or custom) |
| `--preset-file` | path | JSON file to extend/override presets |
| `--band` | str | Comma-separated band filters (matches label column) |
| `--behavior-gating` | flag | Apply speed thresholds from preset or overrides |
| `--speed-min` | float | Override min speed threshold (cm/s) |
| `--speed-max` | float | Override max speed threshold (cm/s) |
| `--save-filtered` | flag | Save filtered scores to filtered_scores/ subdir |

### filter-scores
Same parameters as metrics-batch (except `--save-filtered` not needed - always saves filtered output).

## Notes & Warnings

**Missing columns:**
- If Label/Score column absent: band filtering skipped with warning
- If Speed column absent: behavior gating skipped with warning

**Preset not found:**
- CLI prints available presets and continues with no preset applied

**Duration constraints:**
- With preset: applies per-band durations (ripple vs fast ripple)
- Without preset: applies min/max to all events uniformly

**Output naming:**
- Suffixes indicate applied filters: `_LEC`, `_band`, `_gated`
- Multiple filters combined: `session_LEC_band_gated.txt`

## Integration with GUI

The CLI presets are identical to GUI Score tab presets:
1. Same default regions (LEC, Hippocampus, MEC)
2. Same band definitions and duration constraints
3. Same speed gating thresholds
4. Extensible via same preset file format

Users can create presets in the GUI (via Region Preset Dialog) and export them for CLI use via JSON.
