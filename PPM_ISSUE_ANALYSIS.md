# PPM (Pixels Per Meter) Parameter Gap in metrics-batch & filter-scores

## Issue Summary

The `metrics-batch` and `filter-scores` CLI commands lack the `--ppm` parameter needed for behavior gating, creating an inconsistency:

- **Detection commands** (hilbert-batch, ste-batch, consensus-batch, dl-batch): ✓ Have `--ppm` and `--pos-file` options
- **Metrics/Filtering commands**: ✗ Missing `--ppm` and `--pos-file` options

## Root Cause

The detection pipeline:
1. Loads `.pos` (position) files using `getpos()` with `custom_ppm` parameter
2. Calculates animal speed using `speed2D()` from (x,y,t) coordinates  
3. Uses speed thresholds to filter detections (rest: 0-5 cm/s, active: >5 cm/s)
4. **BUT does NOT save speed values to the output scores TSV file**

### Current Scores File Format
```
ID#:    Start Time(ms):    Stop Time(ms):    Settings File:
HIL1    42500              42750             /path/to/settings.json
HIL2    45200              45480             /path/to/settings.json
...
```

No speed column exists!

## Impact on Behavior Gating

In `metrics-batch` and `filter-scores`, when users specify `--behavior-gating`:

```bash
python -m stlar metrics-batch -f scores.txt --preset Hippocampus --behavior-gating --speed-min 0.5 --speed-max 5.0
```

The code looks for a **speed** column in the scores TSV:

```python
# From _apply_preset_and_gating() in cli.py line 1627
if behavior_gating:
    speed_col = None
    for col in df.columns:
        if 'speed' in col.lower():
            speed_col = col
            break
    if speed_col is None:
        print("Warning: behavior gating requested but no speed column found; skipping behavior gate")
```

**Result:** If the speed column doesn't exist, behavior gating silently fails with just a warning.

## Proposed Solutions

### Option A: Add Speed Column to Detection Output (Recommended)
Modify detection commands to save speed values at event times:

```
ID#:    Start Time(ms):    Stop Time(ms):    Settings File:                    Speed(cm/s)
HIL1    42500              42750             /path/to/settings.json            2.3
HIL2    45200              45480             /path/to/settings.json            0.8
```

**Advantages:**
- Speed data travels with scores files
- metrics-batch and filter-scores would work without any changes
- Consistent with behavior gating philosophy

**Implementation:** Modify `_save_scores()` function in cli.py (around line 335) to accept and include speed values.

### Option B: Add --ppm & --pos-file to metrics-batch and filter-scores
Allow recomputation of speed from position files:

```bash
python -m stlar metrics-batch \
    -f scores.txt \
    --ppm 500 \
    --pos-file recording.pos \
    --behavior-gating \
    --speed-min 0.5 \
    --speed-max 5.0
```

**Advantages:**
- No changes needed to existing scores files
- Maximum flexibility for retrospective analysis
- Can override position lookup with explicit --ppm

**Implementation:**
1. Add `--ppm`, `--pos-file` arguments to metrics_parser and filter_parser in build_parser()
2. Modify `run_metrics_batch()` and `run_filter_scores()` to compute speed if needed
3. Update `_apply_preset_and_gating()` to accept pre-computed speed array

### Option C: Hybrid Approach
- **For detection commands**: Save speed to scores file (Option A)
- **For metrics-batch/filter-scores**: Accept optional --ppm/--pos-file to override/validate (Option B)

## Speed Calculation Formula

From `speed2D()` in Tint_Matlab.py:

```
v[n] = sqrt((x[n+1] - x[n-1])² + (y[n+1] - y[n-1])²) / (t[n+1] - t[n-1])
```

Where (x,y) are in pixels and must be converted to cm using:
```
distance_cm = (distance_pixels / PPM) * 100
speed_cm_s = distance_cm / time_sec
```

## Current Workaround

Users must currently:
1. Run detection with `--ppm` and `--pos-file` (which only filters detections, doesn't save speed)
2. Cannot retroactively apply behavior gating in metrics-batch/filter-scores without speed column
3. If they want to gate HFOs by speed, they must create a custom speed column in the scores TSV manually

## Recommendation

**Implement Option A first** (add speed to detection output):
- Minimal effort to implement
- Solves the immediate need
- Then consider Option B for advanced use cases (speed override in metrics-batch)

As user noted: *"Doesn't lack of PPM affect speed min and max?"* — Yes! Speed thresholds (cm/s) cannot be applied without either:
1. Pre-computed speed values in the scores file (Option A), OR
2. The ability to calculate them from position data using PPM (Option B)

