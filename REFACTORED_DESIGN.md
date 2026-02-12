# Refactored PPM/Speed Design: Analysis-Time Computation

## Summary of Changes

You were right to question the initial design. The code has been refactored to compute speed **at analysis time** (metrics-batch/filter-scores) rather than **at detection time**.

## Design Rationale

**Detection commands** (hilbert-batch, ste-batch, mni-batch, consensus-batch, dl-batch):
- ✅ Focus on detecting HFOs only
- ✅ No speed/PPM computation overhead
- ✅ Users don't need to know PPM unless they want behavior gating
- ✅ Faster execution

**Analysis commands** (metrics-batch, filter-scores):
- ✅ Speed only computed when needed (--behavior-gating flag)
- ✅ Optional --ppm and --pos-file arguments
- ✅ Supports both pre-computed speed (in scores file) and on-demand computation
- ✅ More flexible: can use different PPM values for different analyses

## API Changes

### Detection Commands (No Changes to Usage)
Detection commands work exactly as before - PPM/pos-file no longer needed:

```bash
python -m stlar hilbert-batch -f recording.eeg
python -m stlar ste-batch -f recording.eeg
python -m stlar mni-batch -f recording.eeg
python -m stlar consensus-batch -f recording.eeg
python -m stlar dl-batch -f recording.eeg --model-path model.pt
```

**Output:** Simple scores TSV with no speed column
```
ID#:    Start Time(ms):    Stop Time(ms):    Settings File:
HIL1    42500              42750             /path/to/settings.json
```

### Metrics-Batch: Now with Speed Option
New optional arguments when using --behavior-gating:

```bash
# Simple: no behavior gating (as before)
python -m stlar metrics-batch -f scores.txt --duration-min 30

# With speed column already in scores file
python -m stlar metrics-batch -f scores.txt \
    --behavior-gating \
    --speed-min 0.5 \
    --speed-max 5.0

# Computing speed from position file (NEW)
python -m stlar metrics-batch -f scores.txt \
    --behavior-gating \
    --ppm 500 \
    --pos-file recording.pos \
    --speed-min 0.5 \
    --speed-max 5.0
```

### Filter-Scores: Now with Speed Option
Same pattern:

```bash
# Simple: no behavior gating
python -m stlar filter-scores -f scores.txt --min-duration-ms 15

# With speed column already in scores file
python -m stlar filter-scores -f scores.txt \
    --behavior-gating \
    --speed-max 4.0

# Computing speed from position file (NEW)
python -m stlar filter-scores -f scores.txt \
    --behavior-gating \
    --ppm 500 \
    --pos-file recording.pos \
    --speed-min 1.0 \
    --speed-max 4.0
```

## Code Changes

### Modified Functions

#### `run_metrics_batch()` (lines ~1815)
Added before _apply_preset_and_gating():
```python
# Compute speed if behavior gating is enabled and we have position data
if behavior_flag and hasattr(args, 'ppm') and args.ppm and hasattr(args, 'pos_file') and args.pos_file:
    pos_file = Path(args.pos_file).expanduser()
    if 'Speed(cm/s):' not in df.columns:
        speed_values = _compute_speed_at_events(
            np.column_stack((starts, stops)),
            pos_file,
            args.ppm,
            fs_eeg=50,
            verbose=args.verbose
        )
        if speed_values is not None:
            df['Speed(cm/s):'] = speed_values
```

#### `run_filter_scores()` (lines ~1925)
Same pattern: adds speed column to df before _apply_preset_and_gating()

#### `build_parser()` (metrics & filter sections)
- ✅ Removed --ppm, --pos-file from all 5 detection parsers
- ✅ Added --ppm, --pos-file to metrics_parser
- ✅ Added --ppm, --pos-file to filter_parser
- ✅ Updated help text for --behavior-gating to mention .pos file support

### Preserved Functions

- `_compute_speed_at_events()` - unchanged, still available for use
- `_save_results()` - reverted to simple version (no speed_values parameter)
- `_apply_preset_and_gating()` - unchanged

## Behavior Summary

### Scenario 1: Quick detection (no behavior gating)
```bash
python -m stlar hilbert-batch -f recording.eeg -v
```
✅ Fast, simple, no PPM needed

### Scenario 2: Behavior gating with pre-computed speed
If a previous run saved speed column:
```bash
python -m stlar metrics-batch -f scores_with_speed.txt --behavior-gating -v
```
✅ Uses existing speed column

### Scenario 3: Behavior gating with on-demand speed computation
```bash
python -m stlar metrics-batch -f scores.txt \
    --behavior-gating --ppm 500 --pos-file recording.pos -v
```
✅ Computes speed on-the-fly only when needed

### Scenario 4: Retrospective analysis with different PPM
```bash
# First analysis with PPM=500
python -m stlar metrics-batch -f scores.txt \
    --behavior-gating --ppm 500 --pos-file recording.pos -o results_500/

# Second analysis with PPM=600
python -m stlar metrics-batch -f scores.txt \
    --behavior-gating --ppm 600 --pos-file recording.pos -o results_600/
```
✅ Very flexible, no need to re-detect

## Benefits of This Design

1. **Cleaner separation of concerns**
   - Detection = find HFOs
   - Analysis = apply gating criteria

2. **Simpler detection pipeline**
   - No optional position file logic
   - Faster (no speed computation)
   - Users don't need PPM unless doing behavior gating

3. **More flexible analysis**
   - Can use different PPM values
   - Can apply gating retrospectively
   - Supports both pre-computed and on-demand speed

4. **Better user experience**
   - Detection command is straightforward
   - PPM only needed when explicitly enabling behavior gating
   - Clear: `--behavior-gating` implies you can add `--ppm --pos-file`

## Backward Compatibility

✅ Fully backward compatible:
- Old scores files (without speed column) still work
- Behavior gating detects missing speed and warns/skips gracefully
- Detection commands unchanged from user perspective (just simpler now)

## Files Modified

- **hfoGUI/cli.py**
  - Removed speed computation from 5 detection processors
  - Removed --ppm/--pos-file from 5 detection parsers
  - Added --ppm/--pos-file to metrics and filter parsers
  - Added speed computation to run_metrics_batch()
  - Added speed computation to run_filter_scores()
  - Reverted _save_results() to simple format

