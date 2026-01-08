# CLI Enhancement Implementation Summary

## Overview
Successfully implemented two new CLI subcommands to STLAR for batch metrics export and score filtering, bringing parity with GUI capabilities.

## New Subcommands Implemented

### 1. `metrics-batch` - Compute HFO Metrics from Scores Files
**Purpose:** Batch compute HFO metrics (event count, rate, durations, statistics) from existing score files

**Command:**
```bash
python -m stlar metrics-batch -f <scores_file_or_directory> [options]
```

**Features:**
- Load scores from .txt files (any detection method: HIL, STE, MNI, CON, DL)
- Compute 8 key metrics:
  - `total_events`: Event count
  - `recording_duration_minutes`: Recording length
  - `event_rate_per_min`: Events per minute
  - `mean_duration_ms`, `median_duration_ms`: Average durations
  - `min_duration_ms`, `max_duration_ms`: Duration extremes
  - `std_duration_ms`: Duration standard deviation
- Auto-detect recording duration from data files (.eeg/.egf)
- Fallback to manual duration specification (--duration-min)
- Save metrics as CSV for analysis

**Example:**
```bash
# Single file with manual duration
python -m stlar metrics-batch -f scores.txt --duration-min 30 -v

# Directory batch with auto-detect
python -m stlar metrics-batch -f HFOScores/ --data data/ -o results/
```

### 2. `filter-scores` - Filter Scores by Duration
**Purpose:** Clean score files by removing events outside duration range (noise/artifacts)

**Command:**
```bash
python -m stlar filter-scores -f <scores_file> [options]
```

**Features:**
- Load scores from .txt files
- Apply duration constraints:
  - `--min-duration-ms`: Remove bursts below threshold
  - `--max-duration-ms`: Remove long artifacts above threshold
- Preserve all original columns and format
- Save filtered scores in same TSV format

**Example:**
```bash
# Remove short noise (<15ms) and long artifacts (>150ms)
python -m stlar filter-scores -f scores.txt --min-duration-ms 15 --max-duration-ms 150 -v

# Keep only ripple range (15-120ms)
python -m stlar filter-scores -f scores.txt --min-duration-ms 15 --max-duration-ms 120
```

## Implementation Details

### Files Modified

1. **hfoGUI/cli.py** (1649 lines → expanded)
   - Added 5 new functions:
     - `_load_scores_from_txt()`: Parse TSV score files with flexible column naming
     - `_compute_hfo_metrics()`: Calculate statistical metrics from event times
     - `_infer_data_duration()`: Auto-detect recording duration from data files
     - `run_metrics_batch()`: Main metrics batch processor
     - `run_filter_scores()`: Main score filtering processor
   - Updated `build_parser()`: Added 2 new subcommand parsers
   - Updated `__all__` export list: Added new functions
   - Updated `main()`: Added command routing for new subcommands

2. **stlar/__main__.py** (updated)
   - Added `metrics-batch` and `filter-scores` to subparsers
   - Updated command routing to dispatch to new CLI functions

3. **README.md** (2315 lines)
   - Added comprehensive "HFO Metrics & Score Filtering" section
   - Added metrics-batch and filter-scores command documentation
   - Updated Table of Contents with new command links
   - Updated "Supported commands" overview in CLI Reference section

## Testing Results

✅ **CLI Help Verification**
```
python -m stlar --help
→ Shows both new commands in available commands list
```

✅ **Metrics-Batch Test**
```
python -m stlar metrics-batch -f test_scores_sample.txt --duration-min 30 -v
→ Processing: test_scores_sample.txt
→ Events: 8
→ Rate: 0.27 events/min
→ Duration: 69.38 ± 40.50 ms
→ Saved metrics → metrics/test_scores_hfo_metrics.csv
```

✅ **Filter-Scores Test**
```
python -m stlar filter-scores -f test_scores_sample.txt --min-duration-ms 50 --max-duration-ms 100 -v
→ Filtered: 8 → 5 events
→ Saved → test_scores_sample_filtered.txt
```

✅ **Output Validation**
- Metrics CSV: Correct format with 8 computed metrics
- Filtered TSV: Preserved original format, correctly filtered events

## Design Rationale

### Why These Features?
1. **Gap Analysis**: GUI has `exportHFOMetrics()` but CLI lacked batch equivalent
2. **Common Workflow**: Users often generate scores with one method then need to:
   - Compute summary statistics → `metrics-batch` solves this
   - Remove artifacts/noise → `filter-scores` solves this
3. **Reproducibility**: Enable headless processing pipelines without GUI

### Architecture Pattern
- Follows existing STLAR CLI design:
  - Batch processor pattern (like `run_hilbert_batch()`)
  - Single file or directory processing
  - Verbose logging with `-v` flag
  - Consistent output directory handling

### Reuse from GUI
- Based on `hfoGUI/core/Score.py::exportHFOMetrics()` logic
- Uses `ReadEEG()` for data loading (existing infrastructure)
- Leverages `Tint_Matlab.py::ReadEEG()` for duration inference

## Features Not Yet Implemented (Future Enhancement)
- Brain region presets (would require `RegionPresetDialog` port)
- Behavioral state breakdown (would require motion data alignment)
- Co-occurrence metrics (would require HFO_Classifier integration)
- DL export filtering (would require model preprocessing)

These are noted in code comments as "Future Enhancement" for next iteration.

## Documentation
- **CLI Help**: Available via `python -m stlar metrics-batch --help` and `python -m stlar filter-scores --help`
- **README Section**: Comprehensive examples and parameter documentation
- **Inline Code Comments**: Docstrings and comments explain logic

## Backward Compatibility
✅ All existing commands remain unchanged
✅ New subcommands are additive-only
✅ No breaking changes to CLI or GUI

## Performance Notes
- **metrics-batch**: Linear O(n) in number of score files
- **filter-scores**: Linear O(n) in number of events
- Both use NumPy for fast array operations

---

**Total Time to Implement:** ~2 hours
**Files Modified:** 3 (hfoGUI/cli.py, stlar/__main__.py, README.md)
**Lines Added:** ~200 (functions) + ~300 (documentation)
**Test Coverage:** 2/2 features tested successfully
