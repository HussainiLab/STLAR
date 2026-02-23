# Advanced CLI Guide

Back to quick guide: [README CLI Reference](../README.md#cli-reference)

This document contains advanced command usage, parameter details, and common workflows for STLAR CLI.

## Command Groups

- Detection: `hilbert-batch`, `ste-batch`, `mni-batch`, `consensus-batch`, `dl-batch`
- Analysis: `metrics-batch`, `filter-scores`, `batch-ssm`
- Deep Learning pipeline: `prepare-dl`, `train-dl`, `export-dl`

---

## Detection Methods

### Hilbert Detection

```bash
python -m stlar hilbert-batch -f <data_file_or_directory> [options]
```

Common tuning options:
- `--threshold-sd`
- `--min-freq`, `--max-freq`
- `--min-duration-ms`
- `--required-peaks`

Example:

```bash
python -m stlar hilbert-batch -f data/ --threshold-sd 3.0 --min-freq 80 --max-freq 250 -v
```

### STE Detection

```bash
python -m stlar ste-batch -f <data_file_or_directory> [options]
```

Example:

```bash
python -m stlar ste-batch -f data/ --threshold 3.0 --window-size 0.01 --overlap 0.5 -v
```

### MNI Detection

```bash
python -m stlar mni-batch -f <data_file_or_directory> [options]
```

Example:

```bash
python -m stlar mni-batch -f data/ --threshold-percentile 99.0 --baseline-window 10.0 -v
```

### Consensus Detection

```bash
python -m stlar consensus-batch -f <data_file_or_directory> [options]
```

Example:

```bash
python -m stlar consensus-batch -f data/ --voting-strategy majority --overlap-threshold-ms 10 -v
```

Further consensus details:
- [Consensus Quickstart](CONSENSUS_QUICKSTART.md)
- [Consensus Detection](CONSENSUS_DETECTION.md)

### Deep Learning Detection

```bash
python -m stlar dl-batch -f <data_file_or_directory> --model-path <model.pt|model.onnx> [options]
```

Example:

```bash
python -m stlar dl-batch -f data/ --model-path models/hfo_detector.pt --threshold 0.5 --batch-size 32 -v
```

Tip: use `--dump-probs` to inspect probability spread and identify poorly calibrated models.

---

## Metrics and Filtering

Back to quick guide: [README Metrics](../README.md#hfo-metrics--score-filtering)

### `metrics-batch`

```bash
python -m stlar metrics-batch -f <scores_file_or_directory> [options]
```

Examples:

```bash
python -m stlar metrics-batch -f HFOScores/ --duration-min 30 -v
python -m stlar metrics-batch -f HFOScores/ --preset Hippocampus --band ripple --behavior-gating --speed-max 4.0 --save-filtered -v
```

### `filter-scores`

```bash
python -m stlar filter-scores -f <scores_file_or_directory> [options]
```

Examples:

```bash
python -m stlar filter-scores -f HFOScores/session_HIL.txt --min-duration-ms 15 --max-duration-ms 120 -v
python -m stlar filter-scores -f HFOScores/ --band ripple --behavior-gating --speed-min 0.5 --speed-max 3.0 -v
```

Advanced preset and gating references:
- [Preset Gating Guide](PRESET_GATING_GUIDE.md)
- [Detection Tuning](DETECTION_TUNING.md)

---

## Spatial Mapping (`batch-ssm`)

Back to quick guide: [README Spatial Mapping](../README.md#spatial-mapping-batch-ssm)

```bash
python -m stlar batch-ssm <input_path> --ppm <pixels_per_meter> [options]
```

Examples:

```bash
python -m stlar batch-ssm recordings/rat01_day1.egf --ppm 595
python -m stlar batch-ssm recordings/ --ppm 595 --chunk-size 60 --export-binned-csvs --plot-trajectory
```

Useful options:
- `--chunk-size`
- `--speed-min`, `--speed-max`
- `--export-binned-jpgs`, `--export-binned-csvs`
- `--plot-trajectory`
- `--eoi-file`

---

## Common Advanced Workflows

### High-confidence consensus

```bash
python -m stlar consensus-batch -f data/ --voting-strategy strict --overlap-threshold-ms 5 --hilbert-threshold-sd 3.5 --ste-threshold 3.0 --mni-percentile 99.0 -v
```

### Detection + metrics pipeline

```bash
python -m stlar hilbert-batch -f data/ -o HFOScores/ -v
python -m stlar filter-scores -f HFOScores/ --min-duration-ms 15 --max-duration-ms 120 -v
python -m stlar metrics-batch -f HFOScores/ --duration-min 30 -v
```

### DL inference after training

```bash
python -m stlar dl-batch -f new_data/ --model-path models/model.pt --threshold 0.5 -o results/ -v
```

See advanced DL training details in [Advanced DL Training Guide](DL_TRAINING_ADVANCED.md).
