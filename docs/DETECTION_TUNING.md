# HFO Detection Parameter Tuning Guide

## Problem: Too Many Low-Quality Detections

If you're detecting many low-amplitude or noisy events (e.g., <200 µV peak-to-peak), **don't use absolute amplitude thresholds**. Instead, tighten the relative detection parameters.

---

## Recommended Approach: Adjust Relative Thresholds

### 1. **Hilbert Detection** - Increase Envelope Threshold

**Current default:** `threshold_sd = 3.5`  
**Stricter options:** `4.0`, `4.5`, or `5.0` SD

- **Effect:** Requires stronger envelope deflection relative to baseline
- **Trade-off:** Fewer false positives, but may miss subtle HFOs
- **When to use:** Noisy recordings, high baseline activity

**GUI:** HFO Detection window → Hilbert Parameters → "Threshold (SD)"  
**CLI:** `--hilbert-threshold-sd 4.5`

---

### 2. **STE/RMS Detection** - Increase Energy Threshold

**Current default:** `threshold = 2.5` (RMS)  
**Stricter options:** `3.0`, `3.5`, or `4.0`

- **Effect:** Requires higher RMS energy relative to baseline
- **Trade-off:** Reduces sensitivity to low-energy oscillations
- **When to use:** High background activity

**GUI:** HFO Detection window → STE Parameters → "Threshold (RMS)"  
**CLI:** `--ste-threshold 3.5`

---

### 3. **Peak Validation** - Increase Required Oscillations

**Current default:** `required_peaks = 4` @ `5.0 SD`  
**Stricter options:**
- Increase count: `6` or `8` peaks (requires more oscillations)
- Increase threshold: `6.0` or `7.0 SD` (peaks must be stronger)

- **Effect:** Ensures detected events have clear oscillatory content
- **Trade-off:** Rejects shorter or lower-amplitude bursts
- **When to use:** Prioritizing high-confidence detections

**GUI:** HFO Detection window → Detection Parameters → "Required Peaks" / "Peak Threshold SD"  
**CLI:** `--required-peaks 6 --required-peak-threshold-sd 6.0`

---

### 4. **MNI Detection** - Increase Percentile Threshold

**Current default:** `threshold_percentile = 98.0`  
**Stricter options:** `99.0`, `99.5`, or `99.9`

- **Effect:** Only top percentile of RMS windows trigger detection
- **Trade-off:** Reduces sensitivity significantly
- **When to use:** Very active recordings with frequent high-energy events

**GUI:** HFO Detection window → MNI Parameters → "Threshold Percentile"  
**CLI:** `--mni-percentile 99.5`

---

## Secondary Approach: Minimum Duration Filter

**Current default:** `min_duration = 10 ms`  
**Stricter option:** `15 ms`, `20 ms`, or `25 ms`

- **Effect:** Rejects short-duration events
- **Trade-off:** May miss fast ripples (physiologically valid at 10-15 ms)
- **When to use:** Targeting longer ripple-band events only

**GUI:** HFO Detection window → Parameters → "Min Duration (ms)"  
**CLI:** `--min-duration-ms 20`

---

## Optional: Post-Detection Amplitude Filter (QC Only)

**Location:** Region Preset Dialog → DL Export Options → "Min amplitude (µV)"  
**Default:** `0` (disabled)

- **Use case:** Quality control for known recording characteristics
- **Not recommended** as primary filter; relative thresholds are more robust
- **Example:** Set to `200` if you empirically know your system noise floor

---

## Example Stricter Parameter Sets

### Conservative (Fewer False Positives)
```bash
python -m stlar hilbert-batch -f recording.egf \
  --hilbert-threshold-sd 4.5 \
  --required-peaks 6 \
  --required-peak-threshold-sd 6.0 \
  --min-duration-ms 15
```

### Very Strict (High-Confidence Only)
```bash
python -m stlar consensus-batch -f recording.egf \
  --hilbert-threshold-sd 5.0 \
  --ste-threshold 3.5 \
  --mni-percentile 99.0 \
  --required-peaks 8 \
  --required-peak-threshold-sd 7.0
```

---

## Validation: Check Detection Quality

After adjusting parameters, verify improvements:

1. **Visual inspection:** Open Score GUI and review detected events
2. **Amplitude distribution:** Check metadata CSV for `peak_to_peak_uv` column
3. **Count sanity:** Compare before/after detection counts
4. **SNR estimation:** Compute event amplitude / baseline noise ratio

---

## When Absolute Amplitude Filtering Is Appropriate

Use the optional `min_amplitude_uv` filter **only** if:
- You have consistent recording hardware across sessions
- You've empirically determined a noise floor (e.g., <150 µV)
- You're doing cross-study comparisons with known amplitude ranges
- You're applying as **secondary QC**, not primary filtering

**Best practice:** Start with relative thresholds, validate results, then optionally apply amplitude QC.

---

## Further Reading

- **CONSENSUS_DETECTION.md**: Multi-detector voting strategies
- **TECHNICAL_REFERENCE.md**: Mathematical details of each detector
- **README.md**: Full parameter reference for CLI and GUI
