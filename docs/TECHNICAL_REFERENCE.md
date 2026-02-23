# STLAR Technical Reference: Functions & Formulas

## Overview

This document provides a deep dive into the algorithms, formulas, and mathematical operations powering STLAR. This is for scientists, engineers, and developers who want to understand what's happening under the hood.

**Structure:** Signal Processing → HFO Detection → Spatial Analysis → Visualization

---

## Part 1: Signal Processing & Filtering

### 1.1 Bandpass Filtering (IIR Butterworth)

**File:** `hfoGUI/core/filtering.py`

**Purpose:** Isolate HFO frequency bands from raw LFP data.

**Formula:**

Butterworth IIR filter transfer function (2nd order):

$$H(z) = \frac{b_0 + b_1 z^{-1} + b_2 z^{-2}}{1 + a_1 z^{-1} + a_2 z^{-2}}$$

Digital difference equation:
$$y[n] = b_0 x[n] + b_1 x[n-1] + b_2 x[n-2] - a_1 y[n-1] - a_2 y[n-2]$$

**Parameters:**
- Order: 3 (cubic)
- Filter type: `butter`
- Passband ripple: 3 dB (Rp)
- Stopband attenuation: 60 dB (As)

**Python Implementation:**
```python
from scipy.signal import butter, sosfiltfilt

# Create filter coefficients
sos = butter(order=3, Wn=[min_freq, max_freq], btype='band', output='sos')

# Apply zero-phase filtering (forward-backward)
filtered_data = sosfiltfilt(sos, raw_data)
```

**Use Cases:**
- **Ripple detection:** 80-125 Hz (or 80-250 Hz)
- **Fast ripple detection:** 250-500 Hz
- **Theta analysis:** 6-12 Hz
- **Alpha analysis:** 8-12 Hz

---

### 1.2 Notch Filter (50/60 Hz Powerline Removal)

**File:** `hfoGUI/core/load_intan_rhd_format/intanutil/notch_filter.py`

**Purpose:** Remove AC powerline interference (50 Hz or 60 Hz).

**Formula (2nd-order IIR Notch):**

$$H(z) = \frac{b_0 + b_1 z^{-1} + b_2 z^{-2}}{a_0 + a_1 z^{-1} + a_2 z^{-2}}$$

Difference equation:
$$a_0 y[n] = b_0 x[n] + b_1 x[n-1] + b_2 x[n-2] - a_1 y[n-1] - a_2 y[n-2]$$

**Parameters:**
- Notch frequency: 60 Hz (US) or 50 Hz (EU)
- Bandwidth: Typically 2-5 Hz (narrow, sharp attenuation)

**Quality Factor (Q):**
$$Q = \frac{f_{notch}}{Bandwidth}$$

For 60 Hz with 2 Hz bandwidth: Q ≈ 30 (very selective)

---

### 1.3 FIR Hann Window Filter

**File:** `hfoGUI/core/Intan_to_Tint.py` → `fir_hann()`

**Purpose:** Anti-aliasing before downsampling.

**Formula:**

Windowed sinc filter:
$$h[n] = sinc(n) \cdot w[n]$$

where Hann window:
$$w[n] = 0.5 - 0.5 \cos\left(\frac{2\pi n}{N-1}\right)$$

Nyquist normalized cutoff:
$$\omega_c = \frac{f_{cutoff}}{f_{nyquist}} = \frac{2 f_{cutoff}}{f_s}$$

**Implementation:**
```python
from scipy.signal import firwin, lfilter

nyq_rate = Fs / 2
b = firwin(n_taps=101, cutoff=cutoff_freq/nyq_rate, window='hann')
filtered_data = lfilter(b, 1.0, data)
```

**Default:** 101 taps (better frequency response, ~5ms latency)

---

## Part 2: Hilbert Transform & Envelope Detection

### 2.1 Analytic Signal via Hilbert Transform

**File:** `hfoGUI/core/Score.py` → `hilbert_detect_events()`

**Purpose:** Compute amplitude envelope of bandpass-filtered signal.

**Mathematical Concept:**

The Hilbert transform creates an analytic signal:
$$z(t) = x(t) + i \hat{x}(t)$$

where:
- $x(t)$ = original (real) signal
- $\hat{x}(t)$ = Hilbert transform of $x(t)$ (imaginary component)

**Amplitude Envelope:**
$$A(t) = |z(t)| = \sqrt{x(t)^2 + \hat{x}(t)^2}$$

**Phase Information:**
$$\phi(t) = \arctan\left(\frac{\hat{x}(t)}{x(t)}\right)$$

**Python Implementation:**
```python
from scipy.signal import hilbert
import numpy as np

# Compute analytic signal
analytic_signal = hilbert(filtered_data)

# Extract envelope (amplitude)
envelope = np.abs(analytic_signal)

# Extract instantaneous phase
phase = np.angle(analytic_signal)

# Extract instantaneous frequency
inst_freq = np.diff(np.unwrap(phase)) / (2 * np.pi) * Fs
```

**Why Hilbert Transform?**
- Preserves causality (no lookahead)
- Efficient FFT-based implementation
- Phase-sensitive (useful for burst detection)

---

### 2.2 Envelope Threshold Detection

**File:** `hfoGUI/core/Score.py` → `hilbert_detect_events()`

**Purpose:** Identify times when envelope exceeds threshold.

**Formula:**

For each epoch window (default 300 sec):

$$\text{threshold} = \mu + k \cdot \sigma$$

where:
- $\mu$ = mean envelope amplitude in window
- $\sigma$ = standard deviation
- $k$ = threshold parameter (default: 3.0 SD)

**Rationale:** Assumes signal + noise is Gaussian; threshold at mean + 3σ captures ~0.3% of baseline noise as false positives.

**Detection Logic:**
```python
window_mean = np.mean(envelope[window_start:window_end])
window_std = np.std(envelope[window_start:window_end])
threshold = window_mean + num_sd * window_std

# Find samples above threshold
eoi_samples = np.where(envelope >= threshold)[0]
```

**Adaptive Windowing:** Each 300-second window recomputes mean/std to track non-stationary noise.

---

### 2.3 Event Boundary Refinement

**File:** `hfoGUI/core/Score.py` → `hilbert_detect_events()`

**Purpose:** Find exact start/stop times of HFO bursts within threshold crossings.

**Algorithm:**

1. **Find rough boundaries** where envelope crosses threshold
2. **Search backward** from crossing point to find local minimum
3. **Search forward** to find local minimum after peak

**Implementation:**
```python
boundary_percent = 0.3  # Default: 30% of threshold
boundary_threshold = mean + boundary_percent * (threshold - mean)

# Find start: search backward from crossing
for i in range(crossing_idx, -1, -1):
    if envelope[i] < boundary_threshold:
        start_idx = i
        break

# Find end: search forward from crossing
for i in range(crossing_idx, len(envelope)):
    if envelope[i] < boundary_threshold:
        end_idx = i
        break
```

**Time Conversion:**
$$t_{ms} = \frac{\text{sample\_index}}{F_s} \times 1000$$

---

### 2.4 Peak Detection within Events

**File:** `hfoGUI/core/Tint_Matlab.py` → `detect_peaks()`

**Purpose:** Verify HFO bursts contain high-frequency oscillations (not just noise).

**Algorithm:** Finds local maxima in rectified signal.

**Implementation:**
```python
def detect_peaks(x, threshold=0, edge='rising'):
    """Find peaks using first derivative."""
    dx = np.diff(x)
    
    # Rising edge: derivative goes from positive to non-positive
    rising = np.where((dx[:-1] > 0) & (dx[1:] <= 0))[0]
    
    # Apply height threshold
    peaks = rising[x[rising] >= threshold]
    
    return peaks
```

**Peak Requirement:** 
$$\text{\# peaks} \geq \text{required\_peak\_number}$$

Default: ≥ 6 peaks of threshold in event = at least 6 oscillations

**Peak Threshold:**
$$\text{peak\_height} \geq \mu + k \cdot \sigma$$

Default: peaks ≥ mean + 2 SD

---

### 2.5 Event Duration Filter

**File:** `hfoGUI/core/Score.py` → `hilbert_detect_events()`

**Purpose:** Reject noise spikes that are too brief to be HFOs.

**Formula:**
$$\text{duration} = t_{stop} - t_{start}$$

**Threshold:**
$$\text{duration} \geq \text{min\_duration\_ms}$$

Default: 10 ms (ensures at least 1-2 oscillations at 80-250 Hz)

---

## Part 3: Alternative HFO Detection Methods

### 3.1 STE (Short-Term Energy / RMS Detection)

**File:** `hfoGUI/core/Detector.py` → `_local_ste_rms_detect()`

**Purpose:** Detect HFOs via energy in sliding windows (faster than Hilbert).

**Formula:**

RMS (Root Mean Square) in window:
$$\text{RMS}_i = \sqrt{\frac{1}{N} \sum_{j=0}^{N-1} x[i \cdot \text{step} + j]^2}$$

where:
- $N$ = window size (samples)
- step = window stride

**Threshold:**
$$\text{RMS}_i \geq k \cdot \text{mean\_RMS}$$

or absolute value:
$$\text{RMS}_i \geq k_{abs}$$

**Window Parameters:**
- Default window: 10 ms
- Overlap: 50% (sliding window steps by 5 ms)

**Advantage:** O(N) vs. O(N log N) for Hilbert → 10x faster

---

### 3.2 MNI (Montreal Neurological Institute) Detection

**File:** `hfoGUI/core/Detector.py` → `_local_mni_detect()`

**Purpose:** Detect HFOs using baseline power statistics.

**Formula:**

1. **Compute baseline power** in frequency band
2. **Set threshold** at high percentile (99%) of baseline

$$\text{threshold} = \text{percentile}_{99}(\text{baseline\_power})$$

3. **Detection:** Power above threshold

**Baseline Window:** Default 10 seconds (sliding to adapt to non-stationary data)

**Advantage:** Minimal parameters; works with varying noise levels

---

### 3.3 Consensus Detection (Multi-Method Voting)

**File:** `hfoGUI/cli.py` → Consensus subcommand

**Purpose:** Combine Hilbert, STE, and MNI for robust detection.

**Voting Strategies:**

**Strict (3/3):**
$$\text{HFO if all 3 methods agree}$$

**Majority (2/3):**
$$\text{HFO if at least 2 methods agree}$$

**Lenient (1/3):**
$$\text{HFO if any method detects}$$

**Overlap Detection:**
Events must overlap in time within tolerance window (default 10 ms):
$$|\text{Method}_A \cap \text{Method}_B| \geq \text{overlap\_threshold}$$

---

### 3.4 Deep Learning Detection

**File:** `hfoGUI/core/Detector.py` → `_LocalDLDetector`, `dl_detect_events()`

**Purpose:** Classify HFO segments using trained neural network.

**Pipeline (1D models):**

1. **Preprocess:** Bandpass filter (80-500 Hz)
2. **Segment:** Sliding windows (default: 100 ms with 50% overlap)
3. **Normalize:** Z-score per window
4. **Model:** CNN/LSTM (trained on labeled HFO data)
5. **Threshold:** Probability ≥ threshold (default 0.5)

**Pipeline (2D CWT models):**

1. **Segment:** Sliding windows (default: 100 ms with 50% overlap)
2. **CWT Transform:** Convert 1D signal to 2D scalogram (64 scales)
3. **Model:** 2D CNN (trained on CWT scalograms)
4. **Threshold:** Probability ≥ threshold (default 0.5)

**CLI Usage:**
- 1D models: `dl-batch --model-path model.pt`
- 2D CWT models: `dl-batch --model-path cwt_model.pt --use-cwt --fs 4800`

**Input Features:**
- Time-domain: Raw signal in window (1D models)
- Time-frequency: CWT scalogram (2D models)
- Envelope: Hilbert envelope (legacy)

**Model Export:**
- PyTorch: `.pt` format
- ONNX: Platform-independent inference

---

## Part 4: Time-Frequency Analysis

### 4.1 Fast Fourier Transform (FFT)

**File:** `hfoGUI/core/filtering.py` → `FastFourier()`

**Purpose:** Convert time-domain signal to frequency domain.

**Formula:**
$$X[k] = \sum_{n=0}^{N-1} x[n] e^{-i 2\pi k n / N}$$

**Power Spectral Density (PSD):**
$$\text{PSD}[k] = \frac{2}{N F_s} |X[k]|^2$$

Factor of 2 accounts for positive frequencies only (negative are mirror).

**Implementation:**
```python
from scipy import fftpack
import numpy as np

N = len(signal)
freqs = np.fft.fftfreq(N, 1/Fs)[:N//2]  # Positive freqs only
fft_vals = fftpack.fft(signal)
psd = (2.0 / N) * np.abs(fft_vals[:N//2]) ** 2
```

**Resolution:** 
$$\Delta f = \frac{F_s}{N}$$

To get 0.5 Hz resolution at 1000 Hz sampling, need N ≥ 2000 samples ≈ 2 sec

---

### 4.2 Stockwell Transform (S-Transform)

**File:** `hfoGUI/core/TFA_Functions.py` → `strans()` & `stransform()`

**Purpose:** Time-frequency decomposition combining STFT + wavelets.

**Concept:** Continuous wavelet transform using Gaussian window, frequency-dependent width.

**Formula:**

$$S(t, f) = \int_{-\infty}^{\infty} x(\tau) w(f(\tau - t)) e^{-i 2\pi f \tau} d\tau$$

where Gaussian window:
$$w(t, f) = \frac{|f|}{\sqrt{2\pi}} e^{-\frac{(t f)^2}{2}}$$

**Key Properties:**
- **Higher frequencies:** Narrower time window (better time resolution)
- **Lower frequencies:** Wider time window (better frequency resolution)
- **Phase information:** Preserves phase (unlike spectrogram)
- **Zero DC:** Detrends automatically

**Computational Steps:**

1. **Detrending:** Remove polynomial trend
2. **Edge Tapering:** Hanning window 5% at edges
3. **FFT:** Convert to frequency domain
4. **Gaussian Localization:** Apply frequency-dependent Gaussian
5. **Inverse FFT:** Return to time domain

**Output:**
- 2D matrix: Time × Frequency
- Complex values: Magnitude = amplitude, Phase = phase

---

### 4.3 Spectrogram (STFT)

**File:** `hfoGUI/core/GraphSettings.py` (visualization)

**Purpose:** Time-frequency representation for visualization.

**Formula (Welch's Method):**

1. **Segment:** Divide signal into overlapping windows (e.g., 1 sec, 50% overlap)
2. **Window:** Apply taper (Hanning window)
3. **FFT:** Compute FFT per segment
4. **Average:** Average power across segments

$$\text{Spectrogram}(t, f) = \frac{1}{M} \sum_{m=0}^{M-1} |X_m(f)|^2$$

**Advantages over raw FFT:**
- Noise reduction via averaging
- Better frequency resolution (longer windows)
- Handles non-stationary signals

---

## Part 5: Spatial Analysis & Binning

### 5.1 Arena Detection & Coordinates

**File:** `spatial_mapper/src/initialize_fMap.py` → `detect_arena_shape()`

**Purpose:** Determine arena bounds from position tracking.

**Algorithm:**

```python
# Find spatial extent
x_min, x_max = np.percentile(pos_x, [0, 100])
y_min, y_max = np.percentile(pos_y, [0, 100])

# Add 5% margin
margin = 0.05 * max(x_max - x_min, y_max - y_min)
arena_bounds = [x_min - margin, x_max + margin, 
                y_min - margin, y_max + margin]

# Determine shape (circular vs. rectangular)
aspect_ratio = (x_max - x_min) / (y_max - y_min)
if 0.8 < aspect_ratio < 1.2:
    shape = "circular"
else:
    shape = "rectangular"
```

---

### 5.2 Cartesian Grid Binning

**File:** `spatial_mapper/src/initialize_fMap.py`

**Purpose:** Divide arena into 2D grid for spatial analysis.

**Bin Definition:**

$$\text{x\_bins} = \text{linspace}(x_{min}, x_{max}, 5)$$  (4 bins per axis)

$$\text{y\_bins} = \text{linspace}(y_{min}, y_{max}, 5)$$

This creates a **4×4 grid** (16 total bins)

**Bin Membership:**
$$\text{bin}(x, y) = \left(\text{digitize}(x, x\_bins), \text{digitize}(y, y\_bins)\right)$$

Using numpy:
```python
x_bin = np.digitize(pos_x, x_edges)
y_bin = np.digitize(pos_y, y_edges)
bin_id = (x_bin, y_bin)
```

---

### 5.3 Polar Binning (Circular Arena)

**File:** `spatial_mapper/src/initialize_fMap.py` → `compute_polar_binned_analysis()`

**Purpose:** Bin circular arena by radius and angle.

**Radial Bins:**
- Ring 0: r ∈ [0, r_max/√2)
- Ring 1: r ∈ [r_max/√2, r_max)
- Ring 2: r = r_max (boundary)

**Angular Bins:**
- 8 sectors: θ ∈ [-π, -3π/8), [-3π/8, -π/4), ..., [7π/8, π]

**Conversion (Cartesian → Polar):**
$$r = \sqrt{x^2 + y^2}$$
$$\theta = \arctan2(y, x)$$

**Implementation:**
```python
r = np.sqrt(pos_x**2 + pos_y**2)
theta = np.arctan2(pos_y, pos_x)

r_bin = np.digitize(r, [r_max/np.sqrt(2), r_max])
theta_bin = np.digitize(theta, [-np.pi, -3*np.pi/8, ..., 7*np.pi/8])
```

---

### 5.4 Occupancy Computation

**File:** `spatial_mapper/src/main.py` → `_compute_tracking_occupancy()`

**Purpose:** Calculate percentage of time spent in each bin.

**Formula:**

For each bin:
$$\text{Occupancy}_i = \frac{N_i}{N_{total}} \times 100\%$$

where:
- $N_i$ = samples in bin $i$
- $N_{total}$ = total samples

**Tracking-aware occupancy** (using chunked data):

```python
for bin_idx in range(num_bins):
    # Count position samples in bin
    samples_in_bin = np.sum((x_bin == bin_idx[0]) & (y_bin == bin_idx[1]))
    occupancy[bin_idx] = (samples_in_bin / total_samples) * 100
```

---

### 5.5 Power Spectral Analysis per Bin

**File:** `spatial_mapper/src/core/processors/spectral_functions.py`

**Purpose:** Compute frequency content in each spatial bin.

**Algorithm:**

1. **Extract data** in bin: LFP samples where position ∈ bin
2. **Compute FFT:** frequency × power
3. **Frequency bands:** θ (6-12 Hz), α (12-30 Hz), β (30-100 Hz), γ (100-250 Hz), HFO (250-500 Hz)
4. **Integrate power:** Sum across frequency band

**Power Integration:**
$$P_{band} = \sum_{f \in [f_{min}, f_{max}]} |X(f)|^2 \Delta f$$

**Normalization options:**
- Absolute: µV²/Hz
- Relative: % of total power
- Z-scored: (P - μ) / σ across bins

---

### 5.6 Dominant Frequency per Bin

**File:** `spatial_mapper/src/initialize_fMap.py`

**Purpose:** Identify strongest oscillation frequency in each bin.

**Formula:**
$$f_{dominant} = \arg\max_f |X(f)|^2$$

**Implementation:**
```python
psd_values = np.abs(fft_result) ** 2
peak_freq_idx = np.argmax(psd_values)
dominant_freq = frequencies[peak_freq_idx]
```

**Band Assignment:** Classify into θ, α, β, γ, HFO based on peak frequency

---

### 5.7 Heatmap Generation

**File:** `spatial_mapper/src/main.py` → Visualization

**Purpose:** Visualize spatial distribution of activity/power.

**Data Matrix:**
$$H[i, j] = \text{power or occupancy in bin}(i, j)$$

**Colormap:** Typically `plasma` or `viridis` (perceptually uniform)

**Interpolation:** Upsampled 10× for smooth appearance using `interp2d` or `ndimage.zoom`

```python
from scipy.ndimage import zoom
heatmap_upsampled = zoom(heatmap, 10, order=3)  # Cubic interpolation
```

---

## Part 6: Event Localization

### 6.1 HFO-Position Mapping

**File:** `spatial_mapper/src/main.py` → EOI (Event of Interest) processing

**Purpose:** Determine where in arena HFOs occur.

**Algorithm:**

1. **Load HFO times** from detection file (milliseconds)
2. **Find position** at each HFO time via interpolation
3. **Classify bin:** Which grid bin contains HFO position

**Time Matching:**
```python
# HFO time in ms, position time in ms
hfo_time_ms = 1234.5
pos_times_ms = tracking_times * 1000 / Fs

# Find closest position sample
pos_idx = np.argmin(np.abs(pos_times_ms - hfo_time_ms))
hfo_x, hfo_y = pos_x[pos_idx], pos_y[pos_idx]
```

**Position Interpolation (for higher accuracy):**
```python
from scipy.interpolate import interp1d

# Create interpolators
f_x = interp1d(pos_times_ms, pos_x, kind='linear', bounds_error=False)
f_y = interp1d(pos_times_ms, pos_y, kind='linear', bounds_error=False)

# Interpolate at HFO time
hfo_x = f_x(hfo_time_ms)
hfo_y = f_y(hfo_time_ms)
```

---

### 6.2 Event-Bin Association

**File:** `spatial_mapper/src/main.py`

**Purpose:** Count HFO events per spatial bin.

**Algorithm:**

```python
# For each HFO
eoi_counts = {bin_id: 0 for bin_id in all_bins}

for hfo_time in hfo_times:
    # Get position at HFO time
    pos_x, pos_y = interpolate_position(hfo_time)
    
    # Determine bin
    bin_x = np.digitize(pos_x, x_edges)
    bin_y = np.digitize(pos_y, y_edges)
    bin_id = (bin_x, bin_y)
    
    # Increment count (only if bin touched)
    if bin_id not in visited_bins[hfo_id]:
        eoi_counts[bin_id] += 1
        visited_bins[hfo_id].add(bin_id)
```

**Key Detail:** Count unique HFOs per bin, not position samples during HFOs

---

## Part 7: Statistical Analysis

### 7.1 Autocorrelation Function (ACF)

**File:** `hfoGUI/core/Tint_Matlab.py` (if used)

**Purpose:** Detect rhythmic organization of HFOs.

**Formula:**
$$\text{ACF}(k) = \frac{\sum_{t=0}^{N-k-1} (x_t - \bar{x})(x_{t+k} - \bar{x})}{\sum_{t=0}^{N-1} (x_t - \bar{x})^2}$$

where $k$ = lag in samples

**Interpretation:**
- ACF(k) = 1: Perfect periodicity
- ACF(k) = 0: No correlation
- ACF(k) < 0: Anticorrelation

---

### 7.2 Speed Calculation (for behavioral tracking)

**File:** `hfoGUI/core/Tint_Matlab.py` → `speed2D()`

**Purpose:** Compute instantaneous velocity from position tracking.

**Formula (central difference):**
$$v[n] = \frac{\sqrt{(x[n+1] - x[n-1])^2 + (y[n+1] - y[n-1])^2}}{t[n+1] - t[n-1]}$$

**Advantages:**
- Central difference (symmetric) reduces noise
- Handles variable sampling rate via time differences

**Implementation:**
```python
v = np.zeros(len(x))
for i in range(1, len(x)-1):
    dx = x[i+1] - x[i-1]
    dy = y[i+1] - y[i-1]
    dt = t[i+1] - t[i-1]
    v[i] = np.sqrt(dx**2 + dy**2) / dt

v[0] = v[1]  # Edge handling
v[-1] = v[-2]
```

**Units:** cm/s (if coordinates in cm)

---

### 7.3 Rate Map (Spike Rate per Spatial Bin)

**File:** `hfoGUI/core/Tint_Matlab.py` → `ratemap()`

**Purpose:** Visualize neural firing rate as function of position.

**Formula:**

$$\text{Rate}_{bin} = \frac{N_{spikes}}{T_{occupancy}}$$

where:
- $N_{spikes}$ = spike count in bin
- $T_{occupancy}$ = time animal spent in bin (seconds)

**Spatial Smoothing** (Gaussian kernel):
$$\text{Rate}_{smooth}(x,y) = \sum_{bins} \text{Rate}_{bin} \cdot G(x-x_{bin}, y-y_{bin}; \sigma)$$

where:
$$G(x, y; \sigma) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2 + y^2}{2\sigma^2}}$$

Kernel width σ typically 2-5 cm

---

## Part 8: Data I/O & Format Conversion

### 8.1 Intan RHD Format

**File:** `hfoGUI/core/load_intan_rhd_format/`

**Purpose:** Read binary electrophysiology data from Intan amplifier.

**File Structure:**
```
[Header] [Data Block 1] [Data Block 2] ... [Data Block N]
```

**Data Block Contents:**
- Amplifier samples (16-bit signed int)
- ADC samples
- Timing signals (TTL)
- Temperature sensor
- Supply voltage

**Bit Conversion (Intan → Volts):**
$$V = \text{raw\_value} \times \frac{195.3 \text{ µV}}{32768}$$

(Default gain: 1000× amplification at 20 µV/LSB)

---

### 8.2 Tint EEG/EGF Format

**File:** `hfoGUI/core/Intan_to_Tint.py`

**Purpose:** Export processed data to Tint (analysis software).

**File Structure:**
```
EEG Header + Raw Data (int16)
```

**Header (text):**
```
Fs (sampling rate in Hz)
nchannels
nbytes (total data bytes)
data_offset
date/time stamps
```

**Scaling Factor (Bits → Voltage):**

Via `.set` file:
```
Bits: [channel_number] [LSB_µV] [gain_factor]
```

Example: Channel 1: 0.194 µV/bit at 1000× = 194 µV/LSB

**Conversion:**
$$V = \text{LSB\_µV} \times \text{raw\_int16}$$

---

### 8.3 EOI (Event of Interest) Format

**File:** `spatial_mapper/src/main.py` → `processEOIs()`

**Purpose:** Load HFO event times and durations for spatial analysis.

**File Format (tab or comma-separated):**
```
StartTime  EndTime   Duration
1234.5     1245.67   11.17
2345.67    2356.78   11.11
...
```

**Units Detection Heuristic:**
- If max(start) > session_duration × 5 → Assume milliseconds → divide by 1000
- Otherwise → Assume seconds

**Robust Parser:**
- Skip header lines (if contain non-numeric)
- Handle mixed delimiters (tabs, commas, semicolons)
- Regex tokenization: `[\d\.]+` to extract numbers

---

## Part 9: Visualization

### 9.1 Matplotlib Figure Rendering

**File:** `spatial_mapper/src/main.py` (MplCanvas)

**Purpose:** Display plots in Qt GUI.

**Pipeline:**

1. **Create figure:** `fig = plt.figure(figsize=(w, h), dpi=100)`
2. **Add axes:** `ax = fig.add_subplot(111)`
3. **Plot data:** `ax.imshow()`, `ax.scatter()`, etc.
4. **Embed in Qt:** `FigureCanvas(fig)` → `addToLayout()`
5. **Refresh:** `canvas.draw()` on data change

**Memory Efficiency:**
- Clear old figure: `fig.clear()` before replot
- Limit points: Downsample for >100k points
- Use rasterization for large scatter plots

---

### 9.2 Tracking Overlay on Heatmap

**File:** `spatial_mapper/src/main.py`

**Purpose:** Show animal trajectory over spatial power map.

**Implementation:**

1. **Heatmap:** `imshow(power_grid, origin='lower', cmap='plasma')`
2. **Trajectory:** `scatter(x_positions, y_positions, c=time, cmap='winter')`
3. **Scale:** Convert physical coordinates (cm) to pixel coordinates

**Pixel Mapping:**
$$\text{pixel}_x = \frac{x - x_{min}}{x_{max} - x_{min}} \times \text{image\_width}$$

---

### 9.3 Binned Analysis Visualization

**File:** `spatial_mapper/src/main.py` → TFplots

**Purpose:** Show frequency band distribution per bin.

**Multi-panel Layout:**
- Each bin = subplot
- Y-axis: Power (µV²/Hz)
- X-axis: Frequency (Hz)
- Bar chart or line plot per frequency band

**Color Coding by Frequency Band:**
- θ: Blue
- α: Green
- β: Yellow
- γ: Orange
- HFO: Red

---

## Part 10: Performance & Optimization

### 10.1 Computational Complexity

| Operation | Complexity | Time (1 hour @ 30 kHz) |
|-----------|-----------|----------------------|
| IIR Bandpass | O(N) | <1 sec |
| FFT (N=2048) | O(N log N) | ~2 sec |
| Hilbert Transform | O(N log N) | ~5 sec |
| Spectrogram (1 sec windows) | O(N log N) | ~10 sec |
| S-Transform | O(N²) | ~60 sec |

### 10.2 Memory Usage

Signal types (1 hour @ 30 kHz = 108 M samples):
- Raw int16: 200 MB
- Float32: 400 MB
- Complex64: 800 MB
- Full spectrogram (1000 freq bins): 400 GB (impractical)

**Chunking Strategy:** Process 5-minute blocks, aggregate statistics

### 10.3 Parallelization

**Suitable for Parallelization:**
- Per-channel processing (independent)
- Per-chunk spectral analysis
- Directory batch processing (multiple files)

**Tools:**
- `multiprocessing.Pool` (CPU-bound)
- `concurrent.futures` (mixed I/O + CPU)
- GPU acceleration: CuPy, PyTorch for FFT

---

## Part 11: Validation & Quality Metrics

### 11.1 Detection Validation

**Sensitivity (Recall):**
$$\text{Sensitivity} = \frac{TP}{TP + FN}$$

TP = correctly detected HFOs, FN = missed HFOs

**Specificity:**
$$\text{Specificity} = \frac{TN}{TN + FP}$$

TN = correctly rejected noise, FP = false alarms

**F1-Score (Harmonic Mean):**
$$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

### 11.2 Spatial Stability

**Rayleigh Test:** Tests for non-uniformity in angular distribution.

**Spatial Coherence:** Cross-correlation of power maps between sessions.

---

## Appendix: Key Parameters & Defaults

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| Hilbert min freq | 80 Hz | 40-200 Hz | Adjust for ripple vs. fast-ripple |
| Hilbert max freq | 250 Hz | 200-500 Hz | 250 Hz = ripple; 500 Hz = fast-ripple |
| Threshold SD | 3.0 | 2.0-5.0 | Higher = fewer false positives |
| Min duration | 10 ms | 5-20 ms | Minimum oscillation cycles |
| Required peaks | 6 | 3-10 | Verify burst contains HFOs |
| Epoch window | 300 sec | 60-600 sec | Adaptive baseline update |
| Grid bins | 4×4 | 2×2 to 8×8 | Trade-off: spatial vs. statistical power |
| Chunk size (spatial) | 30 sec | 10-180 sec | Shorter = more temporal detail |
| FFT window (spectral) | 1 sec | 0.5-4 sec | Resolution vs. speed |

---

## References

**Key Papers:**
- Hilbert transform for HFO detection: Staba et al. (2002)
- Consensus detection: Burnos et al. (2016)
- S-Transform: Stockwell et al. (1996)
- Spatial rate mapping: O'Keefe & Burgess (1996)

**Software:**
- SciPy Signal Processing: https://docs.scipy.org/doc/scipy/reference/signal.html
- NumPy: https://numpy.org/doc/
- Matplotlib: https://matplotlib.org/

---

**Document Version:** 1.0  
**Last Updated:** December 29, 2025  
**For:** Scientists, Engineers, Algorithm Developers
