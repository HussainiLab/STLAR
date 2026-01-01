import numpy as np
from scipy.signal import butter, sosfiltfilt
from dataclasses import dataclass
from pathlib import Path
import threading
import os

# All detection methods now use local implementations
# No external dependencies required


def _check_package():
    """Ensure torch is available for DL detection; raise a clear error if not."""
    try:
        import torch  # noqa: F401
    except ImportError as e:
        raise RuntimeError("PyTorch is required for deep learning detection. Install with `pip install torch`." ) from e


@dataclass
class ParamDL:
    sample_freq: float
    model_path: str
    threshold: float = 0.5
    batch_size: int = 32


class _LocalDLDetector:
    """Lightweight DL detector wrapper with TorchScript fallback and RMS backup."""

    def __init__(self, params: ParamDL, progress_callback=None):
        self.params = params
        self.model = None
        self.device = 'cpu'
        self.window_secs = 1.0  # 1-second windows by default
        self.hop_frac = 0.5      # 50% overlap
        self.progress_callback = progress_callback
        self._load_model()

    def _progress(self, msg):
        """Send progress message to callback and print to console."""
        print(msg)
        if self.progress_callback:
            try:
                self.progress_callback(msg)
            except Exception as e:
                print(f"Progress callback error: {e}")

    def _load_model(self):
        path = Path(self.params.model_path)
        if not path.exists():
            self._progress(f"[DL Detection] Model file not found: {path}")
            self.model = None
            return
        
        file_size = path.stat().st_size / (1024*1024)
        self._progress(f"[DL Detection] Loading model ({file_size:.1f} MB)...")
        
        try:
            import torch
            torch.set_grad_enabled(False)
            # Try TorchScript first
            self._progress("[DL Detection] Attempting torch.jit.load...")
            try:
                self.model = torch.jit.load(str(path), map_location=self.device)
                self._progress("[DL Detection] Successfully loaded as TorchScript")
            except Exception as e:
                # Fallback to regular torch.load (state_dict or full model)
                self._progress(f"[DL Detection] TorchScript failed ({type(e).__name__}), trying torch.load...")
                try:
                    obj = torch.load(str(path), map_location=self.device)
                    if hasattr(obj, 'state_dict'):
                        # Attempt to rebuild simple model if state_dict
                        self._progress("[DL Detection] Loading state_dict...")
                        from hfoGUI.dl_training.model import build_model  # local import
                        mdl = build_model()
                        mdl.load_state_dict(obj['model_state'] if isinstance(obj, dict) and 'model_state' in obj else obj)
                        self.model = mdl
                        self._progress("[DL Detection] Model reconstructed from state_dict")
                    else:
                        self.model = obj
                        self._progress("[DL Detection] Model loaded with torch.load")
                except Exception as e2:
                    self._progress(f"[DL Detection] torch.load failed ({type(e2).__name__})")
                    self.model = None
            if self.model is not None:
                self.model.eval()
                self._progress("[DL Detection] Model ready for inference")
        except Exception as e:
            self._progress(f"[DL Detection] Unexpected error: {type(e).__name__}")
            self.model = None

    def detect(self, signal, ch_name='chn1'):
        # If no model, fall back to RMS detector
        if self.model is None:
            events_ms = _local_ste_rms_detect(signal, self.params.sample_freq)
            # Convert ms to seconds for downstream converter
            return [{'start': s/1000.0, 'end': e/1000.0} for s, e in events_ms]

        import torch
        x = np.asarray(signal, dtype=np.float32)
        fs = float(self.params.sample_freq)
        win = max(1, int(self.window_secs * fs))
        hop = max(1, int(win * self.hop_frac))

        pos_windows = []
        prob_values = []
        for start in range(0, len(x), hop):
            end = min(len(x), start + win)
            seg = x[start:end]
            if seg.size == 0:
                continue
            mu = seg.mean()
            sd = seg.std() + 1e-8
            seg = (seg - mu) / sd
            seg = torch.from_numpy(seg).unsqueeze(0).unsqueeze(0)  # (1,1,L)
            with torch.no_grad():
                logit = self.model(seg)
                if isinstance(logit, (list, tuple)):
                    logit = logit[0]
                prob = torch.sigmoid(logit.squeeze()).item()
                prob_values.append(prob)
            if prob >= float(self.params.threshold):
                pos_windows.append((start, end))

        # Debug: show probability distribution
        if prob_values:
            prob_min = min(prob_values)
            prob_max = max(prob_values)
            prob_mean = sum(prob_values) / len(prob_values)
            self._progress(f"[DL Detection] Probability distribution - min: {prob_min:.4f}, max: {prob_max:.4f}, mean: {prob_mean:.4f}")
            self._progress(f"[DL Detection] Threshold: {float(self.params.threshold):.4f}")
        self._progress(f"[DL Detection] Found {len(pos_windows)} positive windows out of {len(prob_values)} total windows")
        
        # Merge overlapping/nearby windows (within 200ms gap)
        # This prevents merging distant positive detections into one event
        max_gap_samples = int(0.2 * fs)  # 200ms gap threshold
        merged = []
        for start, end in pos_windows:
            if not merged:
                merged.append([start, end])
            elif start <= merged[-1][1] + max_gap_samples:
                # Within gap threshold, merge by extending the end
                merged[-1][1] = max(merged[-1][1], end)
            else:
                # Gap too large, start new event
                merged.append([start, end])

        self._progress(f"[DL Detection] After merging: {len(merged)} events (threshold=0.2s gap)")
        for i, (start, end) in enumerate(merged):
            duration_sec = (end - start) / float(fs)
            self._progress(f"[DL Detection]   Event {i+1}: {start//int(fs)}s - {end//int(fs)}s (duration: {duration_sec:.3f}s)")

        return [{'start': s/float(fs), 'end': e/float(fs)} for s, e in merged]


def set_DL_detector(params: ParamDL, progress_callback=None):
    return _LocalDLDetector(params, progress_callback=progress_callback)


def _convert_pyhfo_results_to_eois(hfos, Fs):
    """
    Attempt to convert pyHFO results into Nx2 array of [start_ms, stop_ms].
    This is a copy of the helper function from Score.py.
    """
    # Case 1.5: tuple like (array, channel_name)
    if isinstance(hfos, tuple) and len(hfos) >= 1:
        hfos = hfos[0]

    # Case 1: list of dicts with seconds
    if isinstance(hfos, (list, tuple)) and len(hfos) and isinstance(hfos[0], dict):
        start_keys = ['start', 'start_time', 't_start', 'onset']
        stop_keys = ['end', 'stop_time', 't_end', 'offset']
        rows = []
        for ev in hfos:
            s = None
            e = None
            for k in start_keys:
                if k in ev:
                    s = ev[k]
                    break
            for k in stop_keys:
                if k in ev:
                    e = ev[k]
                    break
            if s is None or e is None:
                continue
            # Many libs report seconds; if clearly too small, assume seconds
            if max(abs(s), abs(e)) < 1e6:  # not already in ms
                s_ms = float(s) * 1000.0
                e_ms = float(e) * 1000.0
            else:
                s_ms = float(s)
                e_ms = float(e)
            rows.append([s_ms, e_ms])
        if rows:
            return np.asarray(rows, dtype=float)

    # Case 2: dict with numpy arrays in samples
    if isinstance(hfos, dict):
        s = None
        e = None
        for k in ['start_samples', 'starts', 'start_idx']:
            if k in hfos:
                s = hfos[k]
                break
        for k in ['end_samples', 'ends', 'stop_idx']:
            if k in hfos:
                e = hfos[k]
                break
        if s is not None and e is not None:
            s = np.asarray(s)
            e = np.asarray(e)
            ms = 1000.0 * s / float(Fs)
            me = 1000.0 * e / float(Fs)
            return np.column_stack([ms, me]).astype(float)

    # Case 3: Nx2 numpy array in seconds or samples
    try:
        arr = np.asarray(hfos)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            first_two = arr[:, :2]
            # Heuristic: if values look like small (<1e6), could be seconds; if integers and large, samples
            if np.issubdtype(first_two.dtype, np.integer):
                ms = 1000.0 * first_two[:, 0] / float(Fs)
                me = 1000.0 * first_two[:, 1] / float(Fs)
                return np.column_stack([ms, me]).astype(float)
            else:
                # assume seconds
                return (first_two.astype(float) * 1000.0)
    except Exception:
        pass

    return np.asarray([])


def _local_ste_rms_detect(data, fs, threshold=3.0, window_size=0.01, overlap=0.5, min_freq=80.0, max_freq=500.0):
    """
    Local STE/RMS detector using scipy signal processing.
    - Bandpass filters to [min_freq, max_freq]
    - Computes windowed RMS with given window size and overlap
    - Flags windows exceeding mean + threshold * std of RMS
    - Merges contiguous windows into events and returns Nx2 [start_ms, end_ms]
    """
    x = np.asarray(data, dtype=np.float32)

    # Optional bandpass
    if min_freq and max_freq and max_freq > (min_freq or 0) and fs > 0:
        nyq = 0.5 * float(fs)
        low = float(min_freq) / nyq
        high = float(max_freq) / nyq
        low = max(low, 1e-6)
        high = min(high, 0.999999)
        if low < high:
            sos = butter(4, [low, high], btype='band', output='sos')
            try:
                x = sosfiltfilt(sos, x)
            except Exception:
                # Fallback to no filter if filtering fails
                pass

    win = max(1, int(window_size * float(fs)))
    step = max(1, int(win * (1.0 - float(overlap))))
    if step <= 0:
        step = 1

    # Compute RMS per window
    rms = []
    starts = []
    n = len(x)
    i = 0
    while i < n:
        j = min(n, i + win)
        seg = x[i:j]
        if seg.size == 0:
            break
        rms.append(float(np.sqrt(np.mean(seg * seg))))
        starts.append(i)
        if j >= n:
            break
        i += step

    if not rms:
        return np.asarray([])

    rms = np.asarray(rms, dtype=float)
    mu = float(np.mean(rms))
    sigma = float(np.std(rms))
    thr = mu + float(threshold) * sigma
    hot = rms > thr

    # Merge contiguous hot windows
    events = []
    k = 0
    W = win
    while k < len(hot):
        if not hot[k]:
            k += 1
            continue
        start_sample = starts[k]
        # extend while contiguous hot
        end_sample = min(n, start_sample + W)
        k += 1
        while k < len(hot) and hot[k]:
            end_sample = min(n, starts[k] + W)
            k += 1
        # Convert to ms
        s_ms = 1000.0 * start_sample / float(fs)
        e_ms = 1000.0 * end_sample / float(fs)
        if e_ms > s_ms:
            events.append([s_ms, e_ms])

    if not events:
        return np.asarray([])
    return np.asarray(events, dtype=float)


def ste_detect_events(data, fs, threshold=3.0, window_size=0.01, overlap=0.5, min_freq=80.0, max_freq=500.0, **kwargs):
    """
    Run Short-Term Energy (RMS) detection using local implementation.
    
    Implements a windowed RMS energy detector inspired by STE/RMS methods:
    - Bandpass filters to [min_freq, max_freq]
    - Computes windowed RMS with given window size and overlap
    - Flags windows exceeding mean + threshold * std of RMS
    - Merges contiguous windows into events
    
    Returns Nx2 [start_ms, end_ms] array.
    """
    return _local_ste_rms_detect(data, fs, threshold, window_size, overlap, min_freq, max_freq)


def mni_detect_events(data, fs, baseline_window=10.0, threshold_percentile=99.0, min_freq=80.0, **kwargs):
    """
    Run MNI-style detection using local implementation.
    
    Implements a percentile-based energy detector inspired by MNI methods:
    - Optional bandpass with lower cutoff at min_freq
    - Computes windowed RMS (20 ms, 50% overlap)
    - Computes global RMS threshold at given percentile
    - Merges contiguous supra-threshold windows into events
    
    Returns Nx2 [start_ms, end_ms] array.
    """
    return _local_mni_detect(data, fs, baseline_window, threshold_percentile, min_freq, kwargs.get('max_freq'))


def _local_mni_detect(data, fs, baseline_window=10.0, threshold_percentile=99.0, min_freq=80.0, max_freq=None):
    """
    Local fallback for MNI-like detection:
    - Optional bandpass with lower cutoff at min_freq
    - Compute windowed RMS (20 ms, 50% overlap)
    - Compute global RMS threshold at given percentile
    - Merge contiguous supra-threshold windows into events
    Returns Nx2 array of [start_ms, end_ms]
    """
    x = np.asarray(data, dtype=np.float32)

    # Optional bandpass using provided max_freq or Nyquist
    nyq = 0.5 * float(fs)
    low = float(min_freq) / nyq if min_freq else 1e-6
    hi_val = (float(max_freq) / nyq) if (max_freq is not None) else 0.999999
    low = max(low, 1e-6)
    high = min(max(low + 1e-6, hi_val), 0.999999)
    if low < high and fs > 0:
        try:
            sos = butter(4, [low, high], btype='band', output='sos')
            x = sosfiltfilt(sos, x)
        except Exception:
            # ignore filter failures
            pass

    # Use 20 ms detection window
    det_win = max(1, int(0.02 * float(fs)))
    step = max(1, int(det_win * 0.5))
    n = len(x)

    rms = []
    starts = []
    i = 0
    while i < n:
        j = min(n, i + det_win)
        seg = x[i:j]
        if seg.size == 0:
            break
        rms.append(float(np.sqrt(np.mean(seg * seg))))
        starts.append(i)
        if j >= n:
            break
        i += step

    if not rms:
        return np.asarray([])

    rms = np.asarray(rms, dtype=float)
    thr = float(np.percentile(rms, float(threshold_percentile)))
    hot = rms > thr

    events = []
    k = 0
    W = det_win
    while k < len(hot):
        if not hot[k]:
            k += 1
            continue
        start_sample = starts[k]
        end_sample = min(n, start_sample + W)
        k += 1
        while k < len(hot) and hot[k]:
            end_sample = min(n, starts[k] + W)
            k += 1
        s_ms = 1000.0 * start_sample / float(fs)
        e_ms = 1000.0 * end_sample / float(fs)
        if e_ms > s_ms:
            events.append([s_ms, e_ms])

    if not events:
        return np.asarray([])
    return np.asarray(events, dtype=float)


def dl_detect_events(data, fs, model_path, threshold=0.5, batch_size=32, progress_callback=None, **kwargs):
    """
    Run Deep Learning detection using local PyTorch/ONNX implementation.
    """
    _check_package()

    args = ParamDL(
        sample_freq=float(fs),
        model_path=str(model_path),
        threshold=float(threshold),
        batch_size=int(batch_size)
    )
    detector = set_DL_detector(args, progress_callback=progress_callback)
    signal = np.asarray(data, dtype=np.float32)
    detection_results = detector.detect(signal, 'chn1')
    return _convert_pyhfo_results_to_eois(detection_results, fs)


def dl_classify_segments(data, fs, segments_ms, model_path, threshold=0.5, batch_size=32, progress_callback=None):
    """
    Classify provided segments (EOIs) using a deep learning model if available.

    Inputs:
    - data: 1D signal array
    - fs: sampling frequency (Hz)
    - segments_ms: Nx2 array/list of [start_ms, end_ms]
    - model_path: path to model file (.pt/.pth for PyTorch, .onnx for ONNX)
    - threshold: probability threshold for positive class
    - batch_size: batch size for inference
    - progress_callback: optional function to call with progress updates

    Returns:
    - probs: 1D array of probabilities per segment (float in [0,1])
    - labels: list of string labels ('positive'/'negative') based on threshold
    """
    def _progress(msg):
        print(msg)
        if progress_callback:
            try:
                progress_callback(msg)
            except Exception as e:
                print(f"Progress callback error: {e}")
    
    segments_ms = np.asarray(segments_ms, dtype=float)
    if segments_ms.size == 0:
        return np.asarray([]), []

    # Verify model file exists and is valid
    if not model_path:
        _progress("ERROR: Model path is empty")
        _progress("WARNING: Using energy-based fallback instead of DL model")
        model_path = None
    elif not os.path.exists(model_path):
        _progress(f"ERROR: Model file not found: {model_path}")
        _progress("WARNING: Using energy-based fallback instead of DL model")
        model_path = None
    else:
        file_size = os.path.getsize(model_path)
        _progress(f"[DL] Model file size: {file_size / (1024*1024):.1f} MB")
    
    # Extract segments
    x = np.asarray(data, dtype=np.float32)
    idx = (np.column_stack([segments_ms[:, 0], segments_ms[:, 1]]) * float(fs) / 1000.0).astype(int)
    idx[:, 0] = np.clip(idx[:, 0], 0, len(x))
    idx[:, 1] = np.clip(idx[:, 1], 0, len(x))
    segs = []
    for s, e in idx:
        if e <= s:
            segs.append(np.zeros(1, dtype=np.float32))
        else:
            segs.append(x[s:e].copy())

    # If model file not available, use energy-based fallback
    if not model_path:
        _progress("[DL] Using energy-based fallback (no model available)")
        energies = np.asarray([float(np.mean(s*s)) for s in segs], dtype=float)
        if energies.size == 0:
            return np.asarray([]), []
        low = np.percentile(energies, 5.0)
        high = np.percentile(energies, 95.0)
        denom = max(high - low, 1e-8)
        probs = np.clip((energies - low) / denom, 0.0, 1.0)
        labels = ['positive' if float(p) >= float(threshold) else 'negative' for p in probs]
        return np.asarray(probs, dtype=float), labels

    # Try PyTorch model first
    probs = None
    try:
        _progress("[DL] Step 1/5: Importing PyTorch...")
        import torch
        _progress(f"[DL] PyTorch version: {torch.__version__}")
        torch.set_grad_enabled(False)
        
        _progress(f"[DL] Step 2/5: Checking model file...")
        _progress(f"[DL] Model path: {model_path}")
        
        # Load model - this is where it likely hangs
        _progress(f"[DL] Step 3/5: Loading model (this may take 10-60 seconds)...")
        mdl = None
        
        # Try TorchScript first
        try:
            _progress(f"[DL] Attempting torch.jit.load...")
            mdl = torch.jit.load(model_path, map_location='cpu')
            _progress(f"[DL] Successfully loaded as TorchScript")
        except Exception as e:
            _progress(f"[DL] TorchScript load failed: {type(e).__name__}")
            # Try regular torch.load
            try:
                _progress(f"[DL] Attempting torch.load...")
                mdl = torch.load(model_path, map_location='cpu')
                _progress(f"[DL] Successfully loaded with torch.load")
            except Exception as e2:
                _progress(f"[DL] torch.load failed: {type(e2).__name__}")
                mdl = None
        
        if mdl is not None:
            _progress(f"[DL] Step 4/5: Preparing {len(segs)} segments for inference...")
            mdl.eval()
            # Prepare batch: pad to max length, add channel dim
            max_len = max(int(s.size) for s in segs)
            padded = []
            for s in segs:
                if s.size < max_len:
                    pad = np.zeros(max_len - s.size, dtype=np.float32)
                    s = np.concatenate([s, pad])
                padded.append(s)
            arr = np.stack(padded, axis=0)
            # Normalize per segment
            mu = arr.mean(axis=1, keepdims=True)
            sd = arr.std(axis=1, keepdims=True) + 1e-8
            arr = (arr - mu) / sd
            tens = torch.from_numpy(arr).unsqueeze(1)  # (N, 1, L)

            _progress(f"[DL] Step 5/5: Running inference on {len(segs)} segments...")
            out = []
            for i in range(0, tens.size(0), int(batch_size)):
                batch = tens[i:i+int(batch_size)]
                y = mdl(batch)
                # Support (N,1) or (N,2) outputs
                y = y.detach().cpu().numpy()
                if y.ndim == 2 and y.shape[1] == 2:
                    p = 1.0 / (1.0 + np.exp(-(y[:, 1] - y[:, 0])))
                else:
                    p = 1.0 / (1.0 + np.exp(-y.squeeze()))
                out.append(p)
            probs = np.concatenate(out, axis=0)
            _progress(f"[DL] Inference complete! Classified {len(probs)} segments")
        else:
            _progress("[DL] Model unavailable, using energy-based fallback")
    except ImportError:
        _progress("[DL] PyTorch not installed, using energy-based fallback")
        probs = None
    except Exception as e:
        _progress(f"[DL] Unexpected error: {type(e).__name__}: {str(e)[:100]}")
        probs = None

    # Try ONNX Runtime if PyTorch unavailable
    if probs is None:
        try:
            import onnxruntime as ort
            max_len = max(int(s.size) for s in segs)
            padded = []
            for s in segs:
                if s.size < max_len:
                    pad = np.zeros(max_len - s.size, dtype=np.float32)
                    s = np.concatenate([s, pad])
                padded.append(s)
            arr = np.stack(padded, axis=0)
            mu = arr.mean(axis=1, keepdims=True)
            sd = arr.std(axis=1, keepdims=True) + 1e-8
            arr = (arr - mu) / sd
            arr = np.expand_dims(arr, 1)  # (N,1,L)
            sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            inp_name = sess.get_inputs()[0].name
            y = sess.run(None, {inp_name: arr})[0]
            if y.ndim == 2 and y.shape[1] == 2:
                probs = 1.0 / (1.0 + np.exp(-(y[:, 1] - y[:, 0])))
            else:
                probs = 1.0 / (1.0 + np.exp(-y.squeeze()))
        except Exception:
            probs = None

    # Final fallback: simple energy-based probability proxy
    if probs is None:
        energies = np.asarray([float(np.mean(s*s)) for s in segs], dtype=float)
        if energies.size == 0:
            return np.asarray([]), []
        # Map to [0,1] by percentile
        low = np.percentile(energies, 5.0)
        high = np.percentile(energies, 95.0)
        denom = max(high - low, 1e-8)
        probs = np.clip((energies - low) / denom, 0.0, 1.0)

    labels = ['positive' if float(p) >= float(threshold) else 'negative' for p in probs]
    return np.asarray(probs, dtype=float), labels


# ============================================================================
# Consensus Detection: Combine Hilbert, STE, and MNI via voting
# ============================================================================

def _merge_overlaps(events, overlap_threshold_ms=10.0):
    """
    Merge overlapping or near-overlapping events.
    
    Args:
        events: Nx2 array of [start_ms, stop_ms]
        overlap_threshold_ms: events within this distance are merged
    
    Returns:
        Merged Nx2 array
    """
    if events is None or len(events) == 0:
        return np.asarray([])
    
    events = np.asarray(events, dtype=float)
    if events.ndim == 1:
        events = events.reshape(-1, 2)
    
    # Sort by start time
    events = events[np.argsort(events[:, 0])]
    
    merged = []
    current_start, current_stop = events[0]
    
    for i in range(1, len(events)):
        start, stop = events[i]
        # Check if this event overlaps or is within threshold of current
        if start <= current_stop + overlap_threshold_ms:
            # Merge: extend current_stop
            current_stop = max(current_stop, stop)
        else:
            # No overlap: save current and start new
            merged.append([current_start, current_stop])
            current_start, current_stop = start, stop
    
    # Add final event
    merged.append([current_start, current_stop])
    
    return np.asarray(merged, dtype=float)


def _vote_consensus(all_events_list, voting_strategy='majority', overlap_threshold_ms=10.0):
    """
    Vote on consensus events: count how many detectors detected each event.
    
    Args:
        all_events_list: List of 3 Nx2 arrays (Hilbert, STE, MNI events)
        voting_strategy: 'strict' (3/3), 'majority' (2/3), 'any' (1/3)
        overlap_threshold_ms: events within this distance are considered same event
    
    Returns:
        Nx2 array of consensus events
    """
    if not all_events_list or all(len(e) == 0 for e in all_events_list):
        return np.asarray([])
    
    # Flatten and collect all events with detector labels
    event_detectors = []  # List of (start, stop, detector_idx)
    for detector_idx, events in enumerate(all_events_list):
        if events is None or len(events) == 0:
            continue
        for start, stop in np.asarray(events, dtype=float):
            event_detectors.append((float(start), float(stop), detector_idx))
    
    if not event_detectors:
        return np.asarray([])
    
    # Sort by start time
    event_detectors.sort(key=lambda x: x[0])
    
    # Group events by overlap
    consensus_events = []
    current_group = [event_detectors[0]]
    
    for i in range(1, len(event_detectors)):
        start, stop, det_idx = event_detectors[i]
        prev_stop = current_group[-1][1]
        
        # Check if overlapping with any in current group
        if start <= prev_stop + overlap_threshold_ms:
            current_group.append((start, stop, det_idx))
        else:
            # Process current group and start new
            if current_group:
                consensus_events.append(current_group)
            current_group = [(start, stop, det_idx)]
    
    # Process final group
    if current_group:
        consensus_events.append(current_group)
    
    # Vote: count unique detectors per group
    voted_events = []
    
    for group in consensus_events:
        # Get unique detectors and aggregate time bounds
        detectors_in_group = set(det_idx for _, _, det_idx in group)
        num_votes = len(detectors_in_group)
        
        # Check voting strategy
        pass_vote = False
        if voting_strategy == 'strict':
            pass_vote = (num_votes == 3)
        elif voting_strategy == 'majority':
            pass_vote = (num_votes >= 2)
        elif voting_strategy == 'any':
            pass_vote = (num_votes >= 1)
        
        if pass_vote:
            # Compute union of time bounds
            starts = [s for s, _, _ in group]
            stops = [st for _, st, _ in group]
            consensus_start = min(starts)
            consensus_stop = max(stops)
            voted_events.append([consensus_start, consensus_stop])
    
    if not voted_events:
        return np.asarray([])
    
    return np.asarray(voted_events, dtype=float)


def consensus_detect_events(data, fs, 
                           hilbert_params=None, ste_params=None, mni_params=None,
                           voting_strategy='majority', overlap_threshold_ms=10.0, **kwargs):
    """
    Run Hilbert, STE, and MNI detectors and return consensus events.
    
    Args:
        data: 1D signal array
        fs: sampling frequency (Hz)
        hilbert_params: dict of hilbert_detect_events kwargs
        ste_params: dict of ste_detect_events kwargs
        mni_params: dict of mni_detect_events kwargs
        voting_strategy: 'strict' (3/3), 'majority' (2/3), 'any' (1/3)
        overlap_threshold_ms: events within this distance are merged (default 10 ms)
    
    Returns:
        Nx2 array of [start_ms, stop_ms] consensus events
    """
    # Import locally to handle circular deps
    from .Score import hilbert_detect_events
    
    # Default parameters (conservative/balanced)
    if hilbert_params is None:
        hilbert_params = {
            'epoch': 300.0,
            'sd_num': 3.5,
            'min_duration': 10.0,
            'min_freq': 80.0,
            'max_freq': 500.0,
            'required_peak_number': 6,
            'required_peak_sd': 2.0,
            'boundary_fraction': 0.3
        }
    
    if ste_params is None:
        ste_params = {
            'threshold': 2.5,
            'window_size': 0.01,
            'overlap': 0.5,
            'min_freq': 80.0,
            'max_freq': 500.0
        }
    
    if mni_params is None:
        mni_params = {
            'baseline_window': 10.0,
            'threshold_percentile': 98.0,
            'min_freq': 80.0,
            'max_freq': 500.0
        }
    
    # Run all three detectors
    try:
        hilbert_eois = hilbert_detect_events(data, fs, **hilbert_params)
    except Exception:
        hilbert_eois = np.asarray([])
    
    try:
        ste_eois = ste_detect_events(data, fs, **ste_params)
    except Exception:
        ste_eois = np.asarray([])
    
    try:
        mni_eois = mni_detect_events(data, fs, **mni_params)
    except Exception:
        mni_eois = np.asarray([])
    
    # Merge overlaps within each detector result
    if len(hilbert_eois) > 0:
        hilbert_eois = _merge_overlaps(hilbert_eois, overlap_threshold_ms)
    if len(ste_eois) > 0:
        ste_eois = _merge_overlaps(ste_eois, overlap_threshold_ms)
    if len(mni_eois) > 0:
        mni_eois = _merge_overlaps(mni_eois, overlap_threshold_ms)
    
    # Vote on consensus
    all_events = [hilbert_eois, ste_eois, mni_eois]
    consensus_eois = _vote_consensus(all_events, voting_strategy, overlap_threshold_ms)
    
    return consensus_eois