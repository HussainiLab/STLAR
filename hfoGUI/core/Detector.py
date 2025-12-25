import numpy as np
from scipy.signal import butter, sosfiltfilt

# Define dummies first so they exist if imports fail
class DummyDetector:
    def detect(self, *args, **kwargs): return np.array([])

def set_STE_detector(*args, **kwargs): return DummyDetector()
def set_MNI_detector(*args, **kwargs): return DummyDetector()
def set_DL_detector(*args, **kwargs): return DummyDetector()

class ParamSTE:
    def __init__(self, *args, **kwargs): pass
class ParamMNI:
    def __init__(self, *args, **kwargs): pass
class ParamDL:
    def __init__(self, *args, **kwargs): pass

_pyhfo_repo_available = False

try:
    import sys
    # This path is hardcoded in Score.py, so we'll use it here too.
    pyhfo_path = r'C:\Users\Abid\Documents\Code\Python\pyhfo_repo'
    if pyhfo_path not in sys.path:
        sys.path.insert(0, pyhfo_path)

    # Try importing detectors individually to handle missing ones gracefully
    try:
        from src.utils.utils_detector import set_STE_detector
        from src.param.param_detector import ParamSTE
        _pyhfo_repo_available = True
    except ImportError:
        # Fallback to RMS if STE not found
        try:
            from src.utils.utils_detector import set_RMS_detector as set_STE_detector
            from src.param.param_detector import ParamRMS as ParamSTE
            _pyhfo_repo_available = True
        except ImportError:
            print("Warning: set_STE_detector/set_RMS_detector not found in pyhfo_repo.")

    try:
        from src.utils.utils_detector import set_MNI_detector
        from src.param.param_detector import ParamMNI
        _pyhfo_repo_available = True
    except ImportError:
        print("Warning: set_MNI_detector not found in pyhfo_repo.")

    try:
        from src.utils.utils_detector import set_DL_detector
        from src.param.param_detector import ParamDL
        _pyhfo_repo_available = True
    except ImportError:
        pass  # Will use local DL implementation instead

except ImportError as e:
    print(f"Warning: Could not import required components from pyhfo_repo: {e}")
    print("Please ensure pyhfo_repo is at C:\\Users\\Abid\\Documents\\Code\\Python\\pyhfo_repo and contains the expected detectors.")
    _pyhfo_repo_available = False


def _check_package():
    if not _pyhfo_repo_available:
        raise ImportError("pyhfo_repo components not found. Please check the path and installation.")


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
    Local fallback STE/RMS detector independent of pyhfo_repo.
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
    Run Short-Term Energy (RMS) detection via pyhfo_repo if available,
    otherwise use a local fallback implementation.
    """
    # If pyhfo_repo isn't available, use local fallback immediately
    try:
        _check_package()
    except ImportError:
        return _local_ste_rms_detect(data, fs, threshold, window_size, overlap, min_freq, max_freq)

    # Assuming ParamRMS takes window_size in samples
    win_size_samples = int(window_size * fs)

    signal = np.asarray(data, dtype=np.float32)

    # Try pyhfo_repo path first, falling back to local if construction fails
    try:
        args = ParamSTE(
            sample_freq=float(fs),
            pass_band=float(min_freq),
            stop_band=float(max_freq),
            sd_threshold=float(threshold),
            window_size=win_size_samples,
            window_overlap=float(overlap)
        )
        detector = set_STE_detector(args)
        detection_results = detector.detect(signal, 'chn1')
        eois = _convert_pyhfo_results_to_eois(detection_results, fs)
        if eois.size:
            return eois
        # If detector returned empty, try local fallback
        return _local_ste_rms_detect(signal, fs, threshold, window_size, overlap, min_freq, max_freq)
    except TypeError as e:
        # Handle mismatched parameter names between versions of pyhfo_repo
        # Try alternative common names, else fallback to local implementation
        try:
            args = ParamSTE(
                sample_freq=float(fs),
                pass_band=float(min_freq),
                stop_band=float(max_freq),
                threshold=float(threshold),
                window_samples=win_size_samples,
                overlap=float(overlap)
            )
            detector = set_STE_detector(args)
            detection_results = detector.detect(signal, 'chn1')
            eois = _convert_pyhfo_results_to_eois(detection_results, fs)
            if eois.size:
                return eois
        except Exception:
            pass
        # Final fallback
        return _local_ste_rms_detect(signal, fs, threshold, window_size, overlap, min_freq, max_freq)
    except Exception:
        # Any other failure: fallback to local implementation
        return _local_ste_rms_detect(signal, fs, threshold, window_size, overlap, min_freq, max_freq)


def mni_detect_events(data, fs, baseline_window=10.0, threshold_percentile=99.0, min_freq=80.0, **kwargs):
    """
    Run MNI detection via pyhfo_repo if available,
    otherwise use a local fallback implementation.
    """
    signal = np.asarray(data, dtype=np.float32)

    # If pyhfo_repo isn't available, use local fallback immediately
    try:
        _check_package()
    except ImportError:
        return _local_mni_detect(signal, fs, baseline_window, threshold_percentile, min_freq, kwargs.get('max_freq'))

    # Try pyhfo_repo path first
    try:
        args = ParamMNI(
            sample_freq=float(fs),
            pass_band=float(min_freq),
            stop_band=kwargs.get('max_freq', fs / 2.0 - 1),
            baseline_duration=float(baseline_window),
            threshold_p=float(threshold_percentile)
        )
        detector = set_MNI_detector(args)
        detection_results = detector.detect(signal, 'chn1')
        eois = _convert_pyhfo_results_to_eois(detection_results, fs)
        if eois.size:
            return eois
        return _local_mni_detect(signal, fs, baseline_window, threshold_percentile, min_freq, kwargs.get('max_freq'))
    except TypeError:
        # Handle mismatched parameter names between versions of pyhfo_repo
        try:
            args = ParamMNI(
                sample_freq=float(fs),
                pass_band=float(min_freq),
                max_freq=kwargs.get('max_freq', fs / 2.0 - 1),
                baseline_window=float(baseline_window),
                threshold_percentile=float(threshold_percentile)
            )
            detector = set_MNI_detector(args)
            detection_results = detector.detect(signal, 'chn1')
            eois = _convert_pyhfo_results_to_eois(detection_results, fs)
            if eois.size:
                return eois
        except Exception:
            pass
        return _local_mni_detect(signal, fs, baseline_window, threshold_percentile, min_freq, kwargs.get('max_freq'))
    except Exception:
        return _local_mni_detect(signal, fs, baseline_window, threshold_percentile, min_freq, kwargs.get('max_freq'))


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


def dl_detect_events(data, fs, model_path, threshold=0.5, batch_size=32, **kwargs):
    """
    Run Deep Learning detection via pyhfo_repo.
    """
    _check_package()

    args = ParamDL(
        sample_freq=float(fs),
        model_path=str(model_path),
        threshold=float(threshold),
        batch_size=int(batch_size)
    )
    detector = set_DL_detector(args)
    signal = np.asarray(data, dtype=np.float32)
    detection_results = detector.detect(signal, 'chn1')
    return _convert_pyhfo_results_to_eois(detection_results, fs)


def dl_classify_segments(data, fs, segments_ms, model_path, threshold=0.5, batch_size=32):
    """
    Classify provided segments (EOIs) using a deep learning model if available.

    Inputs:
    - data: 1D signal array
    - fs: sampling frequency (Hz)
    - segments_ms: Nx2 array/list of [start_ms, end_ms]
    - model_path: path to model file (.pt/.pth for PyTorch, .onnx for ONNX)
    - threshold: probability threshold for positive class
    - batch_size: batch size for inference

    Returns:
    - probs: 1D array of probabilities per segment (float in [0,1])
    - labels: list of string labels ('positive'/'negative') based on threshold
    """
    segments_ms = np.asarray(segments_ms, dtype=float)
    if segments_ms.size == 0:
        return np.asarray([]), []

    x = np.asarray(data, dtype=np.float32)

    # Extract segments in samples
    idx = (np.column_stack([segments_ms[:, 0], segments_ms[:, 1]]) * float(fs) / 1000.0).astype(int)
    idx[:, 0] = np.clip(idx[:, 0], 0, len(x))
    idx[:, 1] = np.clip(idx[:, 1], 0, len(x))
    segs = []
    for s, e in idx:
        if e <= s:
            segs.append(np.zeros(1, dtype=np.float32))
        else:
            segs.append(x[s:e].copy())

    # Try PyTorch model first
    probs = None
    try:
        import torch
        torch.set_grad_enabled(False)
        # Load model
        mdl = None
        try:
            mdl = torch.jit.load(model_path, map_location='cpu')
        except Exception:
            try:
                mdl = torch.load(model_path, map_location='cpu')
            except Exception:
                mdl = None
        if mdl is not None:
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
    except ImportError:
        probs = None
    except Exception:
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