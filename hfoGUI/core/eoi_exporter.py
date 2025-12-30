"""
Utility to export detected EOIs to .npy segments and a CSV manifest for DL training.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import time
import sys


def _safe_write_csv(df, path, max_retries=3, delay=0.5):
    """
    Write DataFrame to CSV with retry logic for locked files.
    
    Args:
        df: DataFrame to write
        path: Output path
        max_retries: Maximum number of retry attempts
        delay: Delay in seconds between retries
    
    Raises:
        PermissionError: If file remains locked after all retries
    """
    path = Path(path)
    
    for attempt in range(max_retries):
        try:
            df.to_csv(path, index=False)
            return  # Success
        except PermissionError as e:
            if attempt < max_retries - 1:
                print(f"Warning: {path.name} is locked (possibly open in Excel). Retrying in {delay}s... ({attempt + 1}/{max_retries})", file=sys.stderr)
                time.sleep(delay)
            else:
                raise PermissionError(
                    f"Cannot write to {path.name} - file is locked.\n"
                    f"Please close the file in Excel or other applications and try again."
                ) from e
        except Exception as e:
            # For other errors, fail immediately
            raise


def export_eois_for_training(signal, fs, eois_ms, out_dir, prefix="seg", metadata=None):
    """
    Extract EOI segments from a signal and write to .npy files with a manifest.

    Args:
        signal: 1D array, the raw signal.
        fs: Sampling frequency (Hz).
        eois_ms: Nx2 array/list of [start_ms, end_ms].
        out_dir: Output directory (will be created).
        prefix: Prefix for segment filenames.
        metadata: Optional list of dicts (same length as eois_ms) merged into the manifest rows.

    Returns:
        Path to the manifest CSV.
    """
    signal = np.asarray(signal, dtype=np.float32)
    eois_ms = np.asarray(eois_ms, dtype=float)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    meta = metadata if metadata is not None else [None] * len(eois_ms)

    for idx, ((s_ms, e_ms), extra) in enumerate(zip(eois_ms, meta)):
        s = int(float(s_ms) * fs / 1000.0)
        e = int(float(e_ms) * fs / 1000.0)
        s = max(0, s)
        e = min(len(signal), e)
        if e <= s:
            continue
        seg = signal[s:e].astype(np.float32)
        seg_path = out_dir / f"{prefix}_{idx:05d}.npy"
        np.save(seg_path, seg)
        row = {"segment_path": str(seg_path), "label": None}
        if isinstance(extra, dict):
            row.update(extra)
        rows.append(row)

    manifest_path = out_dir / "manifest.csv"
    _safe_write_csv(pd.DataFrame(rows), manifest_path)
    return manifest_path


def export_labeled_eois_for_training(signal, fs, eois_ms, labels, out_dir, prefix="seg"):
    """
    Extract EOI segments with labels and write .npy files plus a manifest.

    Args:
        signal: 1D array, the raw signal.
        fs: Sampling frequency (Hz).
        eois_ms: Nx2 array/list of [start_ms, end_ms].
        labels: iterable of same length as eois_ms with int labels (0/1).
        out_dir: Output directory (will be created).
        prefix: Prefix for segment filenames.

    Returns:
        Path to the manifest CSV.
    """
    signal = np.asarray(signal, dtype=np.float32)
    eois_ms = np.asarray(eois_ms, dtype=float)
    labels = np.asarray(labels).astype(int)
    if eois_ms.shape[0] != labels.shape[0]:
        raise ValueError("labels and eois_ms must have the same length")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, ((s_ms, e_ms), lbl) in enumerate(zip(eois_ms, labels)):
        s = int(float(s_ms) * fs / 1000.0)
        e = int(float(e_ms) * fs / 1000.0)
        s = max(0, s)
        e = min(len(signal), e)
        if e <= s:
            continue
        seg = signal[s:e].astype(np.float32)
        seg_path = out_dir / f"{prefix}_{idx:05d}.npy"
        np.save(seg_path, seg)
        rows.append({"segment_path": str(seg_path), "label": int(lbl)})

    manifest_path = out_dir / "manifest.csv"
    _safe_write_csv(pd.DataFrame(rows), manifest_path)
    return manifest_path
