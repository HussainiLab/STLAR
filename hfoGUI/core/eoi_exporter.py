"""
Utility to export detected EOIs to .npy segments and a CSV manifest for DL training.
"""
import numpy as np
import pandas as pd
from pathlib import Path


def export_eois_for_training(signal, fs, eois_ms, out_dir, prefix="seg"):
    """
    Extract EOI segments from a signal and write to .npy files with a manifest.

    Args:
        signal: 1D array, the raw signal.
        fs: Sampling frequency (Hz).
        eois_ms: Nx2 array/list of [start_ms, end_ms].
        out_dir: Output directory (will be created).
        prefix: Prefix for segment filenames.

    Returns:
        Path to the manifest CSV.
    """
    signal = np.asarray(signal, dtype=np.float32)
    eois_ms = np.asarray(eois_ms, dtype=float)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, (s_ms, e_ms) in enumerate(eois_ms):
        s = int(float(s_ms) * fs / 1000.0)
        e = int(float(e_ms) * fs / 1000.0)
        s = max(0, s)
        e = min(len(signal), e)
        if e <= s:
            continue
        seg = signal[s:e].astype(np.float32)
        seg_path = out_dir / f"{prefix}_{idx:05d}.npy"
        np.save(seg_path, seg)
        rows.append({"segment_path": str(seg_path), "label": None})

    manifest_path = out_dir / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
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
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return manifest_path
