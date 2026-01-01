import argparse
from pathlib import Path
import numpy as np
import torch
from hfoGUI.core.Detector import ParamDL, _LocalDLDetector
from hfoGUI.core.Tint_Matlab import ReadEEG


def load_signal(path: Path):
    """Return (data, fs). fs may be None if not determined from file."""
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    if path.is_dir():
        raise IsADirectoryError(f"Data path is a directory, expected a data file: {path}")

    ext = path.suffix.lower()
    if ext in {".npy", ""}:  # allow extensionless .npy
        return np.load(path, allow_pickle=False), None
    if ext == ".npz":
        with np.load(path, allow_pickle=False) as z:
            keys = list(z.keys())
            if not keys:
                raise ValueError(f"No arrays found in {path}")
            return z[keys[0]], None
    if ext in {".egf", ".eeg"}:  # Tint files
        data, fs = ReadEEG(str(path))
        return np.asarray(data), float(fs)
    raise ValueError(f"Unsupported data format for {path}; use .npy/.npz/.egf/.eeg")


def main():
    ap = argparse.ArgumentParser(description="Dump per-window DL probabilities")
    ap.add_argument("--data", required=True, help="Path to 1D signal (.npy/.npz/.egf/.eeg)")
    ap.add_argument("--fs", type=float, default=None, help="Sampling rate in Hz (optional if .egf/.eeg)")
    ap.add_argument("--model", required=True, help="Path to TorchScript model (.pt/.pth)")
    ap.add_argument("--win", type=float, default=1.0, help="Window length (s)")
    ap.add_argument("--hop", type=float, default=0.5, help="Hop length (s)")
    args = ap.parse_args()

    data_path = Path(args.data)
    model_path = Path(args.model)

    data, fs_in_file = load_signal(data_path)
    data = data.astype(np.float32).flatten()
    fs = float(args.fs) if args.fs is not None else fs_in_file
    if fs is None:
        raise ValueError("Sampling rate not provided and not found in file. Use --fs.")
    det = _LocalDLDetector(ParamDL(fs, str(model_path)))

    win = int(args.win * fs)
    hop = int(args.hop * fs)
    probs = []
    for start in range(0, len(data), hop):
        seg = data[start:start+win]
        if seg.size == 0:
            continue
        seg = (seg - seg.mean()) / (seg.std() + 1e-8)
        tens = torch.from_numpy(seg).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            logit = det.model(tens)
            if isinstance(logit, (list, tuple)):
                logit = logit[0]
            probs.append(torch.sigmoid(logit.squeeze()).item())

    probs = np.array(probs)
    print(f"windows: {len(probs)}")
    print(f"min/max/mean: {probs.min():.4f} / {probs.max():.4f} / {probs.mean():.4f}")
    print("pcts 1,5,25,50,75,95,99:", " ".join(f"{p:.4f}" for p in np.percentile(probs, [1, 5, 25, 50, 75, 95, 99])))

    spread = probs.max() - probs.min()
    std = probs.std()
    if spread < 0.10 or std < 0.03:
        verdict = "Very narrow probability spread; model likely undertrained. Retrain with more epochs and ensure good labels."
    elif spread < 0.25:
        verdict = "Moderately narrow spread; add more epochs or harder negatives to improve separation."
    else:
        verdict = "Healthy spread; tweak detection threshold or continue training if quality is still low."
    print(f"assessment: {verdict}")


if __name__ == "__main__":
    main()