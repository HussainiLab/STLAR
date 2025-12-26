import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path


def pad_collate_fn(batch):
    """Custom collate function to pad variable-length segments to max length in batch."""
    # batch is a list of (x, y) tuples where x is (1, L) and y is scalar
    xs, ys = zip(*batch)
    
    # Find max length
    max_len = max(x.shape[1] for x in xs)
    
    # Pad all to max_len
    xs_padded = []
    for x in xs:
        if x.shape[1] < max_len:
            pad = torch.zeros(1, max_len - x.shape[1], dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
        xs_padded.append(x)
    
    # Stack to get (B, 1, max_len) - required for Conv1d
    xs = torch.stack(xs_padded, dim=0)  # (B, 1, L)
    ys = torch.stack(ys)  # (B,)
    
    return xs, ys


class SegmentDataset(Dataset):
    """Dataset reading 1D waveform segments from .npy paths listed in a CSV manifest."""

    def __init__(self, manifest_csv: str):
        self.manifest = pd.read_csv(manifest_csv)
        if not {'segment_path', 'label'} <= set(self.manifest.columns):
            raise ValueError("Manifest must have columns: segment_path,label")

        # Coerce labels to numeric and drop any rows with NaN/inf labels
        manifest_len = len(self.manifest)
        self.manifest['label'] = pd.to_numeric(self.manifest['label'], errors='coerce')
        # Keep only finite labels
        self.manifest = self.manifest[np.isfinite(self.manifest['label'])]
        dropped = manifest_len - len(self.manifest)
        if dropped > 0:
            print(f"⚠️  Dropped {dropped} rows with missing/invalid labels in {manifest_csv}")

        # Normalize paths to strings; keep rows where segment_path is non-null
        self.manifest = self.manifest[self.manifest['segment_path'].notna()]
        self.paths = self.manifest['segment_path'].astype(str).tolist()
        # Use float labels; BCEWithLogits expects float targets
        self.labels = self.manifest['label'].astype(float).tolist()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = Path(self.paths[idx])
        x = np.load(path).astype(np.float32)
        # Per-segment z-score
        mu = x.mean() if x.size else 0.0
        sd = x.std() + 1e-8
        x = (x - mu) / sd
        x = torch.from_numpy(x).unsqueeze(0)  # (1, L)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y
