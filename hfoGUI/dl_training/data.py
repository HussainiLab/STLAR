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
        manifest_path = Path(manifest_csv)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {manifest_csv}")
        
        self.manifest = pd.read_csv(manifest_csv)
        if not {'segment_path', 'label'} <= set(self.manifest.columns):
            raise ValueError("Manifest must have columns: segment_path,label")

        # Store original count
        original_count = len(self.manifest)
        
        # Coerce labels to numeric and drop any rows with NaN/inf labels
        self.manifest['label'] = pd.to_numeric(self.manifest['label'], errors='coerce')
        # Keep only finite labels
        self.manifest = self.manifest[np.isfinite(self.manifest['label'])]
        dropped = original_count - len(self.manifest)
        
        if dropped > 0:
            print(f"WARNING: Dropped {dropped} rows with missing/invalid labels in {manifest_csv}")
        
        # Check if we have any valid data left
        if len(self.manifest) == 0:
            raise ValueError(
                f"ERROR: No valid samples in {manifest_csv}!\n"
                f"  - Original rows: {original_count}\n"
                f"  - Dropped: {dropped} (all had invalid/missing labels)\n"
                f"  - Remaining: 0\n\n"
                f"Possible causes:\n"
                f"  1. Label column has no valid numeric values (0 or 1)\n"
                f"  2. Label column name is incorrect (expected 'label')\n"
                f"  3. CSV file is corrupted or improperly formatted\n\n"
                f"Please check your manifest file:\n"
                f"  {manifest_path.resolve()}"
            )

        # Normalize paths to strings; keep rows where segment_path is non-null
        self.manifest = self.manifest[self.manifest['segment_path'].notna()]
        
        if len(self.manifest) == 0:
            raise ValueError(
                f"ERROR: No valid segment paths in {manifest_csv}!\n"
                f"All rows have missing segment_path values."
            )
        
        self.paths = self.manifest['segment_path'].astype(str).tolist()
        # Use float labels; BCEWithLogits expects float targets
        self.labels = self.manifest['label'].astype(float).tolist()
        
        # Validate that all paths exist
        self._validate_paths(manifest_csv)

    def __len__(self):
        return len(self.paths)
    
    def _validate_paths(self, manifest_csv):
        """Validate that all segment files exist before training starts."""
        missing = []
        for path_str in self.paths:
            path = Path(path_str)
            if not path.exists():
                missing.append(path_str)
        
        if missing:
            error_msg = f"ERROR: {len(missing)} segment files not found!\n\n"
            error_msg += f"First few missing files:\n"
            for path_str in missing[:5]:
                error_msg += f"  - {path_str}\n"
            if len(missing) > 5:
                error_msg += f"  ... and {len(missing) - 5} more\n"
            
            # Check for common issues
            if any('1Scores' in p and 'HFO' not in p for p in missing):
                error_msg += f"\nPossible issue detected:\n"
                error_msg += f"  Paths contain '1Scores' instead of 'HFOScores'\n"
                error_msg += f"  This suggests prepare-dl was run with the wrong directory.\n"
            
            error_msg += f"\nTo fix this:\n"
            error_msg += f"  1. Delete the corrupted manifest files\n"
            error_msg += f"  2. Re-run prepare-dl with correct paths:\n"
            error_msg += f"     python -m stlar prepare-dl \\\\\n"
            error_msg += f"       --eoi-file <detection_file.txt> \\\\\n"
            error_msg += f"       --egf-file <raw_data.egf> \\\\\n"
            error_msg += f"       --set-file <raw_data.set> \\\\\n"
            error_msg += f"       --output <output_dir> \\\\\n"
            error_msg += f"       --split-train-val\n"
            
            raise FileNotFoundError(error_msg)

    def __getitem__(self, idx):
        path_str = self.paths[idx]
        path = Path(path_str)
        
        if not path.exists():
            # Try to provide helpful diagnostics
            possible_issues = []
            if '1Scores' in path_str and 'HFO' not in path_str:
                possible_issues.append("Path has '1Scores' instead of 'HFOScores'")
            if not path.parent.exists():
                possible_issues.append(f"Directory does not exist: {path.parent}")
            
            error_msg = f"Segment file not found:\n  {path.resolve()}\n"
            if possible_issues:
                error_msg += "\nPossible issues:\n"
                for issue in possible_issues:
                    error_msg += f"  - {issue}\n"
            error_msg += f"\nPlease verify:\n"
            error_msg += f"  1. All paths in manifest are correct and accessible\n"
            error_msg += f"  2. You ran prepare-dl with the correct data directory\n"
            error_msg += f"  3. Segment files (*.npy) exist in the output directory\n"
            
            raise FileNotFoundError(error_msg)
        
        try:
            x = np.load(path).astype(np.float32)
        except (OSError, ValueError) as e:
            raise RuntimeError(f"Failed to load segment file {path}: {e}")
        
        # Per-segment z-score
        mu = x.mean() if x.size else 0.0
        sd = x.std() + 1e-8
        x = (x - mu) / sd
        x = torch.from_numpy(x).unsqueeze(0)  # (1, L)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y
