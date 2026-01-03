import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
import scipy.signal as signal


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


def pad_collate_fn_2d(batch):
    """
    Collate function for 2D CWT tensors with variable time length.
    Pads the time dimension (width) to the maximum length in the batch.
    
    Args:
        batch: List of tuples (image_tensor, label) or just image_tensor
               image_tensor shape: (1, Freq, Time)
    """
    # Handle both cases: with labels (from SegmentDataset) and without (from CWT_InferenceDataset)
    if isinstance(batch[0], tuple):
        images, labels = zip(*batch)
    else:
        # No labels provided (inference mode)
        images = batch
        labels = None
    
    # Transpose to (Time, Freq) for padding (pad_sequence pads dim 0)
    # image: (1, 64, T) -> squeeze -> (64, T) -> transpose -> (T, 64)
    images_transposed = [img.squeeze(0).transpose(0, 1) for img in images]
    
    # Pad: Result is (Batch, Max_Time, Freq)
    padded_images = pad_sequence(images_transposed, batch_first=True, padding_value=0)
    
    # Restore dimensions: (Batch, Max_Time, Freq) -> (Batch, Freq, Max_Time) -> (Batch, 1, Freq, Max_Time)
    padded_images = padded_images.transpose(1, 2).unsqueeze(1)
    
    if labels is not None:
        labels = torch.stack(labels)
        return padded_images, labels
    else:
        return padded_images


class SegmentDataset(Dataset):
    """Dataset reading 1D waveform segments from .npy paths listed in a CSV manifest."""

    def __init__(self, manifest_csv: str, use_cwt=False, fs=4800, debug_cwt_dir=None):
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
        
        self.use_cwt = use_cwt
        self.fs = fs
        self.debug_cwt_dir = debug_cwt_dir
        
        # Create debug directory if specified
        if self.debug_cwt_dir:
            debug_path = Path(self.debug_cwt_dir)
            debug_path.mkdir(parents=True, exist_ok=True)
            print(f"[CWT DEBUG] Saving scalogram images to: {debug_path.resolve()}")
        
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
        x_norm = (x - mu) / sd
        
        y = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.use_cwt:
            import scipy.signal as signal
            # CWT logic to create a 2D scalogram
            freqs = np.linspace(80, 500, 64) 
            w = 6.0
            widths = w * self.fs / (2 * freqs * np.pi)
            cwtmatr = signal.cwt(x_norm, signal.morlet2, widths, w=w)
            cwt_image = np.abs(cwtmatr)**2
            cwt_image = np.log1p(cwt_image)
            
            # Debug: Save scalogram image if debug directory specified
            if self.debug_cwt_dir:
                self._save_scalogram_image(cwt_image, idx, y.item())
            
            x_tensor = torch.from_numpy(cwt_image).float().unsqueeze(0)
        else:
            x_tensor = torch.from_numpy(x_norm).unsqueeze(0)  # (1, L)
        
        return x_tensor, y
    
    def _save_scalogram_image(self, cwt_matrix, idx, label):
        """Save CWT scalogram as a PNG image for inspection."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.colors import LogNorm
            
            label_str = "HFO" if label > 0.5 else "NonHFO"
            filename = f"scalogram_{idx:06d}_{label_str}.png"
            filepath = Path(self.debug_cwt_dir) / filename
            
            # Create figure with proper scaling
            fig, ax = plt.subplots(figsize=(10, 4), dpi=100)
            
            # Display scalogram (frequency on y-axis, time on x-axis)
            # Use log scale for better visibility of weak features
            im = ax.imshow(cwt_matrix, aspect='auto', origin='lower', 
                          cmap='jet', norm=LogNorm(vmin=cwt_matrix.min() + 1e-8, vmax=cwt_matrix.max()))
            
            # Set Y-axis to show actual frequencies in Hz
            # Frequencies are linearly spaced from 80-500 Hz across 64 bins
            freq_ticks = [0, 10, 20, 25, 30, 40, 50, 63]  # Index positions
            freq_labels = ['80', '147', '213', '250', '280', '347', '413', '500']  # Hz values
            ax.set_yticks(freq_ticks)
            ax.set_yticklabels(freq_labels)
            ax.set_ylabel('Frequency (Hz)')
            ax.set_xlabel('Time Samples')
            ax.set_title(f'CWT Scalogram - {label_str} (Sample {idx})')
            
            # Add reference lines for ripple/fast-ripple boundary
            ax.axhline(y=25, color='white', linestyle='--', linewidth=0.5, alpha=0.5)
            
            cbar = plt.colorbar(im, ax=ax, label='Power (log scale)')
            
            plt.tight_layout()
            plt.savefig(filepath, dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            if idx < 3:  # Only print for first few samples
                print(f"[CWT DEBUG] Saved: {filepath}")
                
        except ImportError:
            if idx == 0:
                print("[CWT DEBUG] matplotlib not available. Install with: pip install matplotlib")
        except Exception as e:
            print(f"[CWT DEBUG] Warning: Could not save scalogram {idx}: {e}")

class CWT_InferenceDataset(Dataset):
    """
    Lightweight dataset for CWT inference (no labels required).
    Used when running detection on raw signal segments without training.
    Automatically applies CWT scalogram transformation to each segment.
    """
    def __init__(self, raw_signals, fs=4800, debug_cwt_dir=None):
        """
        Args:
            raw_signals (list): List of 1D raw EEG signal arrays (numpy or similar).
            fs (int): Sampling frequency in Hz (default 4800).
            debug_cwt_dir (str): Optional directory to save scalogram images for inspection.
        """
        self.raw_signals = raw_signals
        self.fs = fs
        self.debug_cwt_dir = debug_cwt_dir
        
        # Create debug directory if specified
        if self.debug_cwt_dir:
            debug_path = Path(self.debug_cwt_dir)
            debug_path.mkdir(parents=True, exist_ok=True)
            print(f"[CWT DEBUG] Saving inference scalogram images to: {debug_path.resolve()}")

    def __len__(self):
        return len(self.raw_signals)

    def __getitem__(self, idx):
        """
        Return CWT scalogram tensor for a single segment.
        
        Returns:
            image_tensor (torch.Tensor): Shape (1, 64, T) where T is time dimension.
        """
        sig = np.array(self.raw_signals[idx], dtype=np.float32).flatten()
        
        # Per-segment normalization (z-score)
        mu = sig.mean() if sig.size else 0.0
        sd = sig.std() + 1e-8
        sig_norm = (sig - mu) / sd
        
        # Apply CWT with Morlet wavelet
        # Define frequencies of interest (Ripple & Fast Ripple: 80-500 Hz)
        freqs = np.linspace(80, 500, 64)
        w = 6.0  # Standard Morlet parameter balancing time/frequency resolution
        widths = w * self.fs / (2 * freqs * np.pi)
        
        # Compute CWT: Returns complex matrix (64 frequencies, T time points)
        cwtmatr = signal.cwt(sig_norm, signal.morlet2, widths, w=w)
        
        # Convert to power and apply log normalization
        cwt_power = np.abs(cwtmatr) ** 2
        cwt_log = np.log1p(cwt_power)  # Log scale is crucial for neural networks
        
        # Debug: Save scalogram image if debug directory specified
        if self.debug_cwt_dir:
            self._save_scalogram_image(cwt_log, idx)
        
        # Convert to tensor: (1, 64, T) - treat as 1-channel grayscale image
        image_tensor = torch.from_numpy(cwt_log).float().unsqueeze(0)
        
        return image_tensor
    
    def _save_scalogram_image(self, cwt_matrix, idx):
        """Save CWT scalogram as a PNG image for inspection."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.colors import LogNorm
            
            filename = f"inference_scalogram_{idx:06d}.png"
            filepath = Path(self.debug_cwt_dir) / filename
            
            # Create figure with proper scaling
            fig, ax = plt.subplots(figsize=(10, 4), dpi=100)
            
            # Display scalogram (frequency on y-axis, time on x-axis)
            # Use log scale for better visibility of weak features
            im = ax.imshow(cwt_matrix, aspect='auto', origin='lower', 
                          cmap='jet', norm=LogNorm(vmin=cwt_matrix.min() + 1e-8, vmax=cwt_matrix.max()))
            
            # Set Y-axis to show actual frequencies in Hz
            # Frequencies are linearly spaced from 80-500 Hz across 64 bins
            freq_ticks = [0, 10, 20, 25, 30, 40, 50, 63]  # Index positions
            freq_labels = ['80', '147', '213', '250', '280', '347', '413', '500']  # Hz values
            ax.set_yticks(freq_ticks)
            ax.set_yticklabels(freq_labels)
            ax.set_ylabel('Frequency (Hz)')
            ax.set_xlabel('Time Samples')
            ax.set_title(f'CWT Scalogram - Inference (Sample {idx})')
            
            # Add reference lines for ripple/fast-ripple boundary
            ax.axhline(y=25, color='white', linestyle='--', linewidth=0.5, alpha=0.5)
            
            cbar = plt.colorbar(im, ax=ax, label='Power (log scale)')
            
            plt.tight_layout()
            plt.savefig(filepath, dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            if idx < 3:  # Only print for first few samples
                print(f"[CWT DEBUG] Saved: {filepath}")
                
        except ImportError:
            if idx == 0:
                print("[CWT DEBUG] matplotlib not available. Install with: pip install matplotlib")
        except Exception as e:
            print(f"[CWT DEBUG] Warning: Could not save scalogram {idx}: {e}")