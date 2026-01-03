"""
Utility functions for Continuous Wavelet Transform (CWT) computation and visualization.

This module provides reusable CWT processing functions for generating scalogram
representations of EEG signals, supporting both training and inference workflows.
"""

import numpy as np
import scipy.signal as signal
from pathlib import Path


def compute_cwt_scalogram(signal_data, fs=4800, freq_min=80, freq_max=500, num_freqs=64):
    """
    Compute CWT scalogram from a normalized 1D signal.
    
    Uses Morlet wavelet to generate a time-frequency representation (scalogram)
    suitable for 2D CNN input.
    
    Args:
        signal_data (np.ndarray): Normalized 1D signal (e.g., z-scored EEG).
        fs (int): Sampling frequency in Hz (default: 4800).
        freq_min (int): Minimum frequency of interest in Hz (default: 80).
        freq_max (int): Maximum frequency of interest in Hz (default: 500).
        num_freqs (int): Number of frequency bins for scalogram (default: 64).
    
    Returns:
        np.ndarray: CWT scalogram matrix with shape (num_freqs, time_steps).
                   Values are log-transformed power (log1p applied).
    
    Notes:
        - Input signal should be z-score normalized before calling this function
        - Morlet wavelet parameter w=6.0 provides standard time/frequency balance
        - Log transformation improves neural network training on power values
    """
    # Define frequency bins of interest (Ripple & Fast Ripple: 80-500 Hz)
    freqs = np.linspace(freq_min, freq_max, num_freqs)
    
    # Morlet wavelet parameter balancing time/frequency resolution
    w = 6.0
    
    # Compute wavelet widths for each frequency
    widths = w * fs / (2 * freqs * np.pi)
    
    # Compute CWT: Returns complex matrix (num_freqs, time_steps)
    cwtmatr = signal.cwt(signal_data, signal.morlet2, widths, w=w)
    
    # Convert to power and apply log normalization
    cwt_power = np.abs(cwtmatr) ** 2
    cwt_log = np.log1p(cwt_power)  # Log scale is crucial for neural networks
    
    return cwt_log


def save_scalogram_image(cwt_matrix, output_path, label=None, sample_idx=0, fs=4800):
    """
    Save CWT scalogram as a PNG image for visualization and inspection.
    
    Args:
        cwt_matrix (np.ndarray): CWT scalogram matrix with shape (num_freqs, time_steps).
        output_path (str or Path): Directory or file path where image will be saved.
        label (float, optional): Label value (0 or 1) for HFO/NonHFO classification.
                                 If None, used for unlabeled inference data.
        sample_idx (int): Sample index for filename generation (default: 0).
        fs (int): Sampling frequency in Hz (default: 4800).
    
    Returns:
        bool: True if image was saved successfully, False otherwise.
    
    Notes:
        - Requires matplotlib to be installed
        - Creates parent directories if they don't exist
        - Frequency axis covers 80-500 Hz with reference line at 250 Hz (ripple/FR boundary)
        - Only prints debug info for first 3 samples to avoid log spam
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename based on label
        if label is not None:
            label_str = "HFO" if label > 0.5 else "NonHFO"
            filename = f"scalogram_{sample_idx:06d}_{label_str}.png"
        else:
            filename = f"inference_scalogram_{sample_idx:06d}.png"
        
        filepath = output_dir / filename
        
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
        
        # Set title based on mode
        if label is not None:
            ax.set_title(f'CWT Scalogram - {label_str} (Sample {sample_idx})')
        else:
            ax.set_title(f'CWT Scalogram - Inference (Sample {sample_idx})')
        
        # Add reference line for ripple/fast-ripple boundary (250 Hz -> bin 25)
        ax.axhline(y=25, color='white', linestyle='--', linewidth=0.5, alpha=0.5)
        
        cbar = plt.colorbar(im, ax=ax, label='Power (log scale)')
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        # Only print for first few samples to avoid log spam
        if sample_idx < 3:
            print(f"[CWT DEBUG] Saved: {filepath}")
        
        return True
        
    except ImportError:
        if sample_idx == 0:
            print("[CWT DEBUG] matplotlib not available. Install with: pip install matplotlib")
        return False
    except Exception as e:
        print(f"[CWT DEBUG] Warning: Could not save scalogram {sample_idx}: {e}")
        return False
