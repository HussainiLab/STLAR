#!/usr/bin/env python3
"""
Quick test script to demonstrate CWT scalogram debugging.

This creates sample training data and generates CWT scalogram images
to verify the CWT preprocessing is working correctly.

Usage:
    python test_cwt_debug.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from hfoGUI.dl_training.data import SegmentDataset


def create_sample_data(output_dir, num_samples=10):
    """Create sample training data for testing."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create sample segments (1D signals)
    segment_paths = []
    labels = []
    
    for i in range(num_samples):
        # Generate synthetic signal
        t = np.linspace(0, 1, 4800)  # 1 second at 4800 Hz
        
        if i % 2 == 0:
            # "HFO" signal: add 100 Hz oscillation
            signal = np.sin(2 * np.pi * 100 * t) + 0.5 * np.random.randn(len(t))
            label = 1
        else:
            # "Non-HFO" signal: pure noise
            signal = 0.5 * np.random.randn(len(t))
            label = 0
        
        # Save segment
        seg_path = output_path / f"seg_{i:04d}.npy"
        np.save(seg_path, signal.astype(np.float32))
        
        segment_paths.append(str(seg_path))
        labels.append(label)
    
    # Create manifest CSV
    manifest_path = output_path / "manifest.csv"
    df = pd.DataFrame({
        'segment_path': segment_paths,
        'label': labels
    })
    df.to_csv(manifest_path, index=False)
    
    print(f"Created {num_samples} sample segments in {output_path}")
    return manifest_path


def test_cwt_debug():
    """Test CWT debug mode."""
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        data_dir = temp_path / "data"
        debug_dir = temp_path / "cwt_scalograms"
        
        # Create sample data
        manifest_path = create_sample_data(data_dir)
        
        print(f"\nLoading dataset with CWT preprocessing...")
        print(f"Debug output directory: {debug_dir}")
        
        # Create dataset with debug mode enabled
        dataset = SegmentDataset(
            str(manifest_path),
            use_cwt=True,
            fs=4800,
            debug_cwt_dir=str(debug_dir)
        )
        
        print(f"Dataset size: {len(dataset)}")
        print(f"\nGenerating CWT scalograms (saving first 3 samples)...")
        
        # Load first few samples to trigger scalogram generation
        for i in range(min(3, len(dataset))):
            x, y = dataset[i]
            label_str = "HFO" if y > 0.5 else "NonHFO"
            print(f"  Sample {i}: {x.shape} tensor, label={label_str}")
        
        # Check what was saved
        saved_files = list(debug_dir.glob("*.png"))
        print(f"\nSaved {len(saved_files)} scalogram images:")
        for f in sorted(saved_files):
            print(f"  ✓ {f.name}")
        
        if saved_files:
            print(f"\nScalograms saved to: {debug_dir.resolve()}")
            print("You can now inspect these images to verify CWT preprocessing is working correctly.")
            print("\nEach image shows:")
            print("  - X-axis: Time samples (0 to signal length)")
            print("  - Y-axis: Frequency index (0-63, corresponding to 80-500 Hz)")
            print("  - Color: Power intensity (log scale, darker = lower power)")
            print("  - Title: Sample label (HFO vs NonHFO)")
        
        return len(saved_files) > 0


if __name__ == "__main__":
    try:
        success = test_cwt_debug()
        if success:
            print("\n✅ CWT Debug Test PASSED - Scalograms generated successfully")
            sys.exit(0)
        else:
            print("\n❌ CWT Debug Test FAILED - No scalograms generated")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
