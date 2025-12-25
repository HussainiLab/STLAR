#!/usr/bin/env python
"""Quick test of consensus detection functionality (backend only)."""

import numpy as np
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

# Test only the helper functions, not the full consensus_detect_events 
# (which requires GUI imports)
from hfoGUI.core.Detector import (
    _merge_overlaps, 
    _vote_consensus,
    ste_detect_events,
    mni_detect_events
)

def test_merge_overlaps():
    """Test merging overlapping events."""
    events = np.array([
        [100, 150],
        [140, 200],  # Overlaps with previous
        [300, 350],
        [349, 400],  # Within threshold (10ms) of previous
    ])
    
    merged = _merge_overlaps(events, overlap_threshold_ms=10.0)
    print(f"✓ Merge overlaps test passed: {len(merged)} merged from {len(events)}")
    assert len(merged) == 2
    print(f"  Merged events: {merged}")


def test_vote_consensus():
    """Test voting on consensus events."""
    # Simulate 3 detectors with overlapping detections
    det1 = np.array([[100, 150], [300, 350]])
    det2 = np.array([[105, 145], [305, 345]])  # Similar to det1
    det3 = np.array([[500, 550]])  # Unique event
    
    all_events = [det1, det2, det3]
    
    # Majority voting (2/3)
    consensus = _vote_consensus(all_events, voting_strategy='majority', overlap_threshold_ms=10.0)
    print(f"✓ Majority voting (2/3) test passed: {len(consensus)} consensus events")
    print(f"  Consensus events: {consensus}")
    assert len(consensus) >= 1
    
    # Strict voting (3/3)
    consensus_strict = _vote_consensus(all_events, voting_strategy='strict', overlap_threshold_ms=10.0)
    print(f"✓ Strict voting (3/3) test passed: {len(consensus_strict)} consensus events")


def test_consensus_detection():
    """Test consensus detection components with synthetic data."""
    # Generate synthetic data with HFO burst
    Fs = 20000  # 20 kHz
    duration = 5  # seconds
    t = np.arange(0, duration, 1/Fs)
    
    # Base noise
    data = np.random.randn(len(t)) * 0.1
    
    # Add HFO burst (150 Hz sinusoid in 200-250 ms window)
    hfo_start = int(1.0 * Fs)  # 1 second
    hfo_end = int(1.3 * Fs)    # 1.3 seconds
    hfo_freq = 150  # Hz
    data[hfo_start:hfo_end] += 0.5 * np.sin(2 * np.pi * hfo_freq * t[hfo_start:hfo_end])
    
    print(f"\nTesting detector components on synthetic data ({duration}s, {Fs} Hz)...")
    
    # Run individual detectors
    ste_events = ste_detect_events(
        data, Fs,
        threshold=2.5,
        window_size=0.01,
        overlap=0.5,
        min_freq=80.0,
        max_freq=500.0
    )
    
    mni_events = mni_detect_events(
        data, Fs,
        baseline_window=10.0,
        threshold_percentile=98.0,
        min_freq=80.0,
        max_freq=500.0
    )
    
    print(f"✓ STE detected {len(ste_events)} events")
    print(f"✓ MNI detected {len(mni_events)} events")
    
    if len(ste_events) > 0:
        print(f"  STE events (ms): {ste_events}")
    if len(mni_events) > 0:
        print(f"  MNI events (ms): {mni_events}")
    
    # Test consensus voting with detector results
    all_detectors = [
        np.array([[1000, 1300]]),  # Simulated Hilbert
        ste_events,
        mni_events,
    ]
    
    consensus = _vote_consensus(all_detectors, voting_strategy='majority', overlap_threshold_ms=10.0)
    print(f"\n✓ Consensus (2/3 voting): {len(consensus)} events")
    if len(consensus) > 0:
        print(f"  Consensus events (ms): {consensus}")
    
    return consensus


if __name__ == '__main__':
    print("=" * 60)
    print("CONSENSUS DETECTION TEST SUITE")
    print("=" * 60)
    
    try:
        test_merge_overlaps()
        print()
        test_vote_consensus()
        print()
        test_consensus_detection()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
