#!/usr/bin/env python
"""Debug script for CLI testing"""
from pathlib import Path
from types import SimpleNamespace
import sys
import traceback

sys.path.insert(0, str(Path(__file__).parent))

import hfoGUI.cli as cli

args = SimpleNamespace(
    file=r"E:\DATA\Ephys\STLAR-CLI-Test\Test\20160908-2-NO-3700.egf",
    set_file=r"E:\DATA\Ephys\STLAR-CLI-Test\Test\20160908-2-NO-3700.set",
    output=r"E:\DATA\Ephys\STLAR-CLI-Test\Test\HIL.txt",
    epoch_sec=300.0,
    threshold_sd=5.0,
    min_duration_ms=10.0,
    min_freq=None,
    max_freq=None,
    required_peaks=6,
    required_peak_threshold_sd=4.0,
    no_required_peak_threshold=False,
    boundary_percent=30.0,
    skip_bits2uv=True,
    verbose=True,
)

try:
    print("Starting batch job...")
    cli._run_batch_job(args, cli._process_single_file)
    print("Completed successfully!")
except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()

# Check if files exist
out_file = Path(r"E:\DATA\Ephys\STLAR-CLI-Test\Test\HIL.txt")
settings_file = Path(r"E:\DATA\Ephys\STLAR-CLI-Test\Test\HIL_settings.json")
print(f"\nHIL.txt exists: {out_file.exists()}")
print(f"HIL_settings.json exists: {settings_file.exists()}")

if out_file.exists():
    print(f"HIL.txt size: {out_file.stat().st_size} bytes")
if settings_file.exists():
    print(f"HIL_settings.json size: {settings_file.stat().st_size} bytes")
