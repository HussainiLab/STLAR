#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Batch processing script for Spatial Spectral Mapper (SSM)
Allows command-line analysis of LFP data without GUI
"""

import os
import sys
import argparse
import csv
import numpy as np
from pathlib import Path
import gc  # Garbage collection for memory management
import psutil  # For memory monitoring
import subprocess

# Set matplotlib to non-interactive backend BEFORE importing anything else
import matplotlib
from core.processors.spectral_functions import export_binned_analysis_to_csv, visualize_binned_analysis, export_binned_analysis_jpgs
matplotlib.use('Agg')  # Use Agg backend - no display needed
from matplotlib import pyplot as plt
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'  # Set Qt to offscreen mode

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Now import Qt with QGuiApplication (offscreen mode)
from PyQt5.QtGui import QGuiApplication
app = QGuiApplication(sys.argv)  # Create app in offscreen mode

from src.initialize_fMap import initialize_fMap
from src.core.data_loaders import grab_position_data
from src.core.processors.Tint_Matlab import speed2D
from src.core.processors.spectral_functions import speed_bins


class BatchWorker:
    """Worker class for batch processing without GUI"""
    
    class MockSignals:
        """Mock signal object for batch processing (replaces Qt signals)"""
        def __init__(self):
            self.progress_value = 0
            self.progress_text = ""
        
        def emit(self, value=None):
            """Mock signal emit - just log the value"""
            if isinstance(value, str):
                self.progress_text = value
                print(f"  → {value}")
            else:
                self.progress_value = value
    
    def __init__(self, output_dir=None):
        self.output_dir = output_dir
        self.progress_messages = []
        self.signals = self.MockSignals()
        
        # Create nested structure for mock signals
        class ProgressSignals:
            def __init__(self, worker_signals):
                self.worker_signals = worker_signals
            
            def emit(self, value):
                if isinstance(value, str):
                    self.worker_signals.progress_text = value
                    print(f"  → {value}")
                else:
                    self.worker_signals.progress_value = value
        
        class TextProgressSignals:
            def __init__(self, worker_signals):
                self.worker_signals = worker_signals
            
            def emit(self, value):
                self.worker_signals.progress_text = value
                print(f"  → {value}")
        
        self.signals.progress = ProgressSignals(self.signals)
        self.signals.text_progress = TextProgressSignals(self.signals)
    
    def log(self, message):
        """Log progress messages"""
        print(f"[SSM] {message}")
        self.progress_messages.append(message)


def find_pos_file(eeg_file):
    """Auto-detect .pos file based on .eeg/.egf filename"""
    base_name = os.path.splitext(eeg_file)[0]
    pos_file = base_name + '.pos'
    
    if os.path.exists(pos_file):
        return pos_file
    else:
        raise FileNotFoundError(f"Position file not found: {pos_file}")


def find_recording_files(directory):
    """
    Find all EEG/EGF recording files in a directory.
    Returns list of (egf/eeg file, pos file) tuples.
    
    Priority: Use .egf if exists, otherwise .eeg
    Skip numbered variants (egf2-4, eeg2-4) if base file exists
    """
    import glob
    from pathlib import Path
    
    recordings = {}  # basename -> (electrophys_file, pos_file)
    
    # Find all .egf and .eeg files
    egf_files = glob.glob(os.path.join(directory, "*.egf"))
    eeg_files = glob.glob(os.path.join(directory, "*.eeg"))
    
    # Process EGF files first (higher priority)
    for egf_file in egf_files:
        basename = Path(egf_file).stem
        # Skip numbered variants (egf2, egf3, egf4)
        if basename.endswith(('2', '3', '4')) and basename[:-1] in [Path(f).stem for f in egf_files]:
            continue
        
        try:
            pos_file = find_pos_file(egf_file)
            recordings[basename] = (egf_file, pos_file)
        except FileNotFoundError:
            print(f"⚠ Skipping {os.path.basename(egf_file)} - no .pos file found")
    
    # Process EEG files (only if no EGF exists for same basename)
    for eeg_file in eeg_files:
        basename = Path(eeg_file).stem
        
        # Skip if already have EGF for this basename
        if basename in recordings:
            continue
        
        # Skip numbered variants (eeg2, eeg3, eeg4)
        if basename.endswith(('2', '3', '4')) and basename[:-1] in [Path(f).stem for f in eeg_files]:
            continue
        
        try:
            pos_file = find_pos_file(eeg_file)
            recordings[basename] = (eeg_file, pos_file)
        except FileNotFoundError:
            print(f"⚠ Skipping {os.path.basename(eeg_file)} - no .pos file found")
    
    return list(recordings.values())


def find_eoi_file(electrophys_file):
    """Find associated EOI file (CSV or HFOScores)"""
    base_path = os.path.splitext(electrophys_file)[0]
    base_name = os.path.basename(base_path)
    parent_dir = os.path.dirname(electrophys_file)
    
    # 1. Check for {basename}_EOI.csv
    csv_candidate = f"{base_path}_EOI.csv"
    if os.path.exists(csv_candidate):
        return csv_candidate
        
    # 2. Check for HFOScores/{basename}/{basename}_HIL.txt (or similar)
    # Try standard HFO detection outputs
    for tag in ['HIL', 'STE', 'MNI', 'DL', 'CON']:
        score_path = os.path.join(parent_dir, 'HFOScores', base_name, f"{base_name}_{tag}.txt")
        if os.path.exists(score_path):
            return score_path
            
    return None

def load_eois(eoi_file):
    """Load EOIs from CSV or Score file. Returns list of (start, stop) in seconds."""
    eois = []
    try:
        with open(eoi_file, 'r') as f:
            reader = csv.reader(f, delimiter='\t' if eoi_file.endswith('.txt') else ',')
            for row in reader:
                if not row: continue
                # Skip header if present (heuristic: first col is string "ID" or similar)
                if "ID" in row[0] or "Start" in row[0]: continue
                
                try:
                    # HFO Score format: ID, Start(ms), Stop(ms), ...
                    # CSV format: Start(s), Stop(s) OR Start(ms), Stop(ms) (heuristic)
                    
                    # Assume HFO Score format (ms) if 3+ columns and file is .txt
                    if eoi_file.endswith('.txt') and len(row) >= 3:
                        start = float(row[1]) / 1000.0
                        stop = float(row[2]) / 1000.0
                        eois.append((start, stop))
                    elif len(row) >= 2:
                        # Assume seconds for simple CSV
                        start = float(row[0])
                        stop = float(row[1])
                        eois.append((start, stop))
                except (ValueError, IndexError):
                    continue
    except Exception as e:
        print(f"  ⚠ Error loading EOIs from {eoi_file}: {e}")
    return eois

def export_to_csv(output_path, pos_t, chunk_powers_data, chunk_size, ppm, 
                  pos_x_chunks, pos_y_chunks):
    """Export analysis results to Excel (or CSV fallback)"""
    
    # Try to use Excel; fallback to CSV
    try:
        import openpyxl
        use_excel = True
    except ImportError:
        use_excel = False
    
    bands = [
        ("Delta", "Avg Delta Power"),
        ("Theta", "Avg Theta Power"),
        ("Beta", "Avg Beta Power"),
        ("Low Gamma", "Avg Low Gamma Power"),
        ("High Gamma", "Avg High Gamma Power"),
        ("Ripple", "Avg Ripple Power"),
        ("Fast Ripple", "Avg Fast Ripple Power"),
    ]
    
    band_labels = [label for _, label in bands]
    percent_labels = [f"Percent {name}" for name, _ in bands]
    
    # Calculate distances
    distances_per_bin = []
    cumulative_distances = []
    cumulative_sum = 0.0
    
    if pos_x_chunks is not None and pos_y_chunks is not None:
        for i in range(len(pos_x_chunks)):
            distance_cm_in_bin = 0.0
            
            x_bin = pos_x_chunks[i]
            y_bin = pos_y_chunks[i]
            
            if len(x_bin) > 1:
                dx = np.diff(np.array(x_bin))
                dy = np.diff(np.array(y_bin))
                distances_in_bin_pixels = np.sqrt(dx**2 + dy**2)
                total_distance_pixels_in_bin = np.sum(distances_in_bin_pixels)
                
                if ppm is not None and ppm > 0:
                    distance_cm_in_bin = (total_distance_pixels_in_bin / ppm) * 100
                else:
                    distance_cm_in_bin = total_distance_pixels_in_bin
            
            distances_per_bin.append(distance_cm_in_bin)
            cumulative_sum += distance_cm_in_bin
            cumulative_distances.append(cumulative_sum)
    
    # Determine max full chunks
    actual_duration = float(pos_t[-1])
    max_full_chunks = int(actual_duration / chunk_size)
    num_rows = min(len(pos_t), max_full_chunks)
    
    # Gather per-band arrays
    band_arrays = {}
    for key, label in bands:
        arr = np.array(chunk_powers_data.get(key, [])).reshape(-1)
        band_arrays[label] = arr
    
    # Prepare header and data rows
    header = ["Time Bin (s)", "Distance Per Bin (cm)", "Cumulative Distance (cm)"] + band_labels + percent_labels
    data_rows = []
    
    for i in range(num_rows):
        time_bin_start = i * chunk_size
        time_bin_end = (i + 1) * chunk_size
        time_bin_str = f"{time_bin_start}-{time_bin_end}"
        
        row = [time_bin_str]
        
        # Add distance per bin
        if distances_per_bin and i < len(distances_per_bin):
            row.append(round(distances_per_bin[i], 3))
        else:
            row.append("")
        
        # Add cumulative distance
        if cumulative_distances and i < len(cumulative_distances):
            row.append(round(cumulative_distances[i], 3))
        else:
            row.append("")
        
        band_values = []
        for _, label in bands:
            val = band_arrays[label][i] if i < len(band_arrays[label]) else ""
            band_values.append(val)
        
        row.extend(band_values)
        
        numeric_vals = [float(v) for v in band_values if v != ""]
        total_power = sum(numeric_vals)
        
        for v in band_values:
            if v == "" or total_power == 0:
                row.append("")
            else:
                row.append(round((float(v) / total_power) * 100.0, 3))
        
        data_rows.append(row)
    
    # Write Excel or CSV
    if use_excel:
        output_file = output_path.replace('.csv', '.xlsx')
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = 'SSM Data'
        ws.append(header)
        for row in data_rows:
            ws.append(row)
        wb.save(output_file)
        print(f"✓ Excel exported to: {output_file}")
    else:
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for row in data_rows:
                writer.writerow(row)
        print(f"✓ CSV exported to: {output_path} (openpyxl not installed)")


def main():
    parser = argparse.ArgumentParser(
        description="Batch process LFP data with Spatial Spectral Mapper (SSM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single file
  python batch_ssm.py data/recording.eeg
  
  # Process all files in a directory
  python batch_ssm.py data/recordings/
  
  # With custom parameters
  python batch_ssm.py data/recording.eeg --ppm 500 --chunk-size 5 --speed-filter 5,20
  
  # Batch directory with custom output
  python batch_ssm.py data/recordings/ --ppm 600 -o results/
        """
    )
    
    parser.add_argument(
        "input_path",
        help="Path to .eeg/.egf file OR directory containing multiple recordings"
    )
    
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output directory for CSV file(s) (default: same directory as input file(s))"
    )
    
    parser.add_argument(
        "--ppm",
        type=int,
        default=600,
        help="Pixels per meter for position data (default: 600)"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=60,
        help="Chunk size in seconds for analysis (default: 60)"
    )
    
    parser.add_argument(
        "--speed-filter",
        type=str,
        default="0,100",
        help="Speed filter range in cm/s (format: min,max, default: 0,100)"
    )
    
    parser.add_argument(
        "--window",
        type=str,
        default="hann",
        help="Window type for FFT (default: hann)"
    )
    
    parser.add_argument(
        "--export-binned-jpgs",
        action="store_true",
        help="Export per-chunk JPG visualizations for binned analysis (mean power, percent power, dominant band per chunk; occupancy once)"
    )
    
    parser.add_argument(
        "--export-binned-csvs",
        action="store_true",
        help="Export CSV/Excel summaries for binned analysis"
    )
    
    args = parser.parse_args()
    
    # Parse speed filter
    try:
        speed_range = args.speed_filter.split(',')
        low_speed = float(speed_range[0])
        high_speed = float(speed_range[1])
    except (ValueError, IndexError):
        print(f"✗ Error: Invalid speed filter format. Use 'min,max' (e.g., '5,20')")
        sys.exit(1)
    
    # Check if input is a directory or file
    if os.path.isdir(args.input_path):
        # Directory mode - process all recordings
        print(f"\n{'='*60}")
        print("Spatial Spectral Mapper - Batch Directory Processing")
        print(f"{'='*60}")
        print(f"Directory: {args.input_path}")
        print(f"Parameters: PPM={args.ppm}, Chunk={args.chunk_size}s, Speed={low_speed}-{high_speed} cm/s")
        print(f"{'='*60}\n")
        
        recordings = find_recording_files(args.input_path)
        
        if not recordings:
            print("✗ No valid recording files found in directory.")
            sys.exit(1)
        
        print(f"Found {len(recordings)} recording(s) to process\n")
        
        success_count = 0
        fail_count = 0
        
        for idx, (electrophys_file, pos_file) in enumerate(recordings, 1):
            print(f"\n[{idx}/{len(recordings)}] Processing: {os.path.basename(electrophys_file)}")
            print("-" * 60)
            
            try:
                process_single_file(
                    electrophys_file, 
                    pos_file, 
                    args.output or os.path.dirname(electrophys_file),
                    args.ppm,
                    args.chunk_size,
                    low_speed,
                    high_speed,
                    args.window
                )
                success_count += 1
            except Exception as e:
                print(f"✗ Failed: {e}")
                fail_count += 1
        
        print(f"\n{'='*60}")
        print(f"Batch processing complete!")
        print(f"✓ Successful: {success_count}")
        if fail_count > 0:
            print(f"✗ Failed: {fail_count}")
        print(f"{'='*60}")
        
    else:
        # Single file mode
        if not os.path.exists(args.input_path):
            print(f"✗ Error: File not found: {args.input_path}")
            sys.exit(1)
        
        try:
            pos_file = find_pos_file(args.input_path)
        except FileNotFoundError as e:
            print(f"✗ Error: {e}")
            sys.exit(1)
        
        output_dir = args.output or os.path.dirname(args.input_path) or "."
        
        print(f"\n{'='*60}")
        print("Spatial Spectral Mapper - Single File Processing")
        print(f"{'='*60}")
        
        try:
            process_single_file(
                args.input_path, 
                pos_file, 
                output_dir,
                args.ppm,
                args.chunk_size,
                low_speed,
                high_speed,
                args.window,
                    export_binned_jpgs=args.export_binned_jpgs,
                    export_binned_csvs=args.export_binned_csvs
            )
        except Exception as e:
            print(f"\n✗ Error during processing: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Processing timeout exceeded")

def process_single_file(electrophys_file, pos_file, output_dir, ppm, chunk_size, 
                        low_speed, high_speed, window_type, export_binned_jpgs=False, export_binned_csvs=False, timeout_seconds=300):
    """Process a single EEG/EGF file with timeout protection"""
    
    # Monitor memory at start
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # Generate output filename
    base_name = os.path.splitext(os.path.basename(electrophys_file))[0]
    output_csv = os.path.join(output_dir, f"{base_name}_SSM.csv")
    
    print(f"Input:  {electrophys_file}")
    print(f"Output: {output_csv}")
    print(f"PPM: {ppm}, Chunk: {chunk_size}s, Speed: {low_speed}-{high_speed} cm/s")
    print(f"Memory: {mem_before:.1f} MB")
    
    
    # Create batch worker with mock signals
    worker = BatchWorker(output_dir)
    
    print("[1/5] Starting data processing...")
    print(f"  Timeout set to {timeout_seconds} seconds")

    try:
        # Run initialization with worker as 'self' parameter
        result = initialize_fMap(
            worker,
            files=[pos_file, electrophys_file],
            ppm=ppm,
            chunk_size=chunk_size,
            window_type=window_type,
            low_speed=low_speed,
            high_speed=high_speed
        )
        print("[2/5] Data processing complete")
    except Exception as e:
        print(f"\n✗ Error during data processing: {e}")
        import traceback
        traceback.print_exc()
        raise

    
    
    # Unpack legacy/new result signature
    if len(result) == 6:
        freq_maps, plot_data, pos_t, scaling_factor_crossband, chunk_pows_data, tracking_data = result
        binned_data = None
    elif len(result) == 7:
        freq_maps, plot_data, pos_t, scaling_factor_crossband, chunk_pows_data, tracking_data, binned_data = result
    else:
        freq_maps, plot_data, pos_t, scaling_factor_crossband, chunk_pows_data, tracking_data, binned_data, arena_shape = result
    
    print("[3/5] Extracting tracking data...")
    # Extract tracking data
    if tracking_data:
        # Handle 3-tuple (x, y, t) or 2-tuple (x, y)
        pos_x_chunks = tracking_data[0]
        pos_y_chunks = tracking_data[1]
    else:
        pos_x_chunks, pos_y_chunks = None, None
    
    # Process EOIs if available
    eoi_segments = {}
    eoi_file = find_eoi_file(electrophys_file)
    if eoi_file and tracking_data:
        print(f"[3.5/5] Processing EOIs from {os.path.basename(eoi_file)}...")
        eois = load_eois(eoi_file)
        if eois:
            # Map EOIs to spatial segments
            # tracking_data is (x_chunks, y_chunks, t_chunks)
            if len(tracking_data) == 3:
                x_chunks, y_chunks, t_chunks = tracking_data
                for i, (chunk_x, chunk_y, chunk_t) in enumerate(zip(x_chunks, y_chunks, t_chunks)):
                    if len(chunk_t) == 0: continue
                    for start, stop in eois:
                        # Find indices where time is within EOI
                        # Use searchsorted for speed
                        idx_start = np.searchsorted(chunk_t, start)
                        idx_end = np.searchsorted(chunk_t, stop)
                        
                        if idx_end > idx_start:
                            if i not in eoi_segments: eoi_segments[i] = []
                            eoi_segments[i].append((chunk_x[idx_start:idx_end], chunk_y[idx_start:idx_end]))
            print(f"  → Mapped {len(eois)} EOIs to {sum(len(v) for v in eoi_segments.values())} spatial segments")
    elif eoi_file:
        print(f"  ⚠ EOI file found but no tracking data available.")

    
    print("[4/5] Exporting to CSV...")
    # Export to CSV
    export_to_csv(
        output_csv,
        pos_t,
        chunk_pows_data,
        chunk_size,
        ppm,
        pos_x_chunks,
        pos_y_chunks
    )
    print("[5/5] SSM data export complete")

    # Export binned analysis outputs if available
    try:
        if binned_data is None:
            pass
        elif not (export_binned_jpgs or export_binned_csvs):
            print("  → Binned data available but neither --export-binned-jpgs nor --export-binned-csvs set; skipping binned export.")
        else:
            output_folder = os.path.join(output_dir, f"{base_name}_binned_analysis")
            os.makedirs(output_folder, exist_ok=True)
            
            if binned_data.get('type') == 'polar':
                # Export polar binned analysis in batch: produce JPG visualizations and CSV summaries
                print("  → Polar binned analysis detected.")

                bands = binned_data['bands']
                n_chunks = binned_data['time_chunks']

                # Export per-chunk polar power and percent-power (2x8 grids per band)
                from matplotlib.ticker import FuncFormatter
                theta = np.linspace(-np.pi, np.pi, 9)
                r = [0, 1.0/np.sqrt(2), 1]
                T, R = np.meshgrid(theta, r)

                if export_binned_jpgs:
                    print("  → Exporting polar JPGs...")
                    export_count = 0
                    
                    # Setup grid
                    theta = np.linspace(-np.pi, np.pi, 9)
                    r = [0, 1.0/np.sqrt(2), 1]
                    T, R = np.meshgrid(theta, r)
                    
                    for chunk_idx in range(n_chunks):
                        # Power plots
                        fig, axes = plt.subplots(2, 4, figsize=(16, 8), subplot_kw={'projection': 'polar'})
                        fig.suptitle(f'Polar Bins - Chunk {chunk_idx + 1} (Frequency Band Power)', fontsize=14, fontweight='bold')
                        for idx, band in enumerate(bands):
                            if idx >= 8: break
                            ax = axes.flatten()[idx]
                            data = binned_data['bin_power_timeseries'][band][:, :, chunk_idx]
                            im = ax.pcolormesh(T, R, data, cmap='turbo', shading='flat')
                            ax.set_title(band)
                            ax.set_yticklabels([])
                            ax.set_xticklabels([])
                            ax.grid(True, alpha=0.3)
                            cbar = plt.colorbar(im, ax=ax, pad=0.05, shrink=0.8)
                            cbar.set_label('Power', fontsize=8)
                        for idx in range(len(bands), 8):
                            axes.flatten()[idx].axis('off')
                        plt.tight_layout()
                        jpg_path = os.path.join(output_folder, f"{base_name}_chunk{chunk_idx+1:02d}_polar_power.jpg")
                        fig.savefig(jpg_path, format='jpg', pil_kwargs={'quality':85}, bbox_inches='tight')
                        plt.close(fig)
                        export_count += 1

                        # Percent power plots
                        fig, axes = plt.subplots(2, 4, figsize=(16, 8), subplot_kw={'projection': 'polar'})
                        fig.suptitle(f'Polar Bins - Chunk {chunk_idx + 1} (Frequency Band Percent Power)', fontsize=14, fontweight='bold')
                        total = np.zeros((2,8))
                        for band in bands:
                            total += binned_data['bin_power_timeseries'][band][:, :, chunk_idx]
                        for idx, band in enumerate(bands):
                            if idx >= 8: break
                            ax = axes.flatten()[idx]
                            band_power = binned_data['bin_power_timeseries'][band][:, :, chunk_idx]
                            with np.errstate(divide='ignore', invalid='ignore'):
                                pct = np.where(total>0, (band_power/total)*100.0, 0.0)
                            im = ax.pcolormesh(T, R, pct, cmap='turbo', shading='flat', vmin=0, vmax=100)
                            ax.set_title(band)
                            ax.set_yticklabels([])
                            ax.set_xticklabels([])
                            ax.grid(True, alpha=0.3)
                            cbar = plt.colorbar(im, ax=ax, pad=0.05, shrink=0.8)
                            cbar.set_label('%', fontsize=8)
                        for idx in range(len(bands), 8):
                            axes.flatten()[idx].axis('off')
                        plt.tight_layout()
                        jpg_path = os.path.join(output_folder, f"{base_name}_chunk{chunk_idx+1:02d}_polar_percent.jpg")
                        fig.savefig(jpg_path, format='jpg', pil_kwargs={'quality':85}, bbox_inches='tight')
                        plt.close(fig)
                        export_count += 1

                    # Occupancy
                    fig_occ, ax_occ = plt.subplots(figsize=(8,6), subplot_kw={'projection':'polar'})
                    occ = binned_data['bin_occupancy']
                    im_occ = ax_occ.pcolormesh(T, R, occ, cmap='turbo', shading='flat')
                    ax_occ.set_title('Bin Occupancy')
                    jpg_path = os.path.join(output_folder, f"{base_name}_polar_occupancy.jpg")
                    fig_occ.savefig(jpg_path, format='jpg', pil_kwargs={'quality':85}, bbox_inches='tight')
                    plt.close(fig_occ)
                    export_count += 1
                    
                    print(f"  ✓ Exported {export_count} polar JPG(s)")

                if export_binned_csvs:
                    print("  → Exporting polar data (CSV/Excel)...")
                    output_prefix = os.path.join(output_folder, f"{base_name}_binned")
                    
                    # Helper to reshape (2, 8, n_chunks) -> (n_chunks, 16) and add Chunk column
                    def prepare_matrix(data_3d):
                        flat = data_3d.reshape(16, -1).T
                        chunks_col = np.arange(1, n_chunks + 1).reshape(-1, 1)
                        return np.hstack([chunks_col, flat])

                    # 1. Mean Power (per chunk)
                    power_per_chunk = {}
                    for band in bands:
                        power_per_chunk[band] = prepare_matrix(binned_data['bin_power_timeseries'][band])

                    # 2. Percent Power (per chunk)
                    percent_per_chunk = {}
                    all_bands_power = np.stack([binned_data['bin_power_timeseries'][b] for b in bands])
                    total_power = np.sum(all_bands_power, axis=0) # (2, 8, n_chunks)
                    for band in bands:
                        with np.errstate(divide='ignore', invalid='ignore'):
                            pct_data = np.where(total_power > 0, (binned_data['bin_power_timeseries'][band] / total_power) * 100.0, 0.0)
                        percent_per_chunk[band] = prepare_matrix(pct_data)

                    # 3. EOIs (per chunk)
                    eoi_export_matrix = None
                    if eoi_segments:
                        eoi_counts_per_chunk = np.zeros((2, 8, n_chunks))
                        all_x = np.concatenate(pos_x_chunks)
                        all_y = np.concatenate(pos_y_chunks)
                        min_x, max_x = np.min(all_x), np.max(all_x)
                        min_y, max_y = np.min(all_y), np.max(all_y)
                        width = max_x - min_x
                        height = max_y - min_y
                        if width == 0: width = 1
                        if height == 0: height = 1

                        for chunk_idx in range(n_chunks):
                            if chunk_idx in eoi_segments:
                                for seg in eoi_segments[chunk_idx]:
                                    nx = 2 * (np.array(seg[0]) - min_x) / width - 1
                                    ny = 2 * (np.array(seg[1]) - min_y) / height - 1
                                    r = np.sqrt(nx**2 + ny**2)
                                    theta = np.arctan2(ny, nx)
                                    
                                    equal_area_radius = 1.0 / np.sqrt(2.0)
                                    r_bins = [0, equal_area_radius, np.inf]
                                    r_indices = np.clip(np.digitize(r, r_bins) - 1, 0, 1)
                                    
                                    theta_edges = np.linspace(-np.pi, np.pi, 9)
                                    th_indices = np.clip(np.digitize(theta, theta_edges) - 1, 0, 7)
                                    
                                    for ri, ti in zip(r_indices, th_indices):
                                        eoi_counts_per_chunk[ri, ti, chunk_idx] += 1
                        eoi_export_matrix = prepare_matrix(eoi_counts_per_chunk)

                    # 4. Percent Occupancy (per chunk)
                    occupancy_export_matrix = None
                    if 'bin_occupancy_timeseries' in binned_data:
                        occ_ts = binned_data['bin_occupancy_timeseries']
                        occ_sums = np.sum(occ_ts, axis=(0, 1))
                        with np.errstate(divide='ignore', invalid='ignore'):
                            occ_pct = np.where(occ_sums[None, None, :] > 0, (occ_ts / occ_sums[None, None, :]) * 100.0, 0.0)
                        occupancy_export_matrix = prepare_matrix(occ_pct)

                    # 5. Dominant Band (per chunk)
                    dom_data = np.array(binned_data['bin_dominant_band']) # (n_chunks, 2, 8)
                    dom_data_T = dom_data.transpose(1, 2, 0) # (2, 8, n_chunks)
                    dom_export_matrix = prepare_matrix(dom_data_T)

                    try:
                        import openpyxl
                        bin_headers = ["Chunk"] + [f"Inner_S{i+1}" for i in range(8)] + [f"Outer_S{i+1}" for i in range(8)]
                        
                        # Helper to save multi-sheet workbook
                        def save_multi_sheet(data_dict, filename):
                            wb = openpyxl.Workbook()
                            wb.remove(wb.active)
                            for sheet_name, matrix in data_dict.items():
                                ws = wb.create_sheet(title=sheet_name)
                                ws.append(bin_headers)
                                for row in matrix:
                                    ws.append([float(val) for val in row])
                            wb.save(filename)

                        # Helper to save single-sheet workbook
                        def save_single_sheet(matrix, filename, sheet_title, is_string=False):
                            wb = openpyxl.Workbook()
                            ws = wb.active
                            ws.title = sheet_title
                            ws.append(bin_headers)
                            for row in matrix:
                                if is_string:
                                    ws.append([row[0]] + list(row[1:]))
                                else:
                                    ws.append([float(val) for val in row])
                            wb.save(filename)

                        save_multi_sheet(power_per_chunk, f"{output_prefix}_mean_power_per_chunk.xlsx")
                        save_multi_sheet(percent_per_chunk, f"{output_prefix}_percent_power_per_chunk.xlsx")
                        
                        if eoi_export_matrix is not None:
                            save_single_sheet(eoi_export_matrix, f"{output_prefix}_eois_per_chunk.xlsx", "EOIs")
                        
                        if occupancy_export_matrix is not None:
                            save_single_sheet(occupancy_export_matrix, f"{output_prefix}_percent_occupancy_per_chunk.xlsx", "Percent Occupancy")
                            
                        save_single_sheet(dom_export_matrix, f"{output_prefix}_dominant_band_per_chunk.xlsx", "Dominant Band", is_string=True)

                        print(f"  ✓ Excel files exported to: {output_folder}")
                        
                    except ImportError:
                        # CSV Fallback
                        for band in bands:
                            np.savetxt(f"{output_prefix}_mean_power_{band}.csv", power_per_chunk[band], delimiter=',')
                            np.savetxt(f"{output_prefix}_percent_power_{band}.csv", percent_per_chunk[band], delimiter=',')
                        if eoi_export_matrix is not None:
                            np.savetxt(f"{output_prefix}_eois.csv", eoi_export_matrix, delimiter=',')
                        if occupancy_export_matrix is not None:
                            np.savetxt(f"{output_prefix}_percent_occupancy.csv", occupancy_export_matrix, delimiter=',')
                        # Dominant band CSV might be tricky with strings if using np.savetxt with default fmt
                        np.savetxt(f"{output_prefix}_dominant_band.csv", dom_export_matrix, delimiter=',', fmt='%s')
                        print(f"  ✓ CSVs exported to: {output_folder}")

            else:
                # Rectangular (4x4)
                output_prefix = os.path.join(output_folder, base_name + "_binned")
                
                if export_binned_csvs:
                    print("  → Exporting binned data (CSV/Excel)...")
                    result = export_binned_analysis_to_csv(binned_data, output_prefix, pos_x_chunks=pos_x_chunks, pos_y_chunks=pos_y_chunks, eoi_segments=eoi_segments)
                    if result.get('format') == 'csv' and result.get('reason') == 'openpyxl_not_installed':
                        print("  ⚠ Excel export unavailable (openpyxl missing). Exported CSVs.")
                    else:
                        print(f"  ✓ Exported {result.get('format').upper()} files")
                
                if export_binned_jpgs:
                    print("  → Exporting binned JPGs...")
                    try:
                        jpg_count = export_binned_analysis_jpgs(binned_data, output_folder, base_name)
                        visualize_binned_analysis(binned_data, save_path=os.path.join(output_folder, f"{base_name}_binned_heatmap.jpg"))
                        print(f"  ✓ Exported {jpg_count} JPGs + Heatmap")
                    except Exception as e:
                        print(f"  ⚠ Failed to export JPGs: {e}")
    except Exception as e:
        print(f"⚠ Binned analysis export failed: {e}")
    
    # Print summary
    print(f"✓ Complete - {len(pos_t)} time bins processed")
    for band, power in scaling_factor_crossband.items():
        print(f"  {band:15s}: {power*100:6.2f}%")
    
    # Explicitly delete large objects to free memory
    del freq_maps, plot_data, pos_t, chunk_pows_data, tracking_data
    del pos_x_chunks, pos_y_chunks, worker
    
    # Force garbage collection to clear memory before next file
    gc.collect()
    
    # Monitor memory after cleanup
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    print(f"Memory after cleanup: {mem_after:.1f} MB (freed {mem_before - mem_after:.1f} MB)")


if __name__ == "__main__":
    main()
