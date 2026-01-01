import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .core.Score import hilbert_detect_events
from .core.Detector import ste_detect_events, mni_detect_events, dl_detect_events, consensus_detect_events
from .core.Tint_Matlab import ReadEEG, bits2uV, TintException


def _print_prob_summary(probs):
    probs = np.asarray(probs, dtype=float)
    if probs.size == 0:
        print("[DL Detection] No probability windows to summarize (empty signal or model fallback).")
        return

    pcts = np.percentile(probs, [1, 5, 25, 50, 75, 95, 99])
    spread = probs.max() - probs.min()
    std = probs.std()

    print(f"[DL Detection] Probability stats: windows={len(probs)}, min={probs.min():.4f}, max={probs.max():.4f}, mean={probs.mean():.4f}")
    print("[DL Detection] Percentiles 1,5,25,50,75,95,99:", " ".join(f"{p:.4f}" for p in pcts))

    if spread < 0.10 or std < 0.03:
        verdict = "Very narrow probability spread; model likely undertrained. Retrain with more epochs and ensure good labels."
    elif spread < 0.25:
        verdict = "Moderately narrow spread; add more epochs or harder negatives to improve separation."
    else:
        verdict = "Healthy spread; tweak detection threshold or continue training if quality is still low."
    print(f"[DL Detection] Assessment: {verdict}")

def _default_freqs(data_path: Path, max_freq: Optional[float]) -> Tuple[float, float]:
    """Choose sensible defaults based on file extension."""
    # Defaults mirror the GUI: 80 Hz min, 125 Hz max for EEG, 500 Hz max for EGF
    min_freq = 80.0
    if max_freq is not None:
        return min_freq, max_freq

    if data_path.suffix.lower().startswith('.egf'):
        return min_freq, 500.0

    return min_freq, 125.0


def _build_output_paths(data_path: Path, set_path: Optional[Path], output: Optional[Path], method_tag: str = 'HIL'):
    session_base = set_path.stem if set_path else data_path.stem

    if output:
        output_path = Path(output)
        # If output has a suffix (e.g., .txt), treat it as a file path
        if output_path.suffix:
            scores_path = output_path
            settings_path = output_path.parent / "{}_settings.json".format(output_path.stem)
        else:
            # Treat as directory
            output_dir = output_path
            scores_path = output_dir / "{}_{}.txt".format(session_base, method_tag)
            settings_path = output_dir / "{}_{}_settings.json".format(session_base, method_tag)
    else:
        base_dir = set_path.parent if set_path else data_path.parent
        scores_dir = base_dir / 'HFOScores' / session_base
        scores_path = scores_dir / ("{}_{}.txt".format(session_base, method_tag))
        settings_path = scores_dir / ("{}_{}_settings.json".format(session_base, method_tag))

    scores_path.parent.mkdir(parents=True, exist_ok=True)
    return scores_path, settings_path


def _find_data_files(directory: Path):
    """Find all .eeg and .egf files recursively in a directory.
    
    If both .eeg and .egf exist for the same basename, only return .egf
    since .eeg files typically don't contain HFOs.
    """
    eeg_files = list(directory.rglob('*.eeg'))
    egf_files = list(directory.rglob('*.egf'))
    
    # Create a set of (parent, stem) tuples for .egf files to handle same basenames in same folder
    egf_keys = {(f.parent, f.stem) for f in egf_files}
    
    # Filter out .eeg files that have a corresponding .egf file in the same folder
    filtered_eeg = [f for f in eeg_files if (f.parent, f.stem) not in egf_keys]
    
    return sorted(filtered_eeg + egf_files)


def _find_set_file(data_path: Path):
    """Find corresponding .set file for a data file."""
    set_path = data_path.with_suffix('.set')
    if set_path.exists():
        return set_path
    # Try in parent directory with same stem
    parent_set = data_path.parent / '{}.set'.format(data_path.stem)
    if parent_set.exists():
        return parent_set
    return None


def _load_and_scale_data(data_path: Path, set_path: Optional[Path], args: argparse.Namespace):
    """Helper to load EEG data and convert to uV if possible."""
    raw_data, Fs = ReadEEG(str(data_path))

    if set_path and set_path.exists() and not args.skip_bits2uv:
        try:
            raw_data, _ = bits2uV(raw_data, str(data_path), str(set_path))
        except TintException as exc:
            if not args.skip_bits2uv:
                raise
            if args.verbose:
                print('  Warning: Proceeding without bits->uV conversion: {}'.format(exc))
    
    return np.asarray(raw_data, dtype=float), Fs


def _process_single_file(data_path: Path, set_path: Optional[Path], args: argparse.Namespace):
    """Process a single data file with Hilbert detection.
    
    Returns:
        int: Number of events detected (for summary reporting).
    """
    if args.verbose:
        print('\nProcessing: {}'.format(data_path))

    raw_data, Fs = _load_and_scale_data(data_path, set_path, args)

    min_freq_default, max_freq_default = _default_freqs(data_path, args.max_freq)
    min_freq = args.min_freq if args.min_freq is not None else min_freq_default
    max_freq = args.max_freq if args.max_freq is not None else max_freq_default

    peak_sd = None if args.no_required_peak_threshold else float(args.required_peak_threshold_sd)

    params = {
        'epoch': float(args.epoch_sec),
        'sd_num': float(args.threshold_sd),
        'min_duration': float(args.min_duration_ms),
        'min_freq': float(min_freq),
        'max_freq': float(max_freq),
        'required_peak_number': int(args.required_peaks),
        'required_peak_sd': peak_sd,
        'boundary_fraction': float(args.boundary_percent) / 100.0,
        'verbose': args.verbose,
    }

    events = hilbert_detect_events(raw_data, Fs, **params)

    return _save_results(events, params, data_path, set_path, args, method_tag='HIL')


def _process_ste_file(data_path: Path, set_path: Optional[Path], args: argparse.Namespace):
    """Process a single data file with STE (RMS) detection."""

    if args.verbose:
        print('\nProcessing (STE): {}'.format(data_path))

    raw_data, Fs = _load_and_scale_data(data_path, set_path, args)

    params = {
        'threshold': float(args.threshold),
        'window_size': float(args.window_size),
        'overlap': float(args.overlap),
        'min_freq': float(args.min_freq) if args.min_freq else 80.0,
        'max_freq': float(args.max_freq) if args.max_freq else 500.0,
    }

    events = ste_detect_events(raw_data, Fs, **params)

    return _save_results(events, params, data_path, set_path, args, method_tag='STE')


def _process_mni_file(data_path: Path, set_path: Optional[Path], args: argparse.Namespace):
    """Process a single data file with MNI detection."""

    if args.verbose:
        print('\nProcessing (MNI): {}'.format(data_path))

    raw_data, Fs = _load_and_scale_data(data_path, set_path, args)

    params = {
        'baseline_window': float(args.baseline_window),
        'threshold_percentile': float(args.threshold_percentile),
        'min_freq': float(args.min_freq) if args.min_freq else 80.0,
    }

    events = mni_detect_events(raw_data, Fs, **params)

    return _save_results(events, params, data_path, set_path, args, method_tag='MNI')


def _process_dl_file(data_path: Path, set_path: Optional[Path], args: argparse.Namespace):
    """Process a single data file with Deep Learning detection."""

    if args.verbose:
        print('\nProcessing (DL): {}'.format(data_path))

    raw_data, Fs = _load_and_scale_data(data_path, set_path, args)

    params = {
        'model_path': args.model_path,
        'threshold': float(args.threshold),
        'batch_size': int(args.batch_size),
    }

    dump_probs = getattr(args, 'dump_probs', False)
    dl_result = dl_detect_events(raw_data, Fs, dump_probs=dump_probs, **params)

    prob_values = None
    if dump_probs:
        events, prob_values = dl_result
        _print_prob_summary(prob_values)
    else:
        events = dl_result

    return _save_results(events, params, data_path, set_path, args, method_tag='DL')


def _process_consensus_file(data_path: Path, set_path: Optional[Path], args: argparse.Namespace):
    """Process a single data file with Consensus detection (Hilbert + STE + MNI)."""

    if args.verbose:
        print('\nProcessing (Consensus): {}'.format(data_path))

    raw_data, Fs = _load_and_scale_data(data_path, set_path, args)

    min_freq_default, max_freq_default = _default_freqs(data_path, args.max_freq)

    hilbert_params = {
        'epoch': float(args.epoch_sec),
        'sd_num': float(args.hilbert_threshold_sd),
        'min_duration': float(args.min_duration_ms),
        'min_freq': float(args.min_freq) if args.min_freq else min_freq_default,
        'max_freq': float(args.max_freq) if args.max_freq else max_freq_default,
        'required_peak_number': int(args.required_peaks),
        'required_peak_sd': float(args.required_peak_sd),
        'boundary_fraction': 0.3,
    }

    ste_params = {
        'threshold': float(args.ste_threshold),
        'window_size': 0.01,
        'overlap': 0.5,
        'min_freq': float(args.min_freq) if args.min_freq else min_freq_default,
        'max_freq': float(args.max_freq) if args.max_freq else max_freq_default,
    }

    mni_params = {
        'baseline_window': 10.0,
        'threshold_percentile': float(args.mni_percentile),
        'min_freq': float(args.min_freq) if args.min_freq else min_freq_default,
        'max_freq': float(args.max_freq) if args.max_freq else max_freq_default,
    }

    params = {
        'voting_strategy': args.voting_strategy,
        'overlap_threshold_ms': float(args.overlap_threshold_ms),
        'hilbert': hilbert_params,
        'ste': ste_params,
        'mni': mni_params,
    }

    events = consensus_detect_events(
        raw_data, Fs,
        hilbert_params=hilbert_params,
        ste_params=ste_params,
        mni_params=mni_params,
        voting_strategy=args.voting_strategy,
        overlap_threshold_ms=float(args.overlap_threshold_ms),
    )

    return _save_results(events, params, data_path, set_path, args, method_tag='CON')



def _save_results(events, params, data_path, set_path, args, method_tag):
    """Helper to save events and settings to disk."""
    # Ensure events is a numpy array
    events = np.asarray(events)
    if events.ndim == 1 and len(events) == 0:
        events = np.empty((0, 2))

    out_path = Path(args.output).expanduser() if args.output else None
    scores_path, settings_path = _build_output_paths(
        data_path,
        set_path if set_path and set_path.exists() else None,
        out_path,
        method_tag=method_tag
    )

    with open(str(settings_path), 'w', encoding='utf-8') as f:
        # Convert params to serializable dict if necessary
        serializable_params = {}
        for k, v in params.items():
            if isinstance(v, Path):
                serializable_params[k] = str(v)
            else:
                serializable_params[k] = v
        json.dump(serializable_params, f, indent=2)

    # Generate IDs
    if len(events) > 0:
        ids = ['{}{}'.format(method_tag, idx + 1) for idx in range(len(events))]
        start_times = events[:, 0]
        stop_times = events[:, 1]
    else:
        ids = []
        start_times = []
        stop_times = []

    df = pd.DataFrame({
        'ID#:': ids,
        'Start Time(ms):': start_times,
        'Stop Time(ms):': stop_times,
        'Settings File:': settings_path.as_posix(),
    })

    df.to_csv(str(scores_path), sep='\t', index=False)

    if args.verbose:
        print('  Saved settings -> {}'.format(settings_path))

    print('  Detected {} events; saved scores -> {}'.format(len(events), scores_path))

    return len(events)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='STLAR command-line utilities')
    sub = parser.add_subparsers(dest='command')

    # --- Hilbert Parser ---
    hilbert = sub.add_parser('hilbert-batch', help='Run Hilbert-based automatic detection headlessly')
    hilbert.add_argument('-f', '--file', required=True, help='Path to .eeg/.egf file or directory to process recursively')
    hilbert.add_argument('-s', '--set-file', help='Optional .set file or directory; defaults to sibling of the data file')
    hilbert.add_argument('-o', '--output', help='Output directory; scores saved as <session>.txt, defaults to HFOScores/<session>/<session>_HIL.txt')
    hilbert.add_argument('--epoch-sec', type=float, default=5 * 60, help='Epoch length in seconds (default: 300)')
    hilbert.add_argument('--threshold-sd', type=float, default=3.0,
                         help='Envelope threshold in SD above mean (default: 3)')
    hilbert.add_argument('--min-duration-ms', type=float, default=10.0, help='Minimum event duration in ms (default: 10)')
    hilbert.add_argument('--min-freq', type=float, help='Minimum bandpass frequency (Hz). Default 80 Hz')
    hilbert.add_argument('--max-freq', type=float, help='Maximum bandpass frequency (Hz). Default 125 Hz for EEG, 500 Hz for EGF')
    hilbert.add_argument('--required-peaks', type=int, default=6,
                         help='Minimum peak count inside rectified signal (default: 6)')
    hilbert.add_argument('--required-peak-threshold-sd', type=float, default=2.0,
                         help='Peak threshold in SD above mean (default: 2). Use --no-required-peak-threshold to disable')
    hilbert.add_argument('--no-required-peak-threshold', action='store_true',
                         help='Disable the peak-threshold SD check')
    hilbert.add_argument('--boundary-percent', type=float, default=30.0,
                         help='Percent of threshold to find boundaries (default: 30)')
    hilbert.add_argument('--skip-bits2uv', action='store_true',
                         help='Skip bits-to-uV conversion if the .set file is missing')
    hilbert.add_argument('-v', '--verbose', action='store_true', help='Verbose progress logging')
    
    # --- STE Parser ---
    ste = sub.add_parser('ste-batch', help='Run Short-Term Energy (RMS) detection')
    ste.add_argument('-f', '--file', required=True, help='Path to .eeg/.egf file or directory')
    ste.add_argument('-s', '--set-file', help='Optional .set file')
    ste.add_argument('-o', '--output', help='Output directory')
    ste.add_argument('--threshold', type=float, default=3.0, help='RMS threshold (SD or absolute)')
    ste.add_argument('--window-size', type=float, default=0.01, help='Window size in seconds')
    ste.add_argument('--overlap', type=float, default=0.5, help='Window overlap fraction')
    ste.add_argument('--min-freq', type=float, help='Min frequency (Hz)')
    ste.add_argument('--max-freq', type=float, help='Max frequency (Hz)')
    ste.add_argument('--skip-bits2uv', action='store_true', help='Skip bits-to-uV conversion')
    ste.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')

    # --- MNI Parser ---
    mni = sub.add_parser('mni-batch', help='Run MNI detection')
    mni.add_argument('-f', '--file', required=True, help='Path to .eeg/.egf file or directory')
    mni.add_argument('-s', '--set-file', help='Optional .set file')
    mni.add_argument('-o', '--output', help='Output directory')
    mni.add_argument('--baseline-window', type=float, default=10.0, help='Baseline window in seconds')
    mni.add_argument('--threshold-percentile', type=float, default=99.0, help='Threshold percentile')
    mni.add_argument('--min-freq', type=float, help='Min frequency (Hz)')
    mni.add_argument('--skip-bits2uv', action='store_true', help='Skip bits-to-uV conversion')
    mni.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')

    # --- Consensus Parser ---
    consensus = sub.add_parser('consensus-batch', help='Run Consensus detection (Hilbert + STE + MNI voting)')
    consensus.add_argument('-f', '--file', required=True, help='Path to .eeg/.egf file or directory to process recursively')
    consensus.add_argument('-s', '--set-file', help='Optional .set file or directory')
    consensus.add_argument('-o', '--output', help='Output directory')
    consensus.add_argument('--epoch-sec', type=float, default=5 * 60, help='Hilbert epoch length in seconds (default: 300)')
    consensus.add_argument('--hilbert-threshold-sd', type=float, default=3.5, help='Hilbert threshold in SD (default: 3.5)')
    consensus.add_argument('--ste-threshold', type=float, default=2.5, help='STE threshold in RMS (default: 2.5)')
    consensus.add_argument('--mni-percentile', type=float, default=98.0, help='MNI threshold percentile (default: 98)')
    consensus.add_argument('--min-duration-ms', type=float, default=10.0, help='Minimum event duration in ms (default: 10)')
    consensus.add_argument('--min-freq', type=float, help='Minimum bandpass frequency (Hz). Default 80 Hz')
    consensus.add_argument('--max-freq', type=float, help='Maximum bandpass frequency (Hz). Default 125 Hz for EEG, 500 Hz for EGF')
    consensus.add_argument('--required-peaks', type=int, default=6, help='Hilbert minimum peak count (default: 6)')
    consensus.add_argument('--required-peak-sd', type=float, default=2.0, help='Hilbert peak threshold in SD (default: 2.0)')
    consensus.add_argument('--voting-strategy', choices=['strict', 'majority', 'any'], default='majority', 
                          help='Voting rule: strict=3/3, majority=2/3, any=1/3 (default: majority)')
    consensus.add_argument('--overlap-threshold-ms', type=float, default=10.0, help='Overlap window in ms (default: 10)')
    consensus.add_argument('--skip-bits2uv', action='store_true', help='Skip bits-to-uV conversion')
    consensus.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')

    # --- Deep Learning Parser ---
    dl = sub.add_parser('dl-batch', help='Run Deep Learning detection')
    dl.add_argument('-f', '--file', required=True, help='Path to .eeg/.egf file or directory')
    dl.add_argument('-s', '--set-file', help='Optional .set file')
    dl.add_argument('-o', '--output', help='Output directory')
    dl.add_argument('--model-path', required=True, help='Path to trained model file')
    dl.add_argument('--threshold', type=float, default=0.5, help='Detection probability threshold')
    dl.add_argument('--batch-size', type=int, default=32, help='Inference batch size')
    dl.add_argument('--dump-probs', action='store_true', help='Print per-window probability stats')
    dl.add_argument('--skip-bits2uv', action='store_true', help='Skip bits-to-uV conversion')
    dl.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')

    # --- Prepare DL Training Data Parser ---
    prepare_dl = sub.add_parser('prepare-dl', help='Read EOIs from file and prepare for DL training with region presets')
    prepare_dl.add_argument('--eoi-file', help='Path to EOI file (.txt, .csv with start_ms,stop_ms columns). Required for single-session mode.')
    prepare_dl.add_argument('--egf-file', help='Path to .egf data file for signal. Required for single-session mode.')
    prepare_dl.add_argument('--batch-dir', help='Directory containing subdirectories with .egf and EOI files. Enables batch mode.')
    prepare_dl.add_argument('--set-file', help='Optional .set file for bits-to-uV conversion')
    prepare_dl.add_argument('-o', '--output', help='Output directory for segments and manifest.csv. Required for single-session mode.')
    prepare_dl.add_argument('--region', choices=['LEC', 'Hippocampus', 'MEC'], default='LEC',
                           help='Brain region preset (LEC, Hippocampus, MEC; default: LEC)')
    prepare_dl.add_argument('--pos-file', help='Optional .pos file for behavior gating with speed data')
    prepare_dl.add_argument('--ppm', type=int, help='Pixels-per-millimeter for .pos file (e.g., 595)')
    prepare_dl.add_argument('--prefix', default='seg', help='Prefix for segment filenames (default: seg)')
    prepare_dl.add_argument('--skip-bits2uv', action='store_true', help='Skip bits-to-uV conversion')
    prepare_dl.add_argument('--split-train-val', action='store_true', help='Split manifest into train/val sets')
    prepare_dl.add_argument('--val-fraction', type=float, default=0.2, help='Fraction of data for validation (default: 0.2)')
    prepare_dl.add_argument('--random-seed', type=int, default=42, help='Random seed for reproducible splits (default: 42)')
    prepare_dl.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')

    # --- Train DL Model Parser ---
    train_dl = sub.add_parser('train-dl', help='Train 1D CNN model on prepared data')
    train_dl.add_argument('--train', help='Path to train manifest CSV (single-session mode)')
    train_dl.add_argument('--val', help='Path to val manifest CSV (single-session mode)')
    train_dl.add_argument('--batch-dir', help='Directory with subdirectories containing manifest_train.csv and manifest_val.csv (batch mode)')
    train_dl.add_argument('--epochs', type=int, default=15, help='Number of training epochs (default: 15)')
    train_dl.add_argument('--batch-size', type=int, default=64, help='Training batch size (default: 64)')
    train_dl.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default: 1e-3)')
    train_dl.add_argument('--weight-decay', type=float, default=1e-4, help='L2 regularization (default: 1e-4)')
    train_dl.add_argument('--out-dir', type=str, default='models', help='Output directory for checkpoints (default: models)')
    train_dl.add_argument('--num-workers', type=int, default=2, help='DataLoader workers (default: 2)')
    train_dl.add_argument('--model-type', type=int, default=2, help='Model architecture: 1=SimpleCNN, 2=ResNet1D, 3=InceptionTime, 4=Transformer, 5=2D_CNN (default: 2)')
    train_dl.add_argument('--no-plot', action='store_true', help='Disable saving training curve plots')
    train_dl.add_argument('--gui', action='store_true', help='Show real-time training GUI with live plots')
    train_dl.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')

    # --- Export DL Model Parser ---
    export_dl = sub.add_parser('export-dl', help='Export trained model to TorchScript and ONNX')
    export_dl.add_argument('--ckpt', help='Path to best.pt checkpoint file (single-session mode)')
    export_dl.add_argument('--batch-dir', help='Directory with subdirectories containing best.pt checkpoints (batch mode)')
    export_dl.add_argument('--onnx', help='Output path for ONNX model (single-session mode, or suffix for batch mode like "_model.onnx")')
    export_dl.add_argument('--ts', help='Output path for TorchScript model (single-session mode, or suffix for batch mode like "_model.pt")')
    export_dl.add_argument('--example-len', type=int, default=2000, help='Example segment length for tracing (default: 2000)')
    export_dl.add_argument('--model-type', type=int, default=2, help='Model architecture: 1=SimpleCNN, 2=ResNet1D, 3=InceptionTime, 4=Transformer, 5=2D_CNN (default: 2)')
    export_dl.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')

    return parser


def _run_batch_job(args: argparse.Namespace, process_func):
    """Generic batch runner that iterates files and calls process_func."""
    input_path = Path(args.file).expanduser()
    
    # Check if input is a directory
    if input_path.is_dir():
        print('Scanning directory: {}'.format(input_path))
        data_files = _find_data_files(input_path)
        
        if not data_files:
            print('No .eeg or .egf files found in directory')
            return
        
        print('Found {} data file(s)'.format(len(data_files)))
        
        # Track summary statistics
        successful = 0
        failed = 0
        total_events = 0
        file_results = []
        
        # Process each data file
        for data_path in data_files:
            set_path = _find_set_file(data_path)
            if not set_path and not args.skip_bits2uv:
                if args.verbose:
                    print('  Skipping {} (no .set file found, use --skip-bits2uv to process anyway)'.format(data_path))
                continue
            
            try:
                event_count = process_func(data_path, set_path, args)
                successful += 1
                total_events += event_count
                file_results.append((data_path.name, event_count))
            except Exception as e:
                print('  Error processing {}: {}'.format(data_path, e))
                failed += 1
                if args.verbose:
                    import traceback
                    traceback.print_exc()
                continue
        
        # Print summary
        print('\n' + '='*60)
        print('BATCH PROCESSING SUMMARY')
        print('='*60)
        print('Total files found:     {}'.format(len(data_files)))
        print('Successfully processed: {}'.format(successful))
        print('Failed:                 {}'.format(failed))
        print('Total HFOs detected:    {}'.format(total_events))
        if successful > 0:
            print('Average per file:       {:.1f}'.format(total_events / float(successful)))
        print('='*60)
        
        if args.verbose and file_results:
            print('\nPer-file event counts:')
            for fname, count in file_results:
                print('  {}: {} events'.format(fname, count))
    
    else:
        # Single file mode
        if not input_path.exists():
            raise FileNotFoundError('Data file not found: {}'.format(input_path))
        
        set_input = Path(args.set_file).expanduser() if args.set_file else None
        
        # If set_file is a directory, try to find the matching set file
        if set_input and set_input.is_dir():
            set_path = _find_set_file(input_path)
        else:
            set_path = set_input if set_input else input_path.with_suffix('.set')
        
        if set_path and not set_path.exists() and not args.skip_bits2uv:
            raise FileNotFoundError('Set file not found: {} (pass --skip-bits2uv to continue without scaling)'.format(set_path))
        
        process_func(input_path, set_path, args)


def run_hilbert_batch(args: argparse.Namespace):
    _run_batch_job(args, _process_single_file)


def run_ste_batch(args: argparse.Namespace):
    _run_batch_job(args, _process_ste_file)


def run_mni_batch(args: argparse.Namespace):
    _run_batch_job(args, _process_mni_file)


def run_consensus_batch(args: argparse.Namespace):
    _run_batch_job(args, _process_consensus_file)


def run_dl_batch(args: argparse.Namespace):
    # Optional dependency check: Torch required for DL detection
    try:
        import torch  # noqa: F401
    except Exception:
        print("\nDeep Learning detection requires PyTorch.\n")
        print("Install with: pip install torch")
        print("Optional (for ONNX models): pip install onnxruntime")
        print("\nTip: Activate your environment first (e.g., 'conda activate stlar').")
        return
    _run_batch_job(args, _process_dl_file)


def _split_train_val(manifest_df, val_fraction=0.2, random_seed=42, verbose=False):
    """
    Split manifest dataframe into train and validation sets.
    Uses random event-wise split (stratified if label column exists).
    
    Args:
        manifest_df: DataFrame with manifest data
        val_fraction: Fraction of events for validation (0.0-1.0)
        random_seed: Random seed for reproducibility
        verbose: Print verbose output
    
    Returns:
        train_df, val_df: Split dataframes
    """
    np.random.seed(random_seed)
    
    n_total = len(manifest_df)
    n_val = max(1, int(n_total * val_fraction))
    n_train = n_total - n_val
    
    # Try to do stratified split by label if label column exists
    if 'label' in manifest_df.columns and manifest_df['label'].notna().sum() > 0:
        # Stratified split to preserve label distribution
        from sklearn.model_selection import train_test_split
        try:
            train_df, val_df = train_test_split(
                manifest_df,
                test_size=val_fraction,
                random_state=random_seed,
                stratify=manifest_df['label'].fillna('unknown')
            )
            if verbose:
                print(f"\nUsing stratified split (preserving label distribution)")
        except Exception as e:
            # Fall back to random split if stratification fails
            if verbose:
                print(f"\nWarning: Stratified split failed ({e}), using random split")
            indices = np.random.permutation(n_total)
            val_indices = indices[:n_val]
            train_indices = indices[n_val:]
            train_df = manifest_df.iloc[train_indices].copy()
            val_df = manifest_df.iloc[val_indices].copy()
    else:
        # Random event-wise split
        indices = np.random.permutation(n_total)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        train_df = manifest_df.iloc[train_indices].copy()
        val_df = manifest_df.iloc[val_indices].copy()
        if verbose:
            print(f"\nUsing random event-wise split")
    
    # Reset indices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    
    return train_df, val_df


def _find_eoi_file(session_dir):
    """Find an EOI file (.txt or .csv) in a session directory."""
    session_path = Path(session_dir)
    for pattern in ['*.txt', '*.csv']:
        matches = list(session_path.glob(pattern))
        if matches:
            return matches[0]
    return None


def _find_egf_file(session_dir):
    """Find an EGF file in a session directory."""
    session_path = Path(session_dir)
    matches = list(session_path.glob('*.egf'))
    if matches:
        return matches[0]
    return None


def _process_single_session(eoi_path, egf_path, output_dir, args, region_preset):
    """Process a single session: load EOIs, extract segments, create manifest.
    
    Args:
        eoi_path: Path object to EOI file
        egf_path: Path object to EGF file
        output_dir: Path object for output directory
        args: argparse Namespace with common settings
        region_preset: dict with region-specific parameters
    
    Returns:
        tuple: (manifest_df, num_rows)
    """
    from pathlib import Path
    from .core.eoi_exporter import _safe_write_csv
    
    # Load EOI file
    eoi_path = Path(eoi_path)
    if not eoi_path.exists():
        raise FileNotFoundError(f"EOI file not found: {eoi_path}")
    
    # Determine format and load EOIs
    if eoi_path.suffix.lower() == '.csv':
        eoi_df = pd.read_csv(eoi_path)
        if 'start_ms' not in eoi_df.columns or 'stop_ms' not in eoi_df.columns:
            raise ValueError("CSV must have 'start_ms' and 'stop_ms' columns")
        eois_ms = eoi_df[['start_ms', 'stop_ms']].values
        labels = eoi_df['label'].tolist() if 'label' in eoi_df.columns else None
    else:
        # Try to detect format: STLAR output (tab-separated) or generic txt
        eois_ms = []
        labels = []
        
        with open(eoi_path) as f:
            first_line = f.readline().strip()
            f.seek(0)
            
            # Check if it's STLAR format (tab-separated with headers)
            if '\t' in first_line and ('Start Time' in first_line or 'start_ms' in first_line):
                # STLAR format: tab-separated with "Start Time(ms):" and "Stop Time(ms):" columns
                eoi_df = pd.read_csv(eoi_path, sep='\t')
                
                # Handle different column name formats
                start_col = None
                stop_col = None
                
                for col in eoi_df.columns:
                    if 'start' in col.lower() and 'time' in col.lower():
                        start_col = col
                    if 'stop' in col.lower() and 'time' in col.lower():
                        stop_col = col
                
                if start_col is None or stop_col is None:
                    raise ValueError(f"Could not find start/stop time columns. Found: {eoi_df.columns.tolist()}")
                
                eois_ms = eoi_df[[start_col, stop_col]].values
                # Don't use ID column as label - it contains identifiers like "HIL1", not numeric labels
                labels = None
            else:
                # Generic txt format: whitespace-separated start_ms stop_ms [label]
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            start_ms = float(parts[0])
                            stop_ms = float(parts[1])
                            eois_ms.append([start_ms, stop_ms])
                            labels.append(int(parts[2]) if len(parts) > 2 else None)
                        except ValueError:
                            continue
        
        eois_ms = np.array(eois_ms)
        labels = labels if (labels and any(l is not None for l in labels)) else None
    
    if len(eois_ms) == 0:
        raise ValueError("No EOIs found in file")
    
    if args.verbose:
        print(f"Loaded {len(eois_ms)} EOIs from {eoi_path}")
    
    # Load .egf signal file using ReadEEG
    egf_path = Path(egf_path)
    if not egf_path.exists():
        raise FileNotFoundError(f"EGF file not found: {egf_path}")
    
    try:
        signal, fs = ReadEEG(str(egf_path))
        signal = np.asarray(signal, dtype=np.float32).flatten()
        
        # Apply bits-to-uV conversion if set file provided
        if args.set_file and not args.skip_bits2uv:
            set_path = Path(args.set_file).expanduser()
            if set_path.exists():
                try:
                    conversion_factor = bits2uV(str(set_path))
                    signal = signal * conversion_factor
                    if args.verbose:
                        print(f"Applied bits-to-uV conversion: factor={conversion_factor}")
                except Exception as e:
                    print(f"Warning: Could not apply bits-to-uV conversion: {e}")
        
        if args.verbose:
            print(f"Loaded EGF signal: {signal.shape} samples at {fs} Hz")
    except Exception as e:
        raise RuntimeError(f"Failed to load EGF file: {e}")
    
    # Get region presets (hardcoded to avoid GUI initialization)
    region_presets = _get_region_presets()
    region_preset = region_presets.get(args.region, {})
    
    if not region_preset:
        raise ValueError(f"Unknown region: {args.region}")
    
    if args.verbose:
        print(f"Applied region preset: {args.region}")
        print(f"  Frequency bands: {region_preset.get('bands', {})}")
        print(f"  Durations: {region_preset.get('durations', {})}")
        print(f"  Speed range: {region_preset.get('speed_threshold_min_cm_s', 0)}-{region_preset.get('speed_threshold_max_cm_s', 5)} cm/s")
    
    # Load optional .pos file for behavior gating
    speed_signal = None
    pos_file_path = None
    
    # Try to auto-discover .pos file if not provided
    if args.pos_file:
        pos_file_path = Path(args.pos_file).expanduser()
    else:
        # Look for .pos file in same directory as EGF, with same base name
        egf_base = egf_path.stem  # e.g., "20160908-2-NO-3700" from "20160908-2-NO-3700.egf"
        possible_pos = egf_path.parent / f"{egf_base}.pos"
        if possible_pos.exists():
            pos_file_path = possible_pos
            if args.verbose:
                print(f"Auto-discovered POS file: {pos_file_path}")
    
    if pos_file_path:
        try:
            from .core.Tint_Matlab import getpos, speed2D
            # Load raw position data first, without arena-specific transformations
            x_raw, y_raw, t_raw, fs_speed = getpos(str(pos_file_path), arena='Linear Track', method='raw', custom_ppm=args.ppm)
            
            # Only proceed if we got valid position data
            if x_raw is not None and len(x_raw) > 0 and x_raw.size > 0:
                # Flatten and remove obvious bad values (1023 = missing in Axona system)
                x_clean = x_raw.flatten().copy()
                y_clean = y_raw.flatten().copy()
                t_clean = t_raw.flatten().copy()
                
                # Replace 1023 (missing data marker) with NaN
                x_clean[x_clean == 1023] = np.nan
                y_clean[y_clean == 1023] = np.nan
                
                # Remove remaining NaNs
                valid_idx = ~(np.isnan(x_clean) | np.isnan(y_clean))
                if np.sum(valid_idx) > 0:
                    x_clean = x_clean[valid_idx]
                    y_clean = y_clean[valid_idx]
                    t_clean = t_clean[valid_idx]
                    
                    # Calculate 2D speed from x, y coordinates
                    speed_data = speed2D(x_clean.reshape(-1, 1), y_clean.reshape(-1, 1), t_clean.reshape(-1, 1))
                    speed_signal = (np.asarray(speed_data, dtype=np.float32).flatten(), fs_speed)
                    if args.verbose:
                        print(f"Loaded POS speed data: {speed_signal[0].shape} samples at {fs_speed} Hz")
                else:
                    print(f"Warning: POS file has no valid position data after cleaning, behavior gating disabled")
            else:
                print(f"Warning: POS file loaded but no position samples found, behavior gating disabled")
        except Exception as e:
            if args.verbose:
                import traceback
                traceback.print_exc()
            print(f"Warning: Could not load POS file: {e}, behavior gating disabled")
    
    # Filter and annotate EOIs with region preset logic
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    rows = []
    speed_min = region_preset.get('speed_threshold_min_cm_s', 0.0)
    speed_max = region_preset.get('speed_threshold_max_cm_s', 5.0)
    bands = region_preset.get('bands', {})
    durations = region_preset.get('durations', {})
    
    for idx, (s_ms, e_ms) in enumerate(eois_ms):
        s_ms = float(s_ms)
        e_ms = float(e_ms)
        duration = e_ms - s_ms
        
        # Extract segment
        s_idx = int(max(0, np.floor(s_ms / 1000.0 * fs)))
        e_idx = int(min(len(signal), np.ceil(e_ms / 1000.0 * fs)))
        
        if e_idx <= s_idx:
            continue
        
        seg = signal[s_idx:e_idx].astype(np.float32)
        seg_path = output_dir / f"{args.prefix}_{idx:05d}.npy"
        np.save(seg_path, seg)
        
        # Determine band label (simple heuristic: check duration)
        band_label = 'ripple_fast_ripple'
        r_min = durations.get('ripple_min_ms', 15)
        r_max = durations.get('ripple_max_ms', 120)
        fr_min = durations.get('fast_min_ms', 10)
        fr_max = durations.get('fast_max_ms', 80)
        
        if r_min <= duration <= r_max:
            band_label = 'ripple'
        elif fr_min <= duration <= fr_max:
            band_label = 'fast_ripple'
        
        # Determine behavioral state if speed signal available
        state = 'unknown'
        mean_speed = None
        if speed_signal is not None:
            speed_trace, fs_speed = speed_signal
            s_idx_speed = int(max(0, np.floor(s_ms / 1000.0 * fs_speed)))
            e_idx_speed = int(min(len(speed_trace), np.ceil(e_ms / 1000.0 * fs_speed)))
            if e_idx_speed > s_idx_speed:
                seg_speed = speed_trace[s_idx_speed:e_idx_speed]
                if seg_speed.size > 0:
                    mean_speed = float(np.nanmean(seg_speed))
                    state = 'rest' if (speed_min <= mean_speed <= speed_max) else 'active'
        
        label = int(labels[idx]) if labels and labels[idx] is not None else None
        
        rows.append({
            'segment_path': str(seg_path),
            'label': label,
            'band_label': band_label,
            'duration_ms': duration,
            'state': state,
            'mean_speed_cm_s': mean_speed,
        })
    
    # Write manifest
    manifest_path = output_dir / 'manifest.csv'
    manifest_df = pd.DataFrame(rows)
    _safe_write_csv(manifest_df, manifest_path)
    
    # Return manifest dataframe and number of rows
    return manifest_df, len(rows), manifest_path


def run_prepare_dl(args: argparse.Namespace):
    """Prepare EOIs for DL training with region presets and behavior gating.
    
    Supports two modes:
    1. Single session: --eoi-file, --egf-file, --output
    2. Batch mode: --batch-dir (auto-discovers EOI/EGF files in subdirectories)
    """
    from pathlib import Path
    from .core.eoi_exporter import _safe_write_csv
    
    # Validate arguments
    if args.batch_dir:
        # Batch mode
        batch_path = Path(args.batch_dir).expanduser()
        if not batch_path.is_dir():
            raise ValueError(f"--batch-dir must be a directory: {batch_path}")
        
        # Scan for subdirectories containing .egf files
        subdirs = [d for d in batch_path.iterdir() if d.is_dir()]
        if not subdirs:
            raise ValueError(f"No subdirectories found in {batch_path}")
        
        if args.verbose:
            print(f"Batch mode: Found {len(subdirs)} subdirectories")
        
        # Process each subdirectory
        total_events = 0
        processed_dirs = 0
        failed_dirs = 0
        
        for session_dir in sorted(subdirs):
            session_name = session_dir.name
            
            # Find EOI and EGF files
            eoi_file = _find_eoi_file(session_dir)
            egf_file = _find_egf_file(session_dir)
            
            if not eoi_file or not egf_file:
                if args.verbose:
                    print(f"  Skipping {session_name}: missing EOI or EGF file")
                failed_dirs += 1
                continue
            
            # Create output subdirectory for this session
            session_output = session_dir / 'prepared_dl'
            
            try:
                if args.verbose:
                    print(f"  Processing: {session_name}")
                
                # Get region preset
                region_presets = _get_region_presets()
                region_preset = region_presets.get(args.region, {})
                
                # Process session
                manifest_df, num_rows, manifest_path = _process_single_session(
                    eoi_file, egf_file, session_output, args, region_preset
                )
                
                # Handle train/val splitting if requested
                if args.split_train_val:
                    train_df, val_df = _split_train_val(manifest_df, args.val_fraction, args.random_seed, args.verbose)
                    train_manifest = session_output / 'manifest_train.csv'
                    val_manifest = session_output / 'manifest_val.csv'
                    _safe_write_csv(train_df, train_manifest)
                    _safe_write_csv(val_df, val_manifest)
                    total_events += len(train_df) + len(val_df)
                else:
                    total_events += num_rows
                
                print(f"    ✓ {session_name}: {num_rows} events → {manifest_path}")
                processed_dirs += 1
                
            except Exception as e:
                print(f"    ✗ {session_name}: {e}")
                failed_dirs += 1
                if args.verbose:
                    import traceback
                    traceback.print_exc()
        
        print(f"\n{'='*60}")
        print("BATCH PREPARED DL TRAINING DATA")
        print(f"{'='*60}")
        print(f"Processed:       {processed_dirs} sessions")
        print(f"Failed:          {failed_dirs} sessions")
        print(f"Total events:    {total_events}")
        print(f"Output base:     {batch_path}")
        print(f"Region:          {args.region}")
        print(f"{'='*60}\n")
        
        if processed_dirs == 0:
            raise RuntimeError("No sessions were successfully processed")
        
    else:
        # Single session mode
        if not args.eoi_file or not args.egf_file or not args.output:
            raise ValueError("Single-session mode requires: --eoi-file, --egf-file, --output")
        
        # Get region preset
        region_presets = _get_region_presets()
        region_preset = region_presets.get(args.region, {})
        
        # Process single session
        eoi_path = Path(args.eoi_file).expanduser()
        egf_path = Path(args.egf_file).expanduser()
        output_dir = Path(args.output).expanduser()
        
        manifest_df, num_rows, manifest_path = _process_single_session(
            eoi_path, egf_path, output_dir, args, region_preset
        )
        
        # Handle train/val splitting if requested
        if args.split_train_val:
            train_df, val_df = _split_train_val(manifest_df, args.val_fraction, args.random_seed, args.verbose)
            
            # Write train/val manifests
            train_manifest_path = output_dir / 'manifest_train.csv'
            val_manifest_path = output_dir / 'manifest_val.csv'
            
            _safe_write_csv(train_df, train_manifest_path)
            _safe_write_csv(val_df, val_manifest_path)
        
        print(f"\n{'='*60}")
        print("PREPARED DL TRAINING DATA")
        print(f"{'='*60}")
        print(f"Region:          {args.region}")
        print(f"EOIs processed:  {num_rows}")
        print(f"Output dir:      {output_dir}")
        print(f"Manifest:        {manifest_path}")
        if args.split_train_val:
            print(f"Train manifest:  {train_manifest_path} ({len(train_df)} events)")
            print(f"Val manifest:    {val_manifest_path} ({len(val_df)} events)")
        print(f"{'='*60}\n")


def _get_region_presets():
    """Get region presets (hardcoded to avoid GUI initialization)."""
    return {
        'LEC': {
            'bands': {
                'ripple': [80, 250],
                'fast_ripple': [250, 500],
                'gamma': [30, 80],
            },
            'durations': {
                'ripple_min_ms': 15,
                'ripple_max_ms': 120,
                'fast_min_ms': 10,
                'fast_max_ms': 80,
            },
            'threshold_sd': 3.5,
            'epoch_s': 300,
            'behavior_gating': True,
            'speed_threshold_min_cm_s': 0.0,
            'speed_threshold_max_cm_s': 5.0,
            'dl_export': {
                'filter_by_duration': True,
                'annotate_band': True,
                'behavior_gating': True,
            },
        },
        'Hippocampus': {
            'bands': {
                'ripple': [100, 250],
                'fast_ripple': [250, 500],
                'gamma': [30, 80],
            },
            'durations': {
                'ripple_min_ms': 15,
                'ripple_max_ms': 120,
                'fast_min_ms': 10,
                'fast_max_ms': 80,
            },
            'threshold_sd': 4.0,
            'epoch_s': 300,
            'behavior_gating': True,
            'speed_threshold_min_cm_s': 0.0,
            'speed_threshold_max_cm_s': 5.0,
            'dl_export': {
                'filter_by_duration': True,
                'annotate_band': True,
                'behavior_gating': True,
            },
        },
        'MEC': {
            'bands': {
                'ripple': [80, 200],
                'fast_ripple': [200, 500],
                'gamma': [30, 80],
            },
            'durations': {
                'ripple_min_ms': 15,
                'ripple_max_ms': 120,
                'fast_min_ms': 10,
                'fast_max_ms': 80,
            },
            'threshold_sd': 3.5,
            'epoch_s': 300,
            'behavior_gating': True,
            'speed_threshold_min_cm_s': 0.0,
            'speed_threshold_max_cm_s': 5.0,
            'dl_export': {
                'filter_by_duration': True,
                'annotate_band': True,
                'behavior_gating': True,
            },
        },
    }


def run_train_dl(args: argparse.Namespace):
    """Train a 1D CNN model on prepared DL training data.
    
    Supports two modes:
    1. Single-session: --train, --val
    2. Batch mode: --batch-dir (auto-discovers manifest_train.csv and manifest_val.csv in subdirectories)
    """
    from pathlib import Path
    
    try:
        from .dl_training.train import main as train_main
    except ImportError as e:
        print(f"Error: Deep learning dependencies not installed. Install with: pip install torch")
        raise
    
    if args.batch_dir:
        # Batch mode: find all subdirectories with manifest_train.csv and manifest_val.csv
        batch_path = Path(args.batch_dir).expanduser()
        if not batch_path.is_dir():
            raise ValueError(f"--batch-dir must be a directory: {batch_path}")
        
        # Scan for subdirectories
        subdirs = [d for d in batch_path.iterdir() if d.is_dir()]
        if not subdirs:
            raise ValueError(f"No subdirectories found in {batch_path}")
        
        if args.verbose:
            print(f"Batch mode: Found {len(subdirs)} subdirectories")
        
        # Process each subdirectory
        successful = 0
        failed = 0
        
        for session_dir in sorted(subdirs):
            session_name = session_dir.name
            
            # Look for manifest_train.csv and manifest_val.csv
            train_manifest = session_dir / 'manifest_train.csv'
            val_manifest = session_dir / 'manifest_val.csv'
            
            # Also check in prepared_dl subdirectory
            if not train_manifest.exists():
                train_manifest = session_dir / 'prepared_dl' / 'manifest_train.csv'
                val_manifest = session_dir / 'prepared_dl' / 'manifest_val.csv'
            
            if not train_manifest.exists() or not val_manifest.exists():
                if args.verbose:
                    print(f"  Skipping {session_name}: missing manifest_train.csv or manifest_val.csv")
                failed += 1
                continue
            
            # Create session-specific output directory
            session_output = session_dir / 'models' if (session_dir / 'prepared_dl').exists() else session_dir / 'models'
            session_output.mkdir(parents=True, exist_ok=True)
            
            try:
                if args.verbose:
                    print(f"  Training: {session_name}")
                
                # Train model for this session
                import sys
                old_argv = sys.argv
                try:
                    train_argv = [
                        'train-dl',
                        '--train', str(train_manifest),
                        '--val', str(val_manifest),
                        '--epochs', str(args.epochs),
                        '--batch-size', str(args.batch_size),
                        '--lr', str(args.lr),
                        '--weight-decay', str(args.weight_decay),
                        '--out-dir', str(session_output),
                        '--num-workers', str(args.num_workers),
                        '--model-type', str(args.model_type),
                    ]
                    if args.no_plot:
                        train_argv.append('--no-plot')
                    if args.gui:
                        train_argv.append('--gui')
                    sys.argv = train_argv
                    
                    from .dl_training.train import parse_args, main
                    train_parsed_args = parse_args()
                    main()
                    
                    print(f"    ✓ {session_name}: Models saved to {session_output}")
                    successful += 1
                    
                finally:
                    sys.argv = old_argv
                    
            except Exception as e:
                print(f"    ✗ {session_name}: {e}")
                failed += 1
                if args.verbose:
                    import traceback
                    traceback.print_exc()
        
        print(f"\n{'='*60}")
        print("BATCH TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Successful:      {successful} sessions")
        print(f"Failed:          {failed} sessions")
        print(f"Output base:     {batch_path}")
        print(f"{'='*60}\n")
        
        if successful == 0:
            raise RuntimeError("No sessions were successfully trained")
        
    else:
        # Single-session mode
        if not args.train or not args.val:
            raise ValueError("Single-session mode requires: --train, --val")
        
        import sys
        old_argv = sys.argv
        try:
            train_argv = [
                'train-dl',
                '--train', args.train,
                '--val', args.val,
                '--epochs', str(args.epochs),
                '--batch-size', str(args.batch_size),
                '--lr', str(args.lr),
                '--weight-decay', str(args.weight_decay),
                '--out-dir', args.out_dir,
                '--num-workers', str(args.num_workers),
                '--model-type', str(args.model_type),
            ]
            if args.no_plot:
                train_argv.append('--no-plot')
            if args.gui:
                train_argv.append('--gui')
            sys.argv = train_argv
            
            from .dl_training.train import parse_args, main
            train_parsed_args = parse_args()
            main()
            
        finally:
            sys.argv = old_argv


def run_export_dl(args: argparse.Namespace):
    """Export trained DL model to TorchScript and ONNX formats.
    
    Supports two modes:
    1. Single-session: --ckpt, --onnx, --ts
    2. Batch mode: --batch-dir (auto-discovers best.pt in subdirectories)
    """
    from pathlib import Path
    
    try:
        from .dl_training.export import main as export_main
    except ImportError as e:
        print(f"Error: Deep learning dependencies not installed. Install with: pip install torch")
        raise
    
    if args.batch_dir:
        # Batch mode: find all best.pt checkpoints in subdirectories
        batch_path = Path(args.batch_dir).expanduser()
        if not batch_path.is_dir():
            raise ValueError(f"--batch-dir must be a directory: {batch_path}")
        
        # Scan for subdirectories
        subdirs = [d for d in batch_path.iterdir() if d.is_dir()]
        if not subdirs:
            raise ValueError(f"No subdirectories found in {batch_path}")
        
        if args.verbose:
            print(f"Batch mode: Found {len(subdirs)} subdirectories")
        
        # Process each subdirectory
        successful = 0
        failed = 0
        
        for session_dir in sorted(subdirs):
            session_name = session_dir.name
            
            # Look for best.pt checkpoint in models subdirectory
            ckpt_path = session_dir / 'models' / 'best.pt'
            if not ckpt_path.exists():
                # Try directly in session directory
                ckpt_path = session_dir / 'best.pt'
            
            if not ckpt_path.exists():
                if args.verbose:
                    print(f"  Skipping {session_name}: missing best.pt checkpoint")
                failed += 1
                continue
            
            try:
                if args.verbose:
                    print(f"  Exporting: {session_name}")
                
                # Create output paths
                ckpt_dir = ckpt_path.parent
                onnx_path = ckpt_dir / f"{session_name}_model.onnx" if not args.onnx else ckpt_dir / args.onnx
                ts_path = ckpt_dir / f"{session_name}_model.pt" if not args.ts else ckpt_dir / args.ts
                
                # Export model
                import sys
                old_argv = sys.argv
                try:
                    export_argv = [
                        'export-dl',
                        '--ckpt', str(ckpt_path),
                        '--onnx', str(onnx_path),
                        '--ts', str(ts_path),
                        '--example-len', str(args.example_len),
                        '--model-type', str(args.model_type),
                    ]
                    sys.argv = export_argv
                    
                    from .dl_training.export import parse_args, main
                    export_parsed_args = parse_args()
                    main()
                    
                    print(f"    ✓ {session_name}: Exported to {ts_path} and {onnx_path}")
                    successful += 1
                    
                finally:
                    sys.argv = old_argv
                    
            except Exception as e:
                print(f"    ✗ {session_name}: {e}")
                failed += 1
                if args.verbose:
                    import traceback
                    traceback.print_exc()
        
        print(f"\n{'='*60}")
        print("BATCH EXPORT COMPLETE")
        print(f"{'='*60}")
        print(f"Successful:      {successful} sessions")
        print(f"Failed:          {failed} sessions")
        print(f"Output base:     {batch_path}")
        print(f"{'='*60}\n")
        
        if successful == 0:
            raise RuntimeError("No sessions were successfully exported")
        
    else:
        # Single-session mode
        if not args.ckpt or not args.onnx or not args.ts:
            raise ValueError("Single-session mode requires: --ckpt, --onnx, --ts")
        
        import sys
        old_argv = sys.argv
        try:
            export_argv = [
                'export-dl',
                '--ckpt', args.ckpt,
                '--onnx', args.onnx,
                '--ts', args.ts,
                '--example-len', str(args.example_len),
                '--model-type', str(args.model_type),
            ]
            sys.argv = export_argv
            
            from .dl_training.export import parse_args, main
            export_parsed_args = parse_args()
            main()
            
        finally:
            sys.argv = old_argv


__all__ = ['build_parser', 'run_hilbert_batch', 'run_ste_batch', 'run_mni_batch', 'run_consensus_batch', 'run_dl_batch', 'run_prepare_dl', 'run_train_dl', 'run_export_dl']


def main(args=None):
    """Main CLI entry point. Can accept pre-parsed args or parse from sys.argv."""
    if args is None:
        parser = build_parser()
        args = parser.parse_args()
    
    if args.command == 'hilbert-batch':
        run_hilbert_batch(args)
    elif args.command == 'ste-batch':
        run_ste_batch(args)
    elif args.command == 'mni-batch':
        run_mni_batch(args)
    elif args.command == 'consensus-batch':
        run_consensus_batch(args)
    elif args.command == 'dl-batch':
        run_dl_batch(args)
    elif args.command == 'prepare-dl':
        run_prepare_dl(args)
    elif args.command == 'train-dl':
        run_train_dl(args)
    elif args.command == 'export-dl':
        run_export_dl(args)
    else:
        parser = build_parser()
        parser.print_help()


if __name__ == '__main__':
    main()
