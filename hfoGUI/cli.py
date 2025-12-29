import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .core.Score import hilbert_detect_events
from .core.Detector import ste_detect_events, mni_detect_events, dl_detect_events, consensus_detect_events
from .core.Tint_Matlab import ReadEEG, bits2uV, TintException

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

    events = dl_detect_events(raw_data, Fs, **params)

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
    dl.add_argument('--skip-bits2uv', action='store_true', help='Skip bits-to-uV conversion')
    dl.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')

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
    _run_batch_job(args, _process_dl_file)


__all__ = ['build_parser', 'run_hilbert_batch', 'run_ste_batch', 'run_mni_batch', 'run_consensus_batch', 'run_dl_batch']


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
    else:
        parser = build_parser()
        parser.print_help()


if __name__ == '__main__':
    main()
