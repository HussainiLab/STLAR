import sys
import argparse
from pathlib import Path

def main_cli():
    """Unified STLAR CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='stlar',
        description='STLAR: High-Frequency Oscillation Detection & Spatial Mapping',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # HFO Detection
  python -m stlar hilbert-batch -f data/recording.egf
  python -m stlar consensus-batch -f data/recording.egf --voting-strategy majority
  
  # Spatial Mapping
  python -m stlar batch-ssm data/recording.eeg --ppm 595 --chunk-size 180
  python -m stlar batch-ssm data/recordings/ --export-binned-jpgs --export-binned-csvs
  
  # GUI
  python -m stlar gui
  python -m stlar spatial-gui
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # GUI commands
    gui_parser = subparsers.add_parser('gui', help='Launch STLAR HFO GUI')
    spatial_gui_parser = subparsers.add_parser('spatial-gui', help='Launch Spatial Spectral Mapper GUI')
    
    # HFO detection commands (delegate to hfoGUI.cli)
    hfo_parser = subparsers.add_parser('hilbert-batch', help='Hilbert HFO detection', add_help=False)
    ste_parser = subparsers.add_parser('ste-batch', help='STE/RMS HFO detection', add_help=False)
    mni_parser = subparsers.add_parser('mni-batch', help='MNI HFO detection', add_help=False)
    consensus_parser = subparsers.add_parser('consensus-batch', help='Consensus HFO detection (voting)', add_help=False)
    dl_parser = subparsers.add_parser('dl-batch', help='Deep learning HFO detection', add_help=False)
    prepare_dl_parser = subparsers.add_parser('prepare-dl', help='Prepare EOIs for DL training', add_help=False)
    
    # Spatial mapping command (delegate to batch_ssm)
    ssm_parser = subparsers.add_parser('batch-ssm', help='Batch spatial spectral mapper', add_help=False)
    
    # Parse only the command, let submodules handle their own args
    args, remaining = parser.parse_known_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    if args.command == 'gui':
        from hfoGUI.__main__ import main as gui_main
        return gui_main([])
    
    if args.command == 'spatial-gui':
        import subprocess
        # Run main.py as a subprocess to ensure proper module paths
        spatial_main_script = Path(__file__).parent.parent / 'spatial_mapper' / 'src' / 'main.py'
        cmd = [sys.executable, str(spatial_main_script)]
        result = subprocess.run(cmd)
        return result.returncode
    
    # HFO detection commands
    if args.command in ['hilbert-batch', 'ste-batch', 'mni-batch', 'consensus-batch', 'dl-batch', 'prepare-dl']:
        from hfoGUI.cli import build_parser, main as hfo_main
        hfo_parser_full = build_parser()
        hfo_args = hfo_parser_full.parse_args([args.command] + remaining)
        return hfo_main(hfo_args)
    
    # Spatial mapping command
    if args.command == 'batch-ssm':
        import subprocess
        # Run batch_ssm.py as a subprocess to ensure proper module paths
        ssm_script = Path(__file__).parent.parent / 'spatial_mapper' / 'src' / 'batch_ssm.py'
        cmd = [sys.executable, str(ssm_script)] + remaining
        result = subprocess.run(cmd)
        return result.returncode
    
    parser.print_help()
    return 1

if __name__ == '__main__':
    sys.exit(main_cli())
