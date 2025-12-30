import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import time
import sys


def _safe_write_csv(df, path, max_retries=3, delay=0.5):
    """
    Write DataFrame to CSV with retry logic for locked files.
    
    Args:
        df: DataFrame to write
        path: Output path
        max_retries: Maximum number of retry attempts
        delay: Delay in seconds between retries
    
    Raises:
        PermissionError: If file remains locked after all retries
    """
    path = Path(path)
    
    for attempt in range(max_retries):
        try:
            df.to_csv(path, index=False)
            return  # Success
        except PermissionError as e:
            if attempt < max_retries - 1:
                print(f"Warning: {path.name} is locked (possibly open in Excel). Retrying in {delay}s... ({attempt + 1}/{max_retries})", file=sys.stderr)
                time.sleep(delay)
            else:
                raise PermissionError(
                    f"Cannot write to {path.name} - file is locked.\n"
                    f"Please close the file in Excel or other applications and try again."
                ) from e
        except Exception as e:
            # For other errors, fail immediately
            raise


def extract_segments(signal, fs, eois_ms, out_dir: Path, prefix: str = "seg"):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, (s_ms, e_ms) in enumerate(eois_ms):
        s = int(float(s_ms) * fs / 1000.0)
        e = int(float(e_ms) * fs / 1000.0)
        s = max(0, s)
        e = min(len(signal), e)
        if e <= s:
            continue
        seg = signal[s:e].astype(np.float32)
        fname = out_dir / f"{prefix}_{idx:05d}.npy"
        np.save(fname, seg)
        rows.append({'segment_path': str(fname), 'label': None})
    return rows


def main():
    parser = argparse.ArgumentParser(description="Extract EOIs to .npy segments and build a manifest")
    parser.add_argument('--eois', required=True, help='Path to EOIs CSV with start_ms,stop_ms[,label]')
    parser.add_argument('--signal', required=True, help='Path to a .npy containing the raw signal for the channel')
    parser.add_argument('--fs', type=float, required=True, help='Sampling frequency (Hz)')
    parser.add_argument('--out-dir', required=True, help='Output directory for segments and manifest')
    parser.add_argument('--prefix', default='seg', help='Prefix for segment filenames')
    args = parser.parse_args()

    eois_df = pd.read_csv(args.eois)
    if not {'start_ms', 'stop_ms'} <= set(eois_df.columns):
        raise ValueError('EOI CSV must have columns: start_ms, stop_ms (optional: label)')

    signal = np.load(args.signal).astype(np.float32)
    eois_ms = eois_df[['start_ms', 'stop_ms']].to_numpy()
    out_dir = Path(args.out_dir)

    rows = extract_segments(signal, args.fs, eois_ms, out_dir, prefix=args.prefix)

    # If label column exists, attach it; otherwise leave None
    if 'label' in eois_df.columns:
        labels = eois_df['label'].tolist()
        for r, lbl in zip(rows, labels):
            r['label'] = int(lbl)

    manifest_path = out_dir / 'manifest.csv'
    _safe_write_csv(pd.DataFrame(rows), manifest_path)
    print(f"Wrote {len(rows)} segments to {out_dir} and manifest to {manifest_path}")


if __name__ == '__main__':
    main()
