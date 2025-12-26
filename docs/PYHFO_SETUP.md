# pyHFO Setup (Deprecated)

## Status: Deprecated — pyHFO repo not used

This project no longer relies on the external `pyHFO` repository. All HFO detection methods are implemented locally and available out-of-the-box:

- Hilbert envelope detector
- Short‑Term Energy (STE/RMS) detector
- MNI percentile detector
- Consensus voting (Hilbert + STE + MNI)

No separate `pyHFO` installation or setup is required.

## What to use instead

Use the built‑in detectors via the GUI or CLI:

- GUI: Launch with `python -m stlar` → Score Window → Automatic Detection tab → choose `Hilbert`, `STE`, `MNI`, or `Consensus`.
- CLI: See consensus and single‑detector batch commands in the Quick Start and Detection docs.

Recommended starting points:
- Detection overview: CONSENSUS_DETECTION.md
- Quick start: CONSENSUS_QUICKSTART.md

## Notes

- Frequency bands are user‑configurable in the parameter window (e.g., 80–250 Hz ripples; 250–500 Hz fast ripples).
- Settings and scores are saved under `HFOScores/<session>/` with method tags (`HIL`, `STE`, `MNI`, `CON`).
- If you see import errors, ensure `PyQt5` and `pyqtgraph` are installed (see `requirements.txt`).

## Troubleshooting

- Missing GUI dependencies: Install from `requirements.txt`.
- No events detected: Lower thresholds or widen frequency band.
- Too many events: Use Consensus (majority) or increase thresholds.

## References

- Implementation paths: `hfoGUI/core/Detector.py`, `hfoGUI/core/Score.py`, `hfoGUI/cli.py`, `stlar/__main__.py`
