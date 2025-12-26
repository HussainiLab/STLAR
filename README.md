# STLAR: Spatio-Temporal LFP Analyzer

STLAR (pronounced Stellar) is a Spatio-Temporal LFP analysis tool combining hfoGUI and Spatial Spectral Mapper.

## Features

### 🕐 Temporal Analysis
- HFO detection (ripples 140-250 Hz, fast ripples 250-500 Hz)
- Multi-band filtering and visualization
- Time-frequency analysis (Stockwell transform)
- Event scoring and annotation
- Automated detection algorithms

### 🗺️ Spatial Analysis
- Frequency heatmaps across arena
- Position tracking overlay
- Spatial power distribution
- Arena coverage visualization
- Multi-channel spatial mapping

### 🔗 Spatio-Temporal Integration
- Synchronized temporal-spatial views
- HFO location mapping
- Behavioral state-dependent analysis
- Cross-region coordination metrics

### 🤖 Deep Learning
- Automated HFO classification
- Train custom models on your data
- PyTorch (.pt) and ONNX export
- Pre-trained models available

### ⚡ Batch Processing
- Parallel multi-file analysis
- Configurable pipelines (YAML)
- Command-line interface
- Progress tracking and logging

## Installation
```bash
# Clone repository
git clone https://github.com/HussainiLab/STLAR.git
cd STLAR

# Install dependencies
pip install -r requirements.txt

# (Optional) install package locally
pip install -e .
```

## Project Structure
```
STLAR/
├── hfoGUI/           # Temporal analysis (HFO detection)
├── stlar/            # Main package wrapper
├── spatial_mapper/   # Spatial spectral mapping
├── docs/             # Documentation
├── tests/            # Test files
├── settings/         # User configuration files (gitignored)
├── HFOScores/        # User output data (gitignored)
└── main.py           # Entry point
```

**Note:** `settings/` and `HFOScores/` directories contain user-generated configuration and output data. These are created automatically on first run and should not be committed to version control.

## Quick Start

### HFO Analysis GUI (Temporal)
```bash
python -m stlar
```

### Spatial Spectral Mapper GUI
```bash
python spatial_mapper/src/main.py
```

### Batch Processing

**HFO Detection:**
```bash
python -m stlar.batch_processing.batch_cli \
    --config configs/hfo_detection.yaml \
    --input data/*.set \
    --output results/
```

**Spatial Spectral Mapper:**
```bash
# Single file
python spatial_mapper/src/batch_ssm.py data/recording.eeg --export-binned-jpgs -o output/

# Directory batch mode
python spatial_mapper/src/batch_ssm.py data/ --ppm 511 --chunk-size 10 -o results/
```

### Python API
```python
from stlar.core.analysis import TemporalAnalyzer, SpatialAnalyzer

# Temporal analysis
temporal = TemporalAnalyzer()
ripples = temporal.detect_ripples(lfp_data)

# Spatial analysis
spatial = SpatialAnalyzer()
heatmaps = spatial.create_frequency_maps(lfp_data, position_data)
```

## Module Structure

### Temporal Analysis (HFO Detection)
- Location: `hfoGUI/` and `stlar/`
- Entry: `python -m stlar`
- See: [docs/CONSENSUS_QUICKSTART.md](docs/CONSENSUS_QUICKSTART.md), [docs/CONSENSUS_DETECTION.md](docs/CONSENSUS_DETECTION.md)

### Spatial Analysis (Spectral Mapper)
- Location: `spatial_mapper/`
- Entry: `python spatial_mapper/src/main.py` (GUI) or `python spatial_mapper/src/batch_ssm.py` (CLI)
- See: [spatial_mapper/README.md](spatial_mapper/README.md)

## Original Tools

This project unifies:
- [PyhfoGUI](https://github.com/HussainiLab/PyhfoGUI) - Temporal LFP analysis
- [Spatial_Spectral_Mapper](https://github.com/HussainiLab/Spatial_Spectral_Mapper) - Spatial LFP analysis

<!-- Legacy code note removed: no `legacy/` folder in repo -->

## Documentation

### HFO Detection (Temporal Analysis)
- [Consensus Detection Overview](docs/CONSENSUS_DETECTION.md)
- [Consensus Quick Start](docs/CONSENSUS_QUICKSTART.md)
- [Consensus Summary](docs/CONSENSUS_SUMMARY.md)

### Spatial Spectral Mapper
- [Spatial Mapper Guide](spatial_mapper/README.md)

### General
- Installation: See main README
- Contributing: [CONTRIBUTING.md](CONTRIBUTING.md) (if exists)

## Citation

If you use STLAR in your research, please cite:
```bibtex
@software{stlar2025,
  title = {STLAR: Spatio-Temporal LFP Analysis & Research},
  author = {HussainiLab},
  year = {2025},
  url = {https://github.com/HussainiLab/STLAR}
}
```

And the original tools:
- PyhfoGUI: [Citation info]
- Spatial_Spectral_Mapper: [Citation info]

## License

GPL-3.0 License - see [LICENSE](LICENSE) file for details.

## Contact

- Issues: https://github.com/HussainiLab/STLAR/issues
- Lab Website: [HussainiLab URL]
- Email: [Contact email]

## Acknowledgments

Built upon the foundational work of:
- PyhfoGUI contributors
- Spatial_Spectral_Mapper contributors
- HussainiLab members

---

**STLAR** - Advancing spatio-temporal understanding of neural oscillations 🧠
