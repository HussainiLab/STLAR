# STLAR: Spatio-Temporal LFP Analysis & Research

Unified spatio-temporal LFP analysis tool combining PyhfoGUI and Spatial_Spectral_Mapper.

![STLAR Logo](docs/assets/stlar_logo.png)

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
pip install -e .
```

## Quick Start

### GUI Mode
```bash
python -m stlar
```

### Batch Processing
```bash
python -m stlar.batch_processing.batch_cli \
    --config configs/hfo_detection.yaml \
    --input data/*.set \
    --output results/
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

## Original Tools

This project unifies:
- [PyhfoGUI](https://github.com/HussainiLab/PyhfoGUI) - Temporal LFP analysis
- [Spatial_Spectral_Mapper](https://github.com/HussainiLab/Spatial_Spectral_Mapper) - Spatial LFP analysis

Legacy code preserved in `legacy/` folder with full commit history.

## Documentation

- [Installation Guide](docs/installation.md)
- [User Manual](docs/user_manual.md)
- [API Reference](docs/api_reference.md)
- [Migration Guide](docs/migration_guide.md) - For users of original tools
- [Contributing](CONTRIBUTING.md)

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
