# Spatial_Spectral_Mapper
## Map frequency power spectrums from neural eeg/egf data as a function of position

### Requirements
- **Python 3.13** (Python 3.11+ recommended)
- **Windows** (tested on Windows 10/11)
- PyQt5 for GUI
- NumPy 2.x, SciPy 1.16+, Matplotlib 3.8+
- See `requirements.txt` for full dependency list

### Quick Start

1) Install Python 3.13 (Windows 10/11).
2) Clone the repo and install dependencies:
   - `pip install -r requirements.txt`
3) Run the GUI:
   - `python src/main.py`
4) Batch mode (optional):
   - `python src/batch_ssm.py path/to/recording.eeg --export-binned-jpgs`

See requirements.txt for the full list of packages.

### Features
- **Real-time frequency band analysis** (Delta, Theta, Beta, Gamma, Ripple, Fast Ripple)
- **Spatial heatmap visualization** with occupancy normalization
- **Binned Analysis Studio** (NEW) - Advanced 4×4 and polar spatial binning with:
  - Interactive time-chunk slider for temporal navigation
  - Toggle between absolute power and percent power views
  - Per-chunk occupancy and dominant frequency band visualization
  - **Per-chunk data export**: Mean power, percent power, occupancy, dominant band, and EOIs (Events of Interest)
  - Export all per-chunk visualizations as JPG files
  - Support for circular (polar: 2 rings × 8 sectors) and rectangular (4×4 grid) arena detection
- **EOI (Events of Interest) support**:
  - Automatic detection of EOI files from HFOScores folder or CSV format
  - Per-chunk EOI distribution mapping to spatial bins
  - EOI counts exported alongside other binned metrics
- **Speed filtering** - Analyze only data within specific speed ranges
- **Excel export** for all data outputs (automatic CSV fallback if openpyxl unavailable):
  - Time bins with distance per bin and cumulative distance (in cm)
  - Average power per frequency band
  - Percentage of total power per band
  - Multi-sheet workbooks for binned analysis metrics (per-chunk granularity)
- **Interactive time slider** to view frequency maps across recording duration
- **Graph mode** for power spectral density visualization

### Usage

#### GUI Mode (Interactive)
1. Launch the application: `python src/main.py`
2. Click **Browse file** to select your `.eeg` or `.egf` file (`.pos` file will be auto-detected)
3. Configure parameters:
   - **PPM** (pixels per meter): Spatial resolution of position data (default: 511)
   - **Chunk Size**: Time window for analysis in seconds (default: 10)
   - **Window Type**: FFT window function (default: hann)
   - **Speed Filter**: Analyze only movement within specified speed range (cm/s)
4. Click **Render** to process the data
5. Use the **time slider** to navigate through different time bins
6. Click **Save data** to export main analysis as Excel file
7. Click **Binned Analysis Studio** to open advanced spatial binning interface

#### Binned Analysis Studio
The Binned Analysis Studio provides detailed spatial bin analysis with automatic arena shape detection:

**Arena Detection:**
- **Circular arenas**: Polar binning (2 rings × 8 sectors = 16 bins)
  - Inner ring: 0 to 1/√2 (equal-area split)
  - Outer ring: 1/√2 to 1.0
  - 8 sectors: -π to π
- **Rectangular/Square arenas**: Cartesian binning (4×4 grid = 16 bins)
  - Automatic detection based on trajectory bounds

**Features:**
- **Time-chunk navigation**: Slider to view data across different time windows
- **View modes**:
  - Frequency band power (absolute or percent) - per chunk
  - Occupancy heatmap (time spent in each bin) - per chunk
  - Dominant frequency band per bin - per chunk
  - EOI distribution (if EOI file available) - per chunk
- **Toggle percent power**: Switch between absolute power values and percentage of total power
- **Export Data**: Save binned metrics to Excel files (all per-chunk except occupancy aggregate):
  - `_mean_power_per_chunk.xlsx` - Mean power per band per chunk (one sheet per band)
  - `_percent_power_per_chunk.xlsx` - Percent power per band per chunk (one sheet per band)
  - `_percent_occupancy_per_chunk.xlsx` - Percent occupancy per chunk
  - `_dominant_band_per_chunk.xlsx` - Dominant band per chunk
  - `_eois_per_chunk.xlsx` - EOI counts per chunk (if available)
- **Export All JPGs**: Save all visualizations:
  - Mean power heatmaps per chunk (one per time chunk)
  - Percent power heatmaps per chunk
  - Dominant band & EOI distribution heatmaps per chunk
  - All files saved to `binned_analysis_output/` subfolder

**Output Structure (GUI - Binned Analysis Studio):**
```
binned_analysis_output/
├── {filename}_binned_mean_power_per_chunk.xlsx        # Mean power per band per chunk
├── {filename}_binned_percent_power_per_chunk.xlsx     # Percent power per band per chunk
├── {filename}_binned_percent_occupancy_per_chunk.xlsx # Occupancy distribution per chunk
├── {filename}_binned_dominant_band_per_chunk.xlsx     # Dominant band per chunk
├── {filename}_binned_eois_per_chunk.xlsx              # EOI counts per chunk (if available)
├── {filename}_chunk0_mean_power.jpg                   # Per-chunk visualizations
├── {filename}_chunk0_percent_power.jpg
├── {filename}_chunk0_dominant_eoi.jpg
├── {filename}_chunk1_mean_power.jpg
├── {filename}_chunk1_percent_power.jpg
├── {filename}_chunk1_dominant_eoi.jpg
├── ... (one set per chunk)
└── {filename}_binned_heatmap.jpg                      # Combined static visualization
```

### Examples
- examples/binned_analysis_example.py: end‑to‑end demo that runs the 4×4 binned analysis and writes Excel tables plus JPG visualizations (defaults: window=hann, PPM=511). Outputs to examples/outputs/.
- examples/example_binned_analysis.py: integration pattern for invoking binned analysis inside a pipeline (Excel‑first exports + JPG heatmap).

#### Batch Mode (No GUI)

Basic usage:
```bash
python src/batch_ssm.py path/to/recording.eeg --export-binned-csvs -o ./output/
```

Key flags:
- `--ppm 595` (default: 600), `--chunk-size 60` (seconds), `--window hann`
- `--speed-filter 2,30` (cm/s range), `-o ./output/` (output directory)
- `--export-binned-csvs`: exports binned data (mean power, percent power, occupancy, dominant band, EOIs) to Excel/CSV
- `--export-binned-jpgs`: also generates per-chunk JPG visualizations

**EOI Support:**
- Automatically searches for EOI files in the same directory as the recording:
  - `{filename}_EOI.csv` (two-column CSV: start, stop in seconds or milliseconds)
  - `HFOScores/{filename}/{filename}_*.txt` (Axona HFOScore format: ID, Start(ms), Stop(ms), ...)
- EOI segments are mapped to spatial bins and exported per chunk if found

**Output Structure (Batch Mode):**

*With --export-binned-csvs flag:*
```
output_folder/
├── {filename}_SSM.xlsx                              # Main analysis Excel (power, distance, per-chunk metrics)
└── {filename}_binned_analysis/                      # Binned analysis subfolder
    ├── {filename}_binned_mean_power_per_chunk.xlsx          # Mean power per band, per chunk
    ├── {filename}_binned_percent_power_per_chunk.xlsx       # Percent power per band, per chunk
    ├── {filename}_binned_percent_occupancy_per_chunk.xlsx   # Occupancy distribution per chunk
    ├── {filename}_binned_dominant_band_per_chunk.xlsx       # Dominant band per chunk
    ├── {filename}_binned_eois_per_chunk.xlsx                # EOI counts per chunk (if available)
    ├── {filename}_binned_heatmap.jpg                        # Combined static visualization
    └── (CSV equivalents if openpyxl unavailable)
```

*With both --export-binned-csvs and --export-binned-jpgs flags:*
```
output_folder/
├── {filename}_SSM.xlsx                              # Main analysis Excel
└── {filename}_binned_analysis/                      # Binned analysis subfolder
    ├── {filename}_binned_mean_power_per_chunk.xlsx
    ├── {filename}_binned_percent_power_per_chunk.xlsx
    ├── {filename}_binned_percent_occupancy_per_chunk.xlsx
    ├── {filename}_binned_dominant_band_per_chunk.xlsx
    ├── {filename}_binned_eois_per_chunk.xlsx
    ├── {filename}_binned_heatmap.jpg
    ├── {filename}_chunk0_mean_power.jpg             # Per-chunk visualizations
    ├── {filename}_chunk0_percent_power.jpg
    ├── {filename}_chunk0_dominant_eoi.jpg
    ├── {filename}_chunk1_mean_power.jpg
    ├── {filename}_chunk1_percent_power.jpg
    ├── {filename}_chunk1_dominant_eoi.jpg
    ├── ... (one set per chunk)
    └── {filename}_polar_occupancy.jpg               # (polar only: occupancy once)
```

**Example Script for Multiple Files:**
```python
import subprocess
import glob

# Process all .eeg files with full binned analysis export
for eeg_file in glob.glob("data/*.eeg"):
    subprocess.run([
        "python", "src/batch_ssm.py", 
        eeg_file,
        "--ppm", "511",
        "--chunk-size", "10",
        "--export-binned-jpgs",
        "-o", "results/"
    ])
```
### Excel/CSV Output Format

### Examples

Two quick-start scripts are included in the repository:

- examples/binned_analysis_example.py: End-to-end demo that runs the 4×4 binned analysis and writes Excel tables plus JPG visualizations using current defaults (window=hann, PPM=511).
- examples/example_binned_analysis.py: Integration pattern showing how to invoke the binned analysis from an existing pipeline and export results (Excel-first plus JPG heatmap).

Outputs are written under examples/outputs/ by default when running the examples.

**Main SSM Data (_SSM.xlsx):**
- **Time Bin (s)**: Time interval (e.g., "0-10", "10-20")
- **Distance Per Bin (cm)**: Distance traveled within that time bin
- **Cumulative Distance (cm)**: Total distance from start
- **Avg [Band] Power**: Average power for each frequency band
- **Percent [Band]**: Percentage of total power for each band

**Binned Analysis Data (Per-Chunk):**
- **Mean Power (_mean_power_per_chunk.xlsx)**: Power values per spatial bin per time chunk (rows=chunks, cols=bins), one sheet per frequency band
- **Percent Power (_percent_power_per_chunk.xlsx)**: Percentage power per spatial bin per chunk, one sheet per frequency band
- **Occupancy (_percent_occupancy_per_chunk.xlsx)**: Percent time spent in each bin per chunk (rows=chunks, cols=bins)
- **Dominant Band (_dominant_band_per_chunk.xlsx)**: Frequency band name with highest power per bin per chunk (rows=chunks, cols=bins)
- **EOIs (_eois_per_chunk.xlsx)**: Count of EOI events per spatial bin per chunk (rows=chunks, cols=bins) - only if EOI file detected

**Data Format (Per-Chunk Files):**
- **Column 1**: Chunk number (1-indexed)
- **Columns 2-17** (4×4 bins): Values for each spatial bin (flattened row-major order)
  - Bins ordered: [0,0], [0,1], [0,2], [0,3], [1,0], ..., [3,3]
- **Rows**: One per time chunk
- **Example**: 120-second recording, 60-second chunks → 2 rows + header

### Notes
- **Excel export is now the default** for all outputs (openpyxl automatically installed if missing, CSV fallback available)
- **Per-chunk binned data**: All exports include per-chunk granularity (mean power, percent power, occupancy, dominant band, EOIs)
- **Binned analysis** automatically exported during processing (GUI and batch mode)
- **Per-chunk visualizations** available via "Export All JPGs" in GUI or `--export-binned-jpgs` flag in batch mode
- **Arena shape detection**: Automatically detects circular vs rectangular arenas based on position trajectory
  - Circular/ellipse → Polar binning (2 rings × 8 sectors)
  - Rectangle/square → Cartesian binning (4×4 grid)
- **EOI support**: Automatically detects and maps EOI (HFO, seizure events, etc.) to spatial bins per chunk
  - Searches for `{filename}_EOI.csv` or `HFOScores/{filename}/{filename}_*.txt`
  - EOI counts exported per chunk in dedicated Excel sheet
- Position data (`.pos` files) should be sampled at 50 Hz
- EEG data formats supported: Axona `.eeg` (250 Hz) and `.egf` (1200 Hz)
- **Batch processing**: Handles position data files with mismatched timestamp arrays (automatically trims or extends as needed)
- **Directory mode**: When processing a directory, prioritizes `.egf` files over `.eeg` and skips duplicate variants (e.g., .egf2-.egf4)
- **Consistency**: 4×4 rectangular bins now export occupancy and EOI per chunk, matching polar bin behavior

### Troubleshooting
- **Import errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`
- **Qt errors**: Make sure PyQt5 is properly installed
- **Distance calculation issues**: Verify PPM value matches your position tracking system

### License
See LICENSE file for details. 

