# Training Visualization Feature

## Overview

The `train-dl` command includes two visualization modes to help you:
- **Real-time GUI**: Interactive window with live loss curves during training
- **Static plots**: PNG images and JSON metrics saved after training
- Monitor training progress visually
- Detect overfitting, plateaus, and instability automatically
- Make informed hyperparameter tuning decisions
- Track training history for reproducibility

## Quick Start

### Real-Time GUI (Recommended for Interactive Training)

```bash
python -m stlar train-dl \
  --train data/manifest_train.csv \
  --val data/manifest_val.csv \
  --epochs 20 \
  --gui
```

A window will open showing live training progress with:
- 4 synchronized plots updating every epoch
- Current metrics dashboard
- Diagnostics log with warnings
- Manual stop button

### Static Plots Only (Default)

```bash
python -m stlar train-dl \
  --train data/manifest_train.csv \
  --val data/manifest_val.csv \
  --epochs 20
```

Generates `training_curves.png` and `training_metrics.json` after training.

## Real-Time GUI Features

### What You See

The GUI window displays:

**Top Section - Current Metrics:**
```
Epoch: 10/20 | Train Loss: 0.3245 | Val Loss: 0.3512 | Gap: +0.0267 | Improvement: -0.0023 | Best Val Loss: 0.3512 ✓
```

**Plot Grid (2x2):**
1. **Top-Left**: Training vs Validation Loss curves (blue and red lines)
2. **Top-Right**: Validation loss improvement per epoch (green line)
3. **Bottom-Left**: Generalization gap (val - train, magenta line)
4. **Bottom-Right**: Training stability (rolling std dev)

**Diagnostics Log:**
```
Training monitor initialized. Waiting for training to start...
Epoch 1: Train=0.6234, Val=0.6421, Gap=+0.0187 ✓ NEW BEST
Epoch 2: Train=0.4521, Val=0.4712, Gap=+0.0191 ✓ NEW BEST
...
⚠️  PLATEAU detected - No improvement
```

**Control Button:**
- Red "⏹ Stop Training" button - Click to halt training gracefully after current epoch

### Interactive Features

**Manual Stop:**
1. Click "⏹ Stop Training" button anytime
2. Training completes current epoch then stops
3. Best model is still saved
4. Plots and metrics are still generated

**Post-Training Review:**
- GUI stays open after training completes
- Button changes to green "Close"
- Review all plots before closing
- Take screenshots if needed

**Live Updates:**
- Plots update after each epoch (not real-time within epoch)
- New best models are highlighted in green
- Warnings appear in diagnostics log immediately

### When to Use GUI

✅ **Good for:**
- Interactive experimentation with hyperparameters
- Long training runs (15+ epochs) where you want to monitor
- Learning/understanding training dynamics
- Presentations or demonstrations
- When you need to stop training early based on visual inspection

❌ **Skip GUI for:**
- Batch processing multiple models
- Remote servers without display
- Automated scripts/pipelines
- Very short training runs (< 5 epochs)

## Static Visualization (Post-Training)

### Automatic Training History Tracking

Every training run now records:
- Train loss per epoch
- Validation loss per epoch
- Best epoch and best validation loss
- Early stopping information

### 2. Real-time Console Diagnostics

Enhanced training output shows:
```
Epoch 10/20 | Train: 0.3245 | Val: 0.3512 | Gap: +0.0267 | Δ: -0.0023
  ✓ New best! Saved to models/best.pt
```

- **Gap**: Difference between val and train loss (overfitting indicator)
- **Δ**: Improvement from previous epoch (plateau detector)

### 3. Visual Diagnostic Plots

Automatically generates `training_curves.png` with 4 panels:

#### Panel 1: Training vs Validation Loss
- Blue line: training loss
- Red line: validation loss
- Automatically flags overfitting when detected

#### Panel 2: Loss Improvement per Epoch
- Shows validation loss delta (how much it improved)
- Detects plateaus when improvement < 0.001 for 5+ epochs

#### Panel 3: Generalization Gap
- Plots val_loss - train_loss over time
- Red zone indicates overfitting region
- Ideal: small, stable gap near zero

#### Panel 4: Training Stability
- Shows rolling standard deviation of losses
- Detects unstable training (high variance)
- Flags when std > 0.1

### 4. Post-Training Diagnostic Report

After training completes, you get an automatic diagnosis:

```
📊 Training Diagnostics:
----------------------------------------
✓ No significant overfitting
⚠️  PLATEAU detected
   Val loss not improving
   → Try: Reduce --lr by 2-5x
✓ Training is stable
----------------------------------------
```

### 5. JSON Metrics Export

Saves complete training history to `training_metrics.json`:

```json
{
  "train_loss": [0.6234, 0.4521, 0.3845, 0.3512, ...],
  "val_loss": [0.6421, 0.4712, 0.3912, 0.3598, ...],
  "best_epoch": 10,
  "best_val_loss": 0.3512
}
```

### 6. Early Stopping Implementation

Training now automatically stops if validation loss doesn't improve for 5 consecutive epochs:

```
⚠️  Early stopping: No improvement for 5 epochs
```

## Usage

### Basic Usage

Training automatically generates visualizations:

```bash
python -m stlar train-dl \
  --train data/manifest_train.csv \
  --val data/manifest_val.csv \
  --epochs 20
```

**Outputs:**
- `models/best.pt` - Best model checkpoint
- `models/training_curves.png` - Diagnostic plots
- `models/training_metrics.json` - Training history

### Disable Plots

For batch processing or headless servers:

```bash
python -m stlar train-dl \
  --train data/manifest_train.csv \
  --val data/manifest_val.csv \
  --no-plot
```

### Custom Visualization

Load the JSON history for custom analysis:

```python
import json
import matplotlib.pyplot as plt

# Load training history
with open('models/training_metrics.json') as f:
    history = json.load(f)

# Create custom plot
plt.figure(figsize=(10, 6))
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.axvline(history['best_epoch'] - 1, color='red', 
            linestyle='--', label=f'Best Epoch ({history["best_epoch"]})')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training History')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('custom_plot.png', dpi=150)
```

## Diagnostic Interpretation

### ⚠️ OVERFITTING Detected

**What it means:** Model memorizes training data but doesn't generalize well
- Validation loss much higher than training loss
- Gap > 0.05 between val and train

**Solutions:**
1. Increase regularization: `--weight-decay 1e-3`
2. Add more training data (run prepare-dl on more sessions)
3. Reduce model complexity (contact developers)

### ⚠️ PLATEAU Detected

**What it means:** Model stopped learning
- Validation loss not improving for 5+ epochs
- All recent improvements < 0.001

**Solutions:**
1. Reduce learning rate: `--lr 2e-4` (was 1e-3)
2. Try learning rate schedule (future feature)
3. May indicate model has converged (check if val loss is acceptable)

### ⚠️ INSTABILITY Detected

**What it means:** Training is noisy/chaotic
- Large swings in loss values
- High variance in recent losses

**Solutions:**
1. Reduce batch size: `--batch-size 32` (was 64)
2. Reduce learning rate: `--lr 5e-4` (was 1e-3)
3. Check data quality (corrupted segments?)

## Example Workflow

### Initial Training

```bash
python -m stlar train-dl \
  --train data/manifest_train.csv \
  --val data/manifest_val.csv \
  --epochs 20 \
  --lr 1e-3 \
  --batch-size 64
```

### Check Diagnostics

Open `models/training_curves.png` and review the console output.

### Adjust Based on Findings

If overfitting detected:
```bash
python -m stlar train-dl \
  --train data/manifest_train.csv \
  --val data/manifest_val.csv \
  --epochs 20 \
  --lr 1e-3 \
  --batch-size 64 \
  --weight-decay 1e-3  # Increased from 1e-4
```

If plateau detected:
```bash
python -m stlar train-dl \
  --train data/manifest_train.csv \
  --val data/manifest_val.csv \
  --epochs 20 \
  --lr 2e-4  # Reduced from 1e-3
  --batch-size 64
```

If instability detected:
```bash
python -m stlar train-dl \
  --train data/manifest_train.csv \
  --val data/manifest_val.csv \
  --epochs 20 \
  --lr 5e-4  # Slightly reduced
  --batch-size 32  # Reduced from 64
```

## Requirements

The visualization feature requires matplotlib, which is already in `requirements.txt`:

```bash
pip install matplotlib
```

If matplotlib is not installed, training will still work but plots will be skipped with a warning message.

## Technical Details

### Implementation

All visualization logic is in `hfoGUI/dl_training/train.py`:

- `plot_training_curves(history, out_dir)` - Generates 4-panel diagnostic plot
- Early stopping logic in main training loop
- History tracking via Python lists/dicts
- JSON export using standard library

### Performance Impact

Minimal overhead:
- History tracking: ~100 bytes per epoch
- Plot generation: 1-2 seconds at end of training
- No impact on training speed

### Customization

To modify plot appearance, edit `plot_training_curves()` in train.py:
- Change colors: Modify line color codes ('b-', 'r-', etc.)
- Adjust thresholds: Edit detection logic (0.05 for overfitting, 0.001 for plateau)
- Add metrics: Extend history dict and plotting code

## Comparison: Before vs After

### Before
```
Epoch 1: train_loss=0.6234 val_loss=0.6421
Epoch 2: train_loss=0.4521 val_loss=0.4712
...
Done. Best val loss: 0.3512
```

### After
```
============================================================
Starting training...
============================================================

Epoch  1/20 | Train: 0.6234 | Val: 0.6421 | Gap: +0.0187 | Δ: +0.0000
  ✓ New best! Saved to models/best.pt
Epoch  2/20 | Train: 0.4521 | Val: 0.4712 | Gap: +0.0191 | Δ: -0.1709
  ✓ New best! Saved to models/best.pt
...
Epoch 15/20 | Train: 0.3401 | Val: 0.3598 | Gap: +0.0197 | Δ: +0.0001

⚠️  Early stopping: No improvement for 5 epochs

============================================================
Training complete!
Best epoch: 10 | Best val loss: 0.3512
============================================================

Generating training visualizations...
✓ Saved training curves to models/training_curves.png
✓ Saved training metrics to models/training_metrics.json

📊 Training Diagnostics:
----------------------------------------
✓ No significant overfitting
⚠️  PLATEAU detected
   Val loss not improving
   → Try: Reduce --lr by 2-5x
✓ Training is stable
----------------------------------------
```

## Future Enhancements

Potential additions (not yet implemented):
- Real-time GUI with live updating plots (requested by user)
- Learning rate scheduling based on plateau detection
- Tensorboard integration for experiment tracking
- Confusion matrix and accuracy metrics (requires threshold tuning)
- Training time estimation and ETA display

## FAQ

**Q: Do I need to install anything extra?**
A: No, matplotlib is already in requirements.txt. Just `pip install -r requirements.txt`.

**Q: Can I disable the plots?**
A: Yes, use the `--no-plot` flag.

**Q: Where are the plots saved?**
A: In the same directory as the model checkpoint (`--out-dir`, default: `models/`).

**Q: Can I create plots after training?**
A: Yes, load `training_metrics.json` and use matplotlib to recreate plots (see Custom Visualization section).

**Q: What if training stops early?**
A: Early stopping is automatic if no improvement for 5 epochs. Check diagnostics to see why (plateau, converged, etc.).

**Q: How do I compare multiple training runs?**
A: Load multiple `training_metrics.json` files and plot them together (see Custom Visualization example).

## Support

For issues or feature requests related to training visualization:
1. Check that matplotlib is installed: `pip list | grep matplotlib`
2. Review the diagnostic messages in console output
3. Open `training_curves.png` to visually inspect training
4. Share `training_metrics.json` when reporting issues
