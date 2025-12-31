# Real-Time Training GUI - Quick Start

## How to Enable

Add the `--gui` flag to your train-dl command:

```bash
python -m stlar train-dl \
  --train training_data/manifest_train.csv \
  --val training_data/manifest_val.csv \
  --epochs 20 \
  --gui
```

## What Happens

1. **Training starts** - Console shows: "✓ Real-time training GUI enabled"
2. **Window opens** - GUI displays with 4 plots and metrics dashboard
3. **Each epoch** - Plots update automatically, metrics refresh
4. **Training completes** - GUI stays open for review, button turns green "Close"
5. **Close window** - Exit program

## GUI Layout

```
┌─────────────────────────────────────────────────────────────┐
│                  🧠 Training Monitor                        │
├─────────────────────────────────────────────────────────────┤
│ Epoch: 10/20 | Train: 0.3245 | Val: 0.3512 | Gap: +0.0267  │
│ Improvement: -0.0023 | Best Val Loss: 0.3512 ✓             │
├──────────────────────────────┬──────────────────────────────┤
│                              │                              │
│  Train vs Val Loss           │  Loss Improvement           │
│  (blue & red lines)          │  (green line)               │
│                              │                              │
├──────────────────────────────┼──────────────────────────────┤
│                              │                              │
│  Generalization Gap          │  Training Stability         │
│  (magenta line)              │  (blue & red lines)         │
│                              │                              │
├─────────────────────────────────────────────────────────────┤
│ Diagnostics Log:                                            │
│ Epoch 1: Train=0.6234, Val=0.6421 ✓ NEW BEST              │
│ Epoch 2: Train=0.4521, Val=0.4712 ✓ NEW BEST              │
│ ⚠️  OVERFITTING detected                                   │
├─────────────────────────────────────────────────────────────┤
│                  [ ⏹ Stop Training ]                        │
└─────────────────────────────────────────────────────────────┘
```

## Using the Stop Button

### During Training
1. Click "⏹ Stop Training" (red button)
2. Console shows: "⏹ Training stopped by user at epoch X"
3. Current epoch finishes
4. Training halts gracefully
5. Best model is saved
6. Plots are generated

### After Training
- Button changes to "Close" (green)
- Click to exit the application
- Or close window normally with X button

## Features

### Live Metrics Display

The top bar shows real-time values:
- **Epoch**: Current/Total (e.g., 10/20)
- **Train Loss**: Current training loss
- **Val Loss**: Current validation loss
- **Gap**: Val - Train (positive = overfitting warning)
- **Improvement**: Change in val loss from previous epoch
- **Best Val Loss**: Lowest validation loss achieved (✓ when new best)

When a new best model is saved, the text turns green and shows ✓.

### Live Plots

**Plot 1: Training vs Validation Loss**
- X-axis: Epoch number
- Y-axis: Loss value
- Blue line: Training loss
- Red line: Validation loss
- Grid for easy reading

**Plot 2: Loss Improvement**
- Shows val_loss[i-1] - val_loss[i]
- Positive = improvement
- Negative = getting worse
- Dashed line at zero

**Plot 3: Generalization Gap**
- Shows val_loss - train_loss
- Above zero = overfitting risk
- Below zero = underfitting
- Red zone above zero line

**Plot 4: Training Stability**
- Rolling standard deviation (window=3)
- Blue: training loss variance
- Red: validation loss variance
- Low values = stable training

### Diagnostics Log

Automatically logs:
- ✅ Each epoch completion with metrics
- ✅ New best model saves (✓ NEW BEST)
- ⚠️ Overfitting warnings
- ⚠️ Plateau warnings (no improvement)
- ⚠️ Instability warnings (high variance)
- ⚠️ Early stopping triggers
- ⏹ User-requested stops

Auto-scrolls to show latest messages.

## Example Session

```bash
$ python -m stlar train-dl --train data/train.csv --val data/val.csv --epochs 20 --gui

Using device: cuda
✓ Real-time training GUI enabled

============================================================
Starting training...
============================================================

[GUI window opens]

Epoch  1/20 | Train: 0.6234 | Val: 0.6421 | Gap: +0.0187 | Δ: +0.0000
  ✓ New best! Saved to models\best.pt
Epoch  2/20 | Train: 0.4521 | Val: 0.4712 | Gap: +0.0191 | Δ: -0.1709
  ✓ New best! Saved to models\best.pt
...
Epoch 15/20 | Train: 0.3401 | Val: 0.3598 | Gap: +0.0197 | Δ: +0.0001

⚠️  Early stopping: No improvement for 5 epochs

============================================================
Training complete!
Best epoch: 10 | Best val loss: 0.3512
============================================================

Generating training visualizations...
✓ Saved training curves to models\training_curves.png
✓ Saved training metrics to models\training_metrics.json

📊 Training Diagnostics:
----------------------------------------
✓ No significant overfitting
⚠️  PLATEAU detected
   Val loss not improving
   → Try: Reduce --lr by 2-5x
✓ Training is stable
----------------------------------------

Done. Best val loss: 0.3512

💡 GUI window is open. Close it to exit.

[Program waits for GUI window to close]
```

## Combining with Other Options

### GUI + Custom Hyperparameters

```bash
python -m stlar train-dl \
  --train data/train.csv \
  --val data/val.csv \
  --epochs 30 \
  --batch-size 32 \
  --lr 5e-4 \
  --weight-decay 1e-3 \
  --gui
```

### GUI + Custom Output Directory

```bash
python -m stlar train-dl \
  --train data/train.csv \
  --val data/val.csv \
  --out-dir experiments/run_001 \
  --gui
```

### GUI Without Static Plots

```bash
python -m stlar train-dl \
  --train data/train.csv \
  --val data/val.csv \
  --gui \
  --no-plot
```

Note: `--no-plot` only disables the PNG/JSON files, the GUI still works!

## Troubleshooting

### GUI doesn't appear

**Check PyQt5 installation:**
```bash
python -c "import PyQt5; print('OK')"
```

If error, install:
```bash
pip install PyQt5 pyqtgraph
```

**Check if requirements met:**
- PyQt5 >= 5.15.0 (already in requirements.txt)
- pyqtgraph >= 0.12.0 (already in requirements.txt)

### GUI is slow/laggy

- Reduce `--num-workers` to 0 or 1
- Close other applications
- Training itself isn't affected, only GUI updates

### GUI doesn't close

- Click "Close" button (after training completes)
- Or press Alt+F4 / close window with X
- Or Ctrl+C in terminal (forceful)

### Import error: "cannot import name 'create_training_gui'"

- Check that `training_gui.py` exists in `hfoGUI/dl_training/`
- Verify Python can find the module: `python -c "from hfoGUI.dl_training.training_gui import create_training_gui; print('OK')"`

### Window appears but is blank

- Wait a few seconds for initialization
- Check terminal for error messages
- Try without GUI first to verify data loads correctly

## Tips

1. **Resize window** - Drag corners to make plots larger
2. **Screenshots** - Use OS screenshot tool (Win+Shift+S on Windows) to capture plots
3. **Multiple runs** - Close GUI between runs, or use different terminals
4. **Remote training** - GUI requires display; use SSH with X11 forwarding or run without `--gui`
5. **Background training** - Don't use `--gui` for background jobs (e.g., `nohup` or `screen`)

## Comparison: With and Without GUI

| Feature | `--gui` | Default |
|---------|---------|---------|
| Live plots during training | ✅ Yes | ❌ No |
| Static PNG plots | ✅ Yes* | ✅ Yes |
| JSON metrics | ✅ Yes | ✅ Yes |
| Manual stop button | ✅ Yes | ❌ No (Ctrl+C) |
| Console output | ✅ Yes | ✅ Yes |
| Requires display | ✅ Yes | ❌ No |
| Blocks until closed | ✅ Yes | ❌ No |

*Unless `--no-plot` is also used

## Next Steps

- See [TRAINING_VISUALIZATION.md](TRAINING_VISUALIZATION.md) for detailed visualization guide
- Check [README.md](../README.md) for complete DL workflow
- Try different hyperparameters and watch training dynamics in real-time!
