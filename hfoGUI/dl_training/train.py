import argparse
import os
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .data import SegmentDataset, pad_collate_fn, pad_collate_fn_2d
from .model import build_model
import sys

# Fix OpenMP library conflict (multiple libiomp5md.dll instances)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def parse_args():
    p = argparse.ArgumentParser(description="Train a 1D CNN HFO classifier")
    p.add_argument('--train', required=True, help='Path to train manifest CSV')
    p.add_argument('--val', required=True, help='Path to val manifest CSV')
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--out-dir', type=str, default='models')
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--no-plot', action='store_true', help='Disable training curve plots')
    p.add_argument('--model-type', type=int, default=2, help='Model architecture: 1=SimpleCNN, 2=ResNet1D, 3=InceptionTime, 4=Transformer, 5=2D_CNN, 6=HFO_2D_CNN')
    p.add_argument('--use-cwt', action='store_true', help='Use CWT/Scalogram preprocessing for 2D models')
    p.add_argument('--fs', type=float, default=4800.0, help='Sampling frequency, required for CWT')
    p.add_argument('--gui', action='store_true', help='Show real-time training GUI')
    p.add_argument('--debug-cwt', type=str, default=None, help='(Debug) Directory to save CWT scalogram images for inspection')
    return p.parse_args()


def plot_training_curves(history, out_dir):
    """Generate training visualization plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed, skipping plots. Install with: pip install matplotlib")
        return
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Training Diagnostics', fontsize=16, fontweight='bold')
    
    # 1. Loss curves (train vs val)
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training vs Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Highlight overfitting if present
    if len(history['val_loss']) > 5:
        recent_train = history['train_loss'][-5:]
        recent_val = history['val_loss'][-5:]
        if sum(recent_val) / 5 > sum(recent_train) / 5 * 1.2:
            ax1.text(0.5, 0.95, '[WARN] Possible Overfitting Detected', 
                    transform=ax1.transAxes, ha='center', va='top',
                    bbox=dict(boxstyle='round', facecolor='orange', alpha=0.7))
    
    # 2. Loss improvement rate (detect plateaus)
    ax2 = axes[0, 1]
    val_improvements = []
    for i in range(1, len(history['val_loss'])):
        improvement = history['val_loss'][i-1] - history['val_loss'][i]
        val_improvements.append(improvement)
    
    if val_improvements:
        ax2.plot(range(2, len(history['val_loss']) + 1), val_improvements, 'g-', linewidth=2)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Val Loss Improvement')
        ax2.set_title('Validation Loss Improvement per Epoch')
        ax2.grid(True, alpha=0.3)
        
        # Detect plateau
        if len(val_improvements) >= 5:
            recent_improvements = val_improvements[-5:]
            if all(abs(imp) < 0.001 for imp in recent_improvements):
                ax2.text(0.5, 0.95, '[WARN] Loss Plateau Detected', 
                        transform=ax2.transAxes, ha='center', va='top',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # 3. Train/Val gap (overfitting indicator)
    ax3 = axes[1, 0]
    gap = [v - t for t, v in zip(history['train_loss'], history['val_loss'])]
    ax3.plot(epochs, gap, 'm-', linewidth=2)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Val Loss - Train Loss')
    ax3.set_title('Generalization Gap (Overfitting Indicator)')
    ax3.grid(True, alpha=0.3)
    
    # Annotate overfitting zone
    ax3.fill_between(epochs, 0, gap, where=[g > 0 for g in gap], 
                     alpha=0.3, color='red', label='Overfitting zone')
    ax3.legend()
    
    # 4. Training stability (loss variance)
    ax4 = axes[1, 1]
    # Calculate rolling standard deviation
    window = 3
    if len(history['train_loss']) >= window:
        train_stability = []
        val_stability = []
        for i in range(window - 1, len(history['train_loss'])):
            train_window = history['train_loss'][i-window+1:i+1]
            val_window = history['val_loss'][i-window+1:i+1]
            train_stability.append(sum((x - sum(train_window)/window)**2 for x in train_window)**0.5)
            val_stability.append(sum((x - sum(val_window)/window)**2 for x in val_window)**0.5)
        
        stability_epochs = range(window, len(history['train_loss']) + 1)
        ax4.plot(stability_epochs, train_stability, 'b-', label='Train Stability', linewidth=2)
        ax4.plot(stability_epochs, val_stability, 'r-', label='Val Stability', linewidth=2)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss Std Dev (window=3)')
        ax4.set_title('Training Stability')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Detect instability
        if train_stability and max(train_stability) > 0.1:
            ax4.text(0.5, 0.95, '[WARN] Training Unstable', 
                    transform=ax4.transAxes, ha='center', va='top',
                    bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))
    
    plt.tight_layout()
    
    # Save figure
    plot_path = Path(out_dir) / 'training_curves.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Saved training curves to {plot_path}")
    plt.close()
    
    # Also save metrics as JSON for programmatic access
    import json
    metrics_path = Path(out_dir) / 'training_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"[OK] Saved training metrics to {metrics_path}")


def train_one_epoch(model, loader, opt, device):
    model.train()
    total_loss = 0.0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        opt.zero_grad()
        logit = model(x)
        loss = F.binary_cross_entropy_with_logits(logit.squeeze(-1), y)
        loss.backward()
        opt.step()
        total_loss += loss.item() * x.size(0)
        total += x.size(0)
    return total_loss / max(total, 1)


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logit = model(x)
            loss = F.binary_cross_entropy_with_logits(logit.squeeze(-1), y)
            total_loss += loss.item() * x.size(0)
            total += x.size(0)
    return total_loss / max(total, 1)


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize GUI if requested
    gui_app = None
    gui_window = None
    monitor = None
    if args.gui:
        try:
            # Initialize QApplication before importing GUI modules that might create widgets
            from PyQt5.QtWidgets import QApplication
            gui_app = QApplication.instance()
            if gui_app is None:
                gui_app = QApplication(sys.argv)

            from .training_gui import create_training_gui
            _, gui_window, monitor = create_training_gui(args.epochs)
            print("[OK] Real-time training GUI enabled")
        except ImportError as e:
            print(f"Warning: Could not load GUI: {e}")
            print("Continuing without GUI...")
            args.gui = False
    
    if args.use_cwt:
        if args.model_type < 5: # Assuming 5 and 6 are 2D models
            print("Warning: --use-cwt is enabled but a 1D model type is selected. This will likely fail.")
        train_ds = SegmentDataset(args.train, use_cwt=True, fs=args.fs, debug_cwt_dir=args.debug_cwt)
        val_ds = SegmentDataset(args.val, use_cwt=True, fs=args.fs, debug_cwt_dir=args.debug_cwt)
        collate_fn = pad_collate_fn_2d
    else:
        train_ds = SegmentDataset(args.train)
        val_ds = SegmentDataset(args.val)
        collate_fn = pad_collate_fn

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    model = build_model(args.model_type).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float('inf')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = out_dir / 'best.pt'
    
    # Save training parameters to JSON for tracking
    import json
    import datetime
    training_params = {
        'timestamp': datetime.datetime.now().isoformat(),
        'device': str(device),
        'train_manifest': str(args.train),
        'val_manifest': str(args.val),
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'model_type': args.model_type,
        'model_type_name': {1: 'SimpleCNN', 2: 'ResNet1D', 3: 'InceptionTime', 4: 'Transformer', 5: '2D_CNN', 6: 'HFO_2D_CNN'}.get(args.model_type, 'Unknown'),
        'num_workers': args.num_workers,
        'train_samples': len(train_ds),
        'val_samples': len(val_ds),
        'output_dir': str(out_dir),
    }
    
    params_file = out_dir / 'training_params.json'
    with open(params_file, 'w') as f:
        json.dump(training_params, f, indent=2)
    print(f"[OK] Saved training parameters to {params_file}\n")
    
    # Training history tracking
    best_epoch = 0
    epochs_since_improvement = 0
    patience = 5  # Early stopping patience
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'best_epoch': 0,
        'best_val_loss': float('inf')
    }
    
    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, opt, device)
        val_loss = evaluate(model, val_loader, device)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Calculate diagnostics
        overfitting_gap = val_loss - train_loss
        improvement = history['val_loss'][-2] - val_loss if len(history['val_loss']) > 1 else 0
        is_best = False
        
        # Print epoch summary
        print(f"Epoch {epoch:2d}/{args.epochs} | "
              f"Train: {train_loss:.4f} | "
              f"Val: {val_loss:.4f} | "
              f"Gap: {overfitting_gap:+.4f} | "
              f"Improve: {improvement:+.4f}")
        
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            epochs_since_improvement = 0
            is_best = True
            torch.save({
                'model_state': model.state_dict(),
                'model_type': args.model_type
            }, best_ckpt)
            print(f"  [NEW BEST] Saved to {best_ckpt}")
        else:
            epochs_since_improvement += 1
        
        # Prepare diagnostics for GUI
        diagnostics = {
            'is_best': is_best,
            'best_val_loss': best_val,
            'overfitting': overfitting_gap > 0.05,
            'plateau': False,
            'instability': False,
            'early_stop': False
        }
        
        # Check for plateau
        if len(history['val_loss']) >= 5:
            recent_improvements = [history['val_loss'][i-1] - history['val_loss'][i] 
                                  for i in range(-4, 0)]
            if all(abs(imp) < 0.001 for imp in recent_improvements):
                diagnostics['plateau'] = True
        
        # Check for instability
        if len(history['train_loss']) >= 3:
            recent_train = history['train_loss'][-3:]
            train_mean = sum(recent_train) / 3
            train_std = (sum((x - train_mean)**2 for x in recent_train) / 3)**0.5
            if train_std > 0.1:
                diagnostics['instability'] = True
        
        # Update GUI if enabled
        if args.gui and monitor:
            monitor.epoch_update.emit(epoch, train_loss, val_loss, diagnostics)
            gui_app.processEvents()  # Keep GUI responsive
            
            # Check if user requested stop
            if monitor.stop_requested:
                print(f"\n[STOP] Training stopped by user at epoch {epoch}")
                diagnostics['early_stop'] = True
                break
            
        # Early stopping check
        if epochs_since_improvement >= patience:
            print(f"\n[EARLY STOP] No improvement for {patience} epochs")
            diagnostics['early_stop'] = True
            if args.gui and monitor:
                monitor.epoch_update.emit(epoch, train_loss, val_loss, diagnostics)
                gui_app.processEvents()
            break
    
    # Update final history
    history['best_epoch'] = best_epoch
    history['best_val_loss'] = best_val
    
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"Best epoch: {best_epoch} | Best val loss: {best_val:.4f}")
    print(f"{'='*60}\n")
    
    # Notify GUI of completion
    if args.gui and monitor:
        monitor.training_complete.emit(history)
        gui_app.processEvents()
    
    # Generate visualizations
    if not args.no_plot and len(history['train_loss']) > 1:
        print("Generating training visualizations...")
        plot_training_curves(history, out_dir)
    
    # Print diagnostic summary
    print("\n[DIAGNOSTICS] Training Summary:")
    print("-" * 40)
    
    # Check for overfitting
    final_gap = history['val_loss'][-1] - history['train_loss'][-1]
    if final_gap > 0.05:
        print("[WARN] OVERFITTING detected")
        print(f"   Val loss > Train loss by {final_gap:.4f}")
        if args.weight_decay >= 1e-3:
            print(f"   -> weight_decay already at {args.weight_decay}, try:")
            print("      - Increase weight_decay further (to 1e-2)")
            print("      - Add dropout layers to model")
            print("      - Collect more training data")
            print("      - Reduce model complexity")
        else:
            print(f"   -> Try: Increase --weight-decay (current: {args.weight_decay})")
    else:
        print("[OK] No significant overfitting")
    
    # Check for plateau
    if len(history['val_loss']) >= 5:
        recent_improvements = [history['val_loss'][-i-1] - history['val_loss'][-i] 
                              for i in range(1, 5)]
        if all(abs(imp) < 0.001 for imp in recent_improvements):
            print("[WARN] PLATEAU detected")
            print("   Val loss not improving")
            suggested_lr = args.lr / 2
            print(f"   -> Try: Reduce --lr to {suggested_lr:.2e} (from {args.lr:.2e})")
        else:
            print("[OK] Val loss still improving")
    
    # Check for instability
    if len(history['train_loss']) >= 3:
        recent_train = history['train_loss'][-3:]
        train_mean = sum(recent_train) / 3
        train_std = (sum((x - train_mean)**2 for x in recent_train) / 3)**0.5
        if train_std > 0.1:
            print("[WARN] INSTABILITY detected")
            print(f"   High loss variance: {train_std:.4f}")
            suggested_batch = max(16, args.batch_size // 2)
            print(f"   -> Try: Reduce --batch-size to {suggested_batch} (from {args.batch_size})")
        else:
            print("[OK] Training is stable")
    
    print("-" * 40)

    print("Done. Best val loss: {:.4f}".format(best_val))
    
    # Update training parameters file with final results
    training_params['best_epoch'] = best_epoch
    training_params['best_val_loss'] = float(best_val)
    training_params['final_train_loss'] = float(history['train_loss'][-1])
    training_params['final_val_loss'] = float(history['val_loss'][-1])
    training_params['total_epochs_run'] = len(history['train_loss'])
    training_params['overfitting_detected'] = (history['val_loss'][-1] - history['train_loss'][-1]) > 0.05
    training_params['best_checkpoint'] = str(best_ckpt)
    
    # Add tuning recommendations based on diagnostics
    recommendations = []
    final_gap = history['val_loss'][-1] - history['train_loss'][-1]
    if final_gap > 0.05:
        if args.weight_decay >= 1e-3:
            recommendations.append(f"Overfitting (gap={final_gap:.4f}): weight_decay already at {args.weight_decay}, try 1e-2 or add dropout")
        else:
            recommendations.append(f"Overfitting (gap={final_gap:.4f}): Increase weight_decay to 1e-3 or higher")
    
    if len(history['val_loss']) >= 5:
        recent_improvements = [history['val_loss'][i-1] - history['val_loss'][i] 
                              for i in range(-4, 0)]
        if all(abs(imp) < 0.001 for imp in recent_improvements):
            suggested_lr = args.lr / 2
            recommendations.append(f"Plateau detected: Reduce learning_rate to {suggested_lr:.2e}")
    
    if len(history['train_loss']) >= 3:
        recent_train = history['train_loss'][-3:]
        train_std = (sum((x - sum(recent_train)/3)**2 for x in recent_train) / 3)**0.5
        if train_std > 0.1:
            suggested_batch = max(16, args.batch_size // 2)
            recommendations.append(f"Instability (std={train_std:.4f}): Reduce batch_size to {suggested_batch}")
    
    training_params['tuning_recommendations'] = recommendations if recommendations else ["Training looks good!"]
    
    with open(params_file, 'w') as f:
        json.dump(training_params, f, indent=2)
    print(f"[OK] Updated training parameters with final results: {params_file}")
    
    # Keep GUI open if enabled
    if args.gui and gui_app and gui_window:
        print("\n[INFO] GUI window is open. Close it to exit.")
        gui_app.exec_()  # Block until GUI is closed    # Update final history
    history['best_epoch'] = best_epoch
    history['best_val_loss'] = best_val
    
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"Best epoch: {best_epoch} | Best val loss: {best_val:.4f}")
    print(f"{'='*60}\n")
    
    # Generate visualizations
    if not args.no_plot and len(history['train_loss']) > 1:
        print("Generating training visualizations...")
        plot_training_curves(history, out_dir)
    
    # Print diagnostic summary
    print("\n[DIAGNOSTICS] Training Summary:")
    print("-" * 40)
    
    # Check for overfitting
    final_gap = history['val_loss'][-1] - history['train_loss'][-1]
    if final_gap > 0.05:
        print("[WARN] OVERFITTING detected")
        print(f"   Val loss > Train loss by {final_gap:.4f}")
        print("   -> Try: Increase --weight-decay to 1e-3")
    else:
        print("[OK] No significant overfitting")
    
    # Check for plateau
    if len(history['val_loss']) >= 5:
        recent_improvements = [history['val_loss'][i-1] - history['val_loss'][i] 
                              for i in range(-4, 0)]
        if all(abs(imp) < 0.001 for imp in recent_improvements):
            print("[WARN] PLATEAU detected")
            print("   Val loss not improving")
            print("   -> Try: Reduce --lr by 2-5x")
        else:
            print("[OK] Val loss still improving")
    
    # Check for instability
    if len(history['train_loss']) >= 3:
        recent_train = history['train_loss'][-3:]
        train_std = (sum((x - sum(recent_train)/3)**2 for x in recent_train) / 3)**0.5
        if train_std > 0.1:
            print("[WARN] INSTABILITY detected")
            print(f"   High loss variance: {train_std:.4f}")
            print("   -> Try: Reduce --batch-size to 32")
        else:
            print("[OK] Training is stable")
    
    print("-" * 40)

    print("Done. Best val loss: {:.4f}".format(best_val))


if __name__ == '__main__':
    main()
