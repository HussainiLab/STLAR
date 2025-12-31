"""
Real-time training monitoring GUI for deep learning training.
Shows live loss curves, metrics, and diagnostics during training.
"""

import sys

# CRITICAL: Import QApplication first and create it before any other Qt imports
from PyQt5.QtWidgets import QApplication

# Ensure QApplication exists before importing other Qt classes
_app_instance = None

def _ensure_qapp():
    """Ensure QApplication exists before creating any Qt widgets."""
    global _app_instance
    if _app_instance is None:
        _app_instance = QApplication.instance()
        if _app_instance is None:
            _app_instance = QApplication(sys.argv)
    return _app_instance

# Delay other Qt imports until after QApplication is created
# Store class definitions as None, will be populated in _init_classes()
_TrainingMonitor = None
_TrainingGUI = None

def _init_classes():
    """Initialize Qt classes after QApplication is created."""
    global _TrainingMonitor, _TrainingGUI
    
    if _TrainingMonitor is not None:
        return  # Already initialized
    
    # Now import other Qt classes after QApplication exists
    from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, 
                                 QHBoxLayout, QLabel, QTextEdit, QPushButton, QGroupBox)
    from PyQt5.QtCore import QTimer, pyqtSignal, QObject, Qt
    from PyQt5.QtGui import QFont
    import pyqtgraph as pg
    
    class TrainingMonitor(QObject):
        """Signal emitter for training updates."""
        epoch_update = pyqtSignal(int, float, float, dict)  # epoch, train_loss, val_loss, diagnostics
        training_complete = pyqtSignal(dict)  # final history
        
        def __init__(self):
            super().__init__()
            self.stop_requested = False
    
    
    class TrainingGUI(QMainWindow):
        """Real-time training visualization window."""
        
        def __init__(self, total_epochs, monitor):
            super().__init__()
            self.total_epochs = total_epochs
            self.monitor = monitor
            
            # Data storage
            self.epochs = []
            self.train_losses = []
            self.val_losses = []
            self.gaps = []
            
            self.init_ui()
            
            # Connect signals
            self.monitor.epoch_update.connect(self.update_plots)
            self.monitor.training_complete.connect(self.on_training_complete)
        
        def init_ui(self):
            """Initialize the GUI layout."""
            self.setWindowTitle("Training Monitor - Deep Learning HFO Detector")
            self.setGeometry(100, 100, 1200, 800)
            
            # Central widget
            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            main_layout = QVBoxLayout(central_widget)
            
            # Title
            title = QLabel("🧠 Training Monitor")
            title_font = QFont()
            title_font.setPointSize(16)
            title_font.setBold(True)
            title.setFont(title_font)
            title.setAlignment(Qt.AlignCenter)
            main_layout.addWidget(title)
            
            # Metrics panel
            metrics_group = QGroupBox("Current Metrics")
            metrics_layout = QHBoxLayout()
            
            self.epoch_label = QLabel("Epoch: 0/{}".format(self.total_epochs))
            self.train_loss_label = QLabel("Train Loss: --")
            self.val_loss_label = QLabel("Val Loss: --")
            self.gap_label = QLabel("Gap: --")
            self.improvement_label = QLabel("Improvement: --")
            self.best_label = QLabel("Best Val Loss: --")
            
            for label in [self.epoch_label, self.train_loss_label, self.val_loss_label, 
                          self.gap_label, self.improvement_label, self.best_label]:
                label.setFont(QFont("Consolas", 10))
                metrics_layout.addWidget(label)
            
            metrics_group.setLayout(metrics_layout)
            main_layout.addWidget(metrics_group)
            
            # Plot area (2x2 grid)
            plot_widget = QWidget()
            plot_layout = QVBoxLayout(plot_widget)
            
            # Row 1: Loss curves and Improvement
            row1 = QHBoxLayout()
            
            # Plot 1: Train vs Val Loss
            self.loss_plot = pg.PlotWidget(title="Training vs Validation Loss")
            self.loss_plot.setLabel('left', 'Loss')
            self.loss_plot.setLabel('bottom', 'Epoch')
            self.loss_plot.addLegend()
            self.loss_plot.showGrid(x=True, y=True, alpha=0.3)
            self.train_curve = self.loss_plot.plot(pen=pg.mkPen('b', width=2), name='Train Loss')
            self.val_curve = self.loss_plot.plot(pen=pg.mkPen('r', width=2), name='Val Loss')
            row1.addWidget(self.loss_plot)
            
            # Plot 2: Loss Improvement
            self.improvement_plot = pg.PlotWidget(title="Validation Loss Improvement")
            self.improvement_plot.setLabel('left', 'Improvement')
            self.improvement_plot.setLabel('bottom', 'Epoch')
            self.improvement_plot.showGrid(x=True, y=True, alpha=0.3)
            self.improvement_curve = self.improvement_plot.plot(pen=pg.mkPen('g', width=2))
            # Add zero line
            self.improvement_plot.addLine(y=0, pen=pg.mkPen('k', style=Qt.DashLine))
            row1.addWidget(self.improvement_plot)
            
            plot_layout.addLayout(row1)
            
            # Row 2: Gap and Stability
            row2 = QHBoxLayout()
            
            # Plot 3: Generalization Gap
            self.gap_plot = pg.PlotWidget(title="Generalization Gap (Val - Train)")
            self.gap_plot.setLabel('left', 'Gap')
            self.gap_plot.setLabel('bottom', 'Epoch')
            self.gap_plot.showGrid(x=True, y=True, alpha=0.3)
            self.gap_curve = self.gap_plot.plot(pen=pg.mkPen('m', width=2))
            # Add zero line
            self.gap_plot.addLine(y=0, pen=pg.mkPen('k', style=Qt.DashLine))
            row2.addWidget(self.gap_plot)
            
            # Plot 4: Training Stability
            self.stability_plot = pg.PlotWidget(title="Training Stability (Rolling Std)")
            self.stability_plot.setLabel('left', 'Std Dev')
            self.stability_plot.setLabel('bottom', 'Epoch')
            self.stability_plot.showGrid(x=True, y=True, alpha=0.3)
            self.train_stability_curve = self.stability_plot.plot(pen=pg.mkPen('b', width=2), name='Train')
            self.val_stability_curve = self.stability_plot.plot(pen=pg.mkPen('r', width=2), name='Val')
            row2.addWidget(self.stability_plot)
            
            plot_layout.addLayout(row2)
            main_layout.addWidget(plot_widget)
            
            # Diagnostics log
            log_group = QGroupBox("Diagnostics Log")
            log_layout = QVBoxLayout()
            self.log_text = QTextEdit()
            self.log_text.setReadOnly(True)
            self.log_text.setMaximumHeight(150)
            self.log_text.setFont(QFont("Consolas", 9))
            log_layout.addWidget(self.log_text)
            log_group.setLayout(log_layout)
            main_layout.addWidget(log_group)
            
            # Control buttons
            button_layout = QHBoxLayout()
            self.stop_button = QPushButton("⏹ Stop Training")
            self.stop_button.setStyleSheet("background-color: #ff6b6b; color: white; font-weight: bold;")
            self.stop_button.clicked.connect(self.request_stop)
            button_layout.addStretch()
            button_layout.addWidget(self.stop_button)
            button_layout.addStretch()
            main_layout.addLayout(button_layout)
            
            self.log("Training monitor initialized. Waiting for training to start...")
        
        def log(self, message):
            """Add message to diagnostics log."""
            self.log_text.append(message)
            # Auto-scroll to bottom
            self.log_text.verticalScrollBar().setValue(
                self.log_text.verticalScrollBar().maximum()
            )
        
        def update_plots(self, epoch, train_loss, val_loss, diagnostics):
            """Update plots and metrics with new epoch data."""
            # Store data
            self.epochs.append(epoch)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            gap = val_loss - train_loss
            self.gaps.append(gap)
            
            # Calculate improvement
            improvement = self.val_losses[-2] - val_loss if len(self.val_losses) > 1 else 0
            
            # Update metrics labels
            self.epoch_label.setText(f"Epoch: {epoch}/{self.total_epochs}")
            self.train_loss_label.setText(f"Train Loss: {train_loss:.4f}")
            self.val_loss_label.setText(f"Val Loss: {val_loss:.4f}")
            self.gap_label.setText(f"Gap: {gap:+.4f}")
            self.improvement_label.setText(f"Improvement: {improvement:+.4f}")
            
            if diagnostics.get('is_best', False):
                self.best_label.setText(f"Best Val Loss: {val_loss:.4f} ✓")
                self.best_label.setStyleSheet("color: green; font-weight: bold;")
            else:
                self.best_label.setText(f"Best Val Loss: {diagnostics.get('best_val_loss', '--'):.4f}")
                self.best_label.setStyleSheet("")
            
            # Update loss curves
            self.train_curve.setData(self.epochs, self.train_losses)
            self.val_curve.setData(self.epochs, self.val_losses)
            
            # Update improvement plot
            if len(self.val_losses) > 1:
                improvements = [self.val_losses[i-1] - self.val_losses[i] 
                              for i in range(1, len(self.val_losses))]
                self.improvement_curve.setData(self.epochs[1:], improvements)
            
            # Update gap plot
            self.gap_curve.setData(self.epochs, self.gaps)
            
            # Update stability plot
            window = 3
            if len(self.train_losses) >= window:
                train_stability = []
                val_stability = []
                stability_epochs = []
                for i in range(window - 1, len(self.train_losses)):
                    train_window = self.train_losses[i-window+1:i+1]
                    val_window = self.val_losses[i-window+1:i+1]
                    train_mean = sum(train_window) / window
                    val_mean = sum(val_window) / window
                    train_std = (sum((x - train_mean)**2 for x in train_window) / window)**0.5
                    val_std = (sum((x - val_mean)**2 for x in val_window) / window)**0.5
                    train_stability.append(train_std)
                    val_stability.append(val_std)
                    stability_epochs.append(self.epochs[i])
                
                self.train_stability_curve.setData(stability_epochs, train_stability)
                self.val_stability_curve.setData(stability_epochs, val_stability)
            
            # Log epoch results
            log_msg = f"Epoch {epoch}: Train={train_loss:.4f}, Val={val_loss:.4f}, Gap={gap:+.4f}"
            if diagnostics.get('is_best', False):
                log_msg += " ✓ NEW BEST"
            self.log(log_msg)
            
            # Log warnings
            if diagnostics.get('overfitting', False):
                self.log("⚠️  OVERFITTING detected - Val loss >> Train loss")
            if diagnostics.get('plateau', False):
                self.log("⚠️  PLATEAU detected - No improvement")
            if diagnostics.get('instability', False):
                self.log("⚠️  INSTABILITY detected - High variance")
            if diagnostics.get('early_stop', False):
                self.log(f"⚠️  Early stopping triggered after epoch {epoch}")
            
            # Process events to keep GUI responsive
            QApplication.processEvents()
        
        def on_training_complete(self, history):
            """Handle training completion."""
            self.log("\n" + "="*50)
            self.log("Training Complete!")
            self.log(f"Best Epoch: {history['best_epoch']}")
            self.log(f"Best Val Loss: {history['best_val_loss']:.4f}")
            self.log("="*50)
            
            self.stop_button.setText("Close")
            self.stop_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
            self.stop_button.clicked.disconnect()
            self.stop_button.clicked.connect(self.close)
        
        def request_stop(self):
            """Request training to stop."""
            self.monitor.stop_requested = True
            self.log("\n⏹ Stop requested. Training will halt after current epoch...")
            self.stop_button.setEnabled(False)
            self.stop_button.setText("Stopping...")
        
        def closeEvent(self, event):
            """Handle window close event."""
            if not self.stop_button.text() == "Close":
                # Training is still running, request stop
                self.request_stop()
            event.accept()
    
    _TrainingMonitor = TrainingMonitor
    _TrainingGUI = TrainingGUI


def create_training_gui(total_epochs):
    """
    Create and show training GUI.
    
    Returns:
        (QApplication, TrainingGUI, TrainingMonitor) tuple
    """
    # Ensure QApplication exists before creating any Qt objects
    app = _ensure_qapp()
    
    # Initialize classes after QApplication is created
    _init_classes()
    
    # Now create Qt objects
    monitor = _TrainingMonitor()
    gui = _TrainingGUI(total_epochs, monitor)
    gui.show()
    
    # Process events to show the window
    app.processEvents()
    
    return app, gui, monitor
