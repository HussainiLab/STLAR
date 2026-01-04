from pyqtgraph.Qt import QtGui, QtCore, QtWidgets

# Import Signal and Slot from the Qt backend
try:
    from PyQt5.QtCore import pyqtSignal, pyqtSlot, QObject
except ImportError:
    try:
        from PyQt6.QtCore import pyqtSignal, pyqtSlot, QObject
    except ImportError:
        from PySide6.QtCore import Signal as pyqtSignal, Slot as pyqtSlot, QObject

from core.GUI_Utils import background, center, find_consec
import os, time, json, functools, sys, csv
from scipy.signal import hilbert, welch
import numpy as np
from core.Tint_Matlab import detect_peaks
from core.GUI_Utils import Worker
import pandas as pd
import core.filtering as filt
from core.Detector import ste_detect_events, mni_detect_events, dl_detect_events

# Try to import torch and hfo_detection for DL features
try:
    import torch
    from torch.utils.data import DataLoader
except ImportError:
    torch = None

try:
    # Import CWT inference dataset and model building from consolidated training module
    # Use sibling import pattern consistent with other core imports (e.g., "from core.GUI_Utils")
    from dl_training.data import CWT_InferenceDataset
    from dl_training.model import build_model
except ImportError:
    # Fallback for when package is installed or run as module
    try:
        from ..dl_training.data import CWT_InferenceDataset
        from ..dl_training.model import build_model
    except ImportError:
        CWT_InferenceDataset = None
        build_model = None

try:
    # Legacy fallback for hfo_detection (deprecated)
    import hfo_detection
except ImportError:
    hfo_detection = None

class TreeWidgetItem(QtWidgets.QTreeWidgetItem):
    """This subclass was created so that the __lt__ method could be overwritten so that the numerical data values
    are treated correctly (not as strings)"""
    def __init__(self, parent=None):
        QtWidgets.QTreeWidgetItem.__init__(self, parent)

    def __lt__(self, otherItem):
        column = self.treeWidget().sortColumn()
        text1 = self.text(column)
        text2 = otherItem.text(column)
        
        # Try to extract numeric portion for ID columns or fully numeric values
        try:
            # First try direct numeric conversion
            return float(text1) < float(text2)
        except ValueError:
            # If that fails, try to extract numeric suffix (for IDs like "H_1", "H_10", etc.)
            try:
                # Extract the numeric part after underscore or other separators
                num1 = float(''.join(c for c in text1 if c.isdigit() or c == '.'))
                num2 = float(''.join(c for c in text2 if c.isdigit() or c == '.'))
                return num1 < num2
            except (ValueError, IndexError):
                # Fall back to string comparison
                return text1 < text2


class AddItemSignal(QObject):
    """This is a signal was created so that we could add QTreeWidgetItems
    from the main thread since it did not like that we were adding EOIs
    from a thread"""
    childAdded = pyqtSignal(object)


class custom_signal(QObject):
    """This method will contain the signal that will allow for the linear region selector to be
    where the current score/EOI is so the user can change if they want"""

    set_lr_signal = pyqtSignal(str, str)


class ProgressSignal(QObject):
    """Signal for detection progress updates."""
    progress = pyqtSignal(str)  # Emits status message like "Hilbert: 45%"


class RegionPresetDialog(QtWidgets.QDialog):
    """Dialog for reviewing and fine-tuning region preset parameters before applying."""
    
    def __init__(self, parent, region_name, preset_data):
        super().__init__(parent)
        self.setWindowTitle(f"Region Preset - {region_name}")
        self.preset_data = preset_data.copy()
        self.modified_preset = None
        self.field_widgets = {}
        
        self.setMinimumWidth(500)
        layout = QtWidgets.QVBoxLayout()
        
        info_label = QtWidgets.QLabel(
            f"<b>Configure parameters for {region_name}</b><br>"
            "Adjust band limits, durations, gating, and DL export options before applying."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Scroll area for all parameters
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QtWidgets.QWidget()
        form_layout = QtWidgets.QFormLayout()

        # Bands
        form_layout.addRow(QtWidgets.QLabel("<b>Bands (Hz)</b>"))
        bands = preset_data.get('bands', {})
        for band_name, bounds in bands.items():
            min_val, max_val = (bounds + [0, 0])[:2] if isinstance(bounds, (list, tuple)) else (0.0, 0.0)
            row_layout = QtWidgets.QHBoxLayout()
            min_spin = QtWidgets.QDoubleSpinBox()
            min_spin.setRange(0.0, 1000.0)
            min_spin.setSingleStep(5.0)
            min_spin.setValue(min_val)
            max_spin = QtWidgets.QDoubleSpinBox()
            max_spin.setRange(0.0, 1000.0)
            max_spin.setSingleStep(5.0)
            max_spin.setValue(max_val)
            self.field_widgets[f'bands_{band_name}_min'] = min_spin
            self.field_widgets[f'bands_{band_name}_max'] = max_spin
            row_layout.addWidget(QtWidgets.QLabel("Min:"))
            row_layout.addWidget(min_spin)
            row_layout.addWidget(QtWidgets.QLabel("Max:"))
            row_layout.addWidget(max_spin)
            row_layout.addStretch()
            row_widget = QtWidgets.QWidget()
            row_widget.setLayout(row_layout)
            form_layout.addRow(f"{band_name.replace('_', ' ').title()}:", row_widget)

        # Durations
        form_layout.addRow(QtWidgets.QLabel("<b>Durations (ms)</b>"))
        durations = preset_data.get('durations', {})
        for dur_name, dur_val in durations.items():
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(1.0, 2000.0)
            spin.setSingleStep(5.0)
            spin.setValue(dur_val)
            self.field_widgets[f'durations_{dur_name}'] = spin
            form_layout.addRow(f"{dur_name.replace('_', ' ')}:", spin)

        # Detection params
        form_layout.addRow(QtWidgets.QLabel("<b>Detection defaults</b>"))
        thr_spin = QtWidgets.QDoubleSpinBox()
        thr_spin.setRange(0.1, 10.0)
        thr_spin.setSingleStep(0.1)
        thr_spin.setValue(preset_data.get('threshold_sd', 3.5))
        self.field_widgets['threshold_sd'] = thr_spin
        form_layout.addRow("Threshold (SD):", thr_spin)

        epoch_spin = QtWidgets.QDoubleSpinBox()
        epoch_spin.setRange(1.0, 3600.0)
        epoch_spin.setSingleStep(10.0)
        epoch_spin.setDecimals(1)
        epoch_spin.setValue(preset_data.get('epoch_s', 300.0))
        self.field_widgets['epoch_s'] = epoch_spin
        form_layout.addRow("Epoch (s):", epoch_spin)
        form_layout.addRow(QtWidgets.QLabel(""))  # Spacer
        behavior_label = QtWidgets.QLabel("<b>Behavioral Gating</b>")
        form_layout.addRow(behavior_label)
        
        # Behavior gating checkbox
        gating_check = QtWidgets.QCheckBox("Enable behavior gating")
        gating_check.setChecked(preset_data.get('behavior_gating', True))
        self.field_widgets['behavior_gating'] = gating_check
        form_layout.addRow("  ", gating_check)
        
        # Speed threshold
        speed_layout = QtWidgets.QHBoxLayout()
        speed_widget = QtWidgets.QWidget()
        speed_widget.setLayout(speed_layout)
        
        speed_min_spin = QtWidgets.QDoubleSpinBox()
        speed_min_spin.setRange(0.0, 100.0)
        speed_min_spin.setSingleStep(0.5)
        speed_min_spin.setValue(preset_data.get('speed_threshold_min_cm_s', 0.0))
        speed_min_spin.setSuffix(" cm/s")
        self.field_widgets['speed_threshold_min_cm_s'] = speed_min_spin
        
        speed_max_spin = QtWidgets.QDoubleSpinBox()
        speed_max_spin.setRange(0.0, 100.0)
        speed_max_spin.setSingleStep(0.5)
        speed_max_spin.setValue(preset_data.get('speed_threshold_max_cm_s', 5.0))
        speed_max_spin.setSuffix(" cm/s")
        self.field_widgets['speed_threshold_max_cm_s'] = speed_max_spin
        
        speed_layout.addWidget(QtWidgets.QLabel("Min:"))
        speed_layout.addWidget(speed_min_spin)
        speed_layout.addWidget(QtWidgets.QLabel("Max:"))
        speed_layout.addWidget(speed_max_spin)
        speed_layout.addStretch()
        
        form_layout.addRow("  Speed Range:", speed_widget)
        
        # DL Export options section
        form_layout.addRow(QtWidgets.QLabel(""))  # Spacer
        dl_label = QtWidgets.QLabel("<b>Deep Learning Export Options</b>")
        form_layout.addRow(dl_label)
        
        dl_export = preset_data.get('dl_export', {})
        
        filter_dur_check = QtWidgets.QCheckBox("Filter by duration")
        filter_dur_check.setChecked(dl_export.get('filter_by_duration', True))
        self.field_widgets['dl_filter_by_duration'] = filter_dur_check
        form_layout.addRow("  ", filter_dur_check)
        
        annotate_check = QtWidgets.QCheckBox("Annotate band")
        annotate_check.setChecked(dl_export.get('annotate_band', True))
        self.field_widgets['dl_annotate_band'] = annotate_check
        form_layout.addRow("  ", annotate_check)
        
        dl_gating_check = QtWidgets.QCheckBox("Apply behavior gating")
        dl_gating_check.setChecked(dl_export.get('behavior_gating', True))
        self.field_widgets['dl_behavior_gating'] = dl_gating_check
        form_layout.addRow("  ", dl_gating_check)
        
        scroll_widget.setLayout(form_layout)
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        
        apply_btn = QtWidgets.QPushButton("Apply")
        apply_btn.clicked.connect(self.accept)
        apply_btn.setDefault(True)
        
        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        
        button_layout.addStretch()
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(apply_btn)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
    
    def get_modified_preset(self):
        """Extract values from all widgets and return the modified preset."""
        preset = self.preset_data.copy()
        
        # Update bands
        bands = {}
        band_names = list(self.preset_data.get('bands', {}).keys())
        for band_name in band_names:
            min_val = self.field_widgets[f'bands_{band_name}_min'].value()
            max_val = self.field_widgets[f'bands_{band_name}_max'].value()
            bands[band_name] = [min_val, max_val]
        preset['bands'] = bands
        
        # Update durations
        durations = {}
        duration_names = list(self.preset_data.get('durations', {}).keys())
        for dur_name in duration_names:
            durations[dur_name] = self.field_widgets[f'durations_{dur_name}'].value()
        preset['durations'] = durations
        
        # Update detection parameters
        preset['threshold_sd'] = self.field_widgets['threshold_sd'].value()
        preset['epoch_s'] = self.field_widgets['epoch_s'].value()
        
        # Update behavioral gating
        preset['behavior_gating'] = self.field_widgets['behavior_gating'].isChecked()
        preset['speed_threshold_min_cm_s'] = self.field_widgets['speed_threshold_min_cm_s'].value()
        preset['speed_threshold_max_cm_s'] = self.field_widgets['speed_threshold_max_cm_s'].value()
        
        # Update DL export
        preset['dl_export'] = {
            'filter_by_duration': self.field_widgets['dl_filter_by_duration'].isChecked(),
            'annotate_band': self.field_widgets['dl_annotate_band'].isChecked(),
            'behavior_gating': self.field_widgets['dl_behavior_gating'].isChecked(),
        }
        
        return preset


class ScoreWindow(QtWidgets.QWidget):
    '''This is the window that will pop up to score the '''

    def __init__(self, main, settings):
        super(ScoreWindow, self).__init__()

        self.AddItemSignal = AddItemSignal()
        self.AddItemSignal.childAdded.connect(self.add_item)

        self.mainWindow = main

        self.customSignals = custom_signal()
        self.customSignals.set_lr_signal.connect(self.mainWindow.set_lr)

        self.progressSignal = ProgressSignal()
        self.progressSignal.progress.connect(self._on_detection_progress)

        self.settingsWindow = settings

        self.initialize_attributes()

        # Region-specific analysis presets (Phase 1)
        self.region_presets = self._build_region_presets()
        self.current_region = 'None'
        self.region_profile = self.region_presets.get(self.current_region, {}).copy()

        background(self)
        width = int(self.deskW / 6)
        height = int(self.deskH / 1.5)

        self.setWindowTitle(
            os.path.splitext(os.path.basename(__file__))[0] + " - Score Window")  # sets the title of the window

        main_location = main.frameGeometry().getCoords()

        self.setGeometry(main_location[2], main_location[1]+30, width, height)

        tabs = QtWidgets.QTabWidget()
        score_tab = QtWidgets.QWidget()
        eoi_tab = QtWidgets.QWidget()

        # ----------------------- score filename widget - score tab ----------------------------------------

        self.save_score_btn = QtWidgets.QPushButton('Save Scores', self)
        self.save_score_btn.clicked.connect(self.saveScores)

        self.load_score_btn = QtWidgets.QPushButton('Load Scores', self)
        self.load_score_btn.clicked.connect(self.loadScores)

        score_filename_btn_layout = QtWidgets.QHBoxLayout()
        score_filename_btn_layout.addWidget(self.load_score_btn)
        score_filename_btn_layout.addWidget(self.save_score_btn)

        score_filename_label = QtWidgets.QLabel('Score Filename:')
        self.score_filename = QtWidgets.QLineEdit()
        self.score_filename.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        self.score_filename.setText("Please add a source!")

        score_filename_layout = QtWidgets.QHBoxLayout()
        score_filename_layout.addWidget(score_filename_label)
        score_filename_layout.addWidget(self.score_filename)

        scorer_filename_label = QtWidgets.QLabel('Scorer:')
        self.scorer = QtWidgets.QLineEdit()

        scorer_layout = QtWidgets.QHBoxLayout()
        scorer_layout.addWidget(scorer_filename_label)
        scorer_layout.addWidget(self.scorer)

        source_label = QtWidgets.QLabel('Source:')
        self.source = QtWidgets.QComboBox()
        self.source.setEditable(True)
        self.source.lineEdit().setReadOnly(True)
        self.source.lineEdit().setAlignment(QtCore.Qt.AlignHCenter)
        self.source.currentIndexChanged.connect(self.changeSources)
        self.source.setSizePolicy(QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed))
        self.source.addItem("None")
        source_layout = QtWidgets.QHBoxLayout()
        source_layout.addWidget(source_label)
        source_layout.addWidget(self.source)

        # -------------------------- scores widget --------------------------------------

        self.scores = QtWidgets.QTreeWidget()
        self.scores.setSortingEnabled(True)
        self.scores.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.scores.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.scores.customContextMenuRequested.connect(functools.partial(self.openMenu, 'score'))

        self.scores.itemSelectionChanged.connect(functools.partial(self.changeEventText, 'score'))

        self.score_headers = {
            'ID#:': 0,
            'Score:': 1,
            'Start Time(ms):': 2,
            'Stop Time(ms):': 3,
            'Duration(ms):': 4,
            'Behavioral State:': 5,
            'Scorer:': 6,
            'Settings File:': 7,
        }

        for key, value in self.score_headers.items():
            self.scores.headerItem().setText(value, key)
            if 'Start Time' in key:
                self.scores.sortItems(value, QtCore.Qt.AscendingOrder)

        # ----------------------- scoring widgets -------------------------

        self.score = QtWidgets.QComboBox()
        self.score.setEditable(True)
        self.score.lineEdit().setReadOnly(True)
        self.score.lineEdit().setAlignment(QtCore.Qt.AlignHCenter)
        self.score.setSizePolicy(QtWidgets.QSizePolicy(
                        QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed))

        score_values = ['None', 'Spike', 'Theta', 'Gamma Low', 'Gamma High', 'Ripple', 'Fast Ripple', 'Sharp Wave Ripple',
                        'Artifact',
                        'Other']

        self.id_abbreviations = {
            'manual': 'MAN', 
            'hilbert': 'HIL', 
            'ste': 'STE',
            'mni': 'MNI',
            'consensus': 'CON',
            'deep learning': 'DL',
            'unknown': 'UNK'
        }

        for score in score_values:
            self.score.addItem(score)

        score_label = QtWidgets.QLabel("Score:")

        score_layout = QtWidgets.QHBoxLayout()
        score_layout.addWidget(score_label)
        score_layout.addWidget(self.score)

        # -------- Region Preset for Score exports --------
        region_label = QtWidgets.QLabel("Region Preset:")
        self.score_region_selector = QtWidgets.QComboBox()
        self.score_region_selector.addItem("None")
        for region_name in self.region_presets.keys():
            self.score_region_selector.addItem(region_name)
        self.score_region_selector.setCurrentText(self.current_region)
        self.score_region_selector.currentTextChanged.connect(self._on_region_changed)

        self.score_apply_region_btn = QtWidgets.QPushButton("Configure Preset")
        self.score_apply_region_btn.setToolTip("Configure region-specific defaults for export filtering and metrics.")
        self.score_apply_region_btn.clicked.connect(self.openRegionPresetDialog)

        region_layout = QtWidgets.QHBoxLayout()
        region_layout.addWidget(region_label)
        region_layout.addWidget(self.score_region_selector)
        region_layout.addWidget(self.score_apply_region_btn)

        # ------------------------------button layout --------------------------------------
        self.hide_btn = QtWidgets.QPushButton('Hide', self)
        self.add_btn = QtWidgets.QPushButton('Add Score', self)
        self.add_btn.clicked.connect(self.addScore)
        self.update_btn = QtWidgets.QPushButton('Update Selected Scores')
        self.update_btn.clicked.connect(self.updateScores)
        self.delete_btn = QtWidgets.QPushButton('Delete Selected Scores', self)
        self.delete_btn.clicked.connect(self.deleteScores)
        self.export_metrics_btn = QtWidgets.QPushButton('Export HFO Metrics (CSV)', self)
        self.export_metrics_btn.clicked.connect(self.exportHFOMetrics)
        self.export_dl_btn = QtWidgets.QPushButton('Export for DL Training', self)
        self.export_dl_btn.clicked.connect(self.exportForDLTraining)

        btn_layout = QtWidgets.QHBoxLayout()

        # Keep Hide at the end, after exports
        for button in [self.add_btn, self.update_btn, self.delete_btn, self.export_metrics_btn, self.export_dl_btn, self.hide_btn]:
            btn_layout.addWidget(button)
        # ------------------ layout ------------------------------

        layout_order = [score_filename_btn_layout, score_filename_layout, scorer_layout, region_layout, self.scores, score_layout,
                btn_layout]

        layout_score = QtWidgets.QVBoxLayout()
        for order in layout_order:
            if 'Layout' in order.__str__():
                layout_score.addLayout(order)
            else:
                layout_score.addWidget(order)

        # ------- EOI widgets -----------

        eoi_filename_label = QtWidgets.QLabel('EOI Filename:')
        self.eoi_filename = QtWidgets.QLineEdit()
        self.eoi_filename.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        self.eoi_filename.setText("Please add a source!")

        eoi_filename_layout = QtWidgets.QHBoxLayout()
        eoi_filename_layout.addWidget(eoi_filename_label)
        eoi_filename_layout.addWidget(self.eoi_filename)

        self.save_eoi_btn = QtWidgets.QPushButton("Save EOIs")
        self.save_eoi_btn.clicked.connect(self.saveAutomaticEOIs)
        eoi_button_layout = QtWidgets.QHBoxLayout()

        self.load_eois = QtWidgets.QPushButton("Load EOIs")
        self.load_eois.clicked.connect(self.loadAutomaticEOIs)

        eoi_button_layout.addWidget(self.load_eois)
        eoi_button_layout.addWidget(self.save_eoi_btn)

        self.EOI_score = QtWidgets.QComboBox()
        self.EOI_score.setEditable(True)
        self.EOI_score.lineEdit().setReadOnly(True)
        self.EOI_score.lineEdit().setAlignment(QtCore.Qt.AlignHCenter)
        self.EOI_score.setSizePolicy(QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed))

        for score in score_values:
            self.EOI_score.addItem(score)

        EOI_score_label = QtWidgets.QLabel("Score:")

        EOI_score_layout = QtWidgets.QHBoxLayout()
        EOI_score_layout.addWidget(EOI_score_label)
        EOI_score_layout.addWidget(self.EOI_score)

        eoi_method_label = QtWidgets.QLabel("EOI Detection Method:")
        self.eoi_method = QtWidgets.QComboBox()
        self.eoi_method.currentIndexChanged.connect(self.setEOIfilename)
        methods = ['Hilbert', 'STE', 'MNI', 'Consensus', 'Deep Learning']

        events_detected_label = QtWidgets.QLabel('Events Detected:')
        self.events_detected = QtWidgets.QLineEdit()

        events_detected_layout = QtWidgets.QHBoxLayout()
        events_detected_layout.addWidget(events_detected_label)
        events_detected_layout.addWidget(self.events_detected)
        self.events_detected.setText('0')
        self.events_detected.setEnabled(0)

        for method in methods:
            self.eoi_method.addItem(method)

        eoi_method_layout = QtWidgets.QHBoxLayout()
        eoi_method_layout.addWidget(eoi_method_label)
        eoi_method_layout.addWidget(self.eoi_method)

        eoi_parameter_layout = QtWidgets.QHBoxLayout()
        eoi_parameter_layout.addLayout(eoi_method_layout)
        eoi_parameter_layout.addLayout(events_detected_layout)

        self.EOI = QtWidgets.QTreeWidget()
        self.EOI.setSortingEnabled(True)
        self.EOI.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.EOI.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.EOI.customContextMenuRequested.connect(functools.partial(self.openMenu, 'EOI'))
        self.EOI_headers = {
            'ID#:': 0,
            'Start Time(ms):': 1,
            'Stop Time(ms):': 2,
            'Duration(ms):': 3,
            'Settings File:': 4,
        }

        for key, value in self.EOI_headers.items():
            self.EOI.headerItem().setText(value, key)
            if 'Start Time' in key:
                self.EOI.sortItems(value, QtCore.Qt.AscendingOrder)

        self.EOI.itemSelectionChanged.connect(functools.partial(self.changeEventText, 'EOI'))

        # ----- EOI tab buttons ---------------
        self.update_eoi_region = QtWidgets.QPushButton("Update EOI Region")
        self.update_eoi_region.setToolTip("This will modify the times based on the current selected window")
        self.update_eoi_region.clicked.connect(self.updateEOIRegion)

        self.eoi_hide = QtWidgets.QPushButton("Hide")
        self.find_eoi_btn = QtWidgets.QPushButton("Find EOIs")
        self.find_eoi_btn.clicked.connect(self.findEOIs)
        self.delete_eoi_btn = QtWidgets.QPushButton("Remove Selected EOI(s)")
        self.delete_eoi_btn.clicked.connect(self.deleteEOI)

        self.add_eoi_btn = QtWidgets.QPushButton("Add Selected EOI(s) to Score")
        self.add_eoi_btn.clicked.connect(self.addEOI)

        btn_layout = QtWidgets.QHBoxLayout()
        for btn in [self.find_eoi_btn, self.add_eoi_btn, self.update_eoi_region, self.delete_eoi_btn, self.eoi_hide]:
            btn_layout.addWidget(btn)

        layout_eoi = QtWidgets.QVBoxLayout()

        for item in [eoi_button_layout, eoi_filename_layout, eoi_parameter_layout, self.EOI, EOI_score_layout, btn_layout]:
            if 'Layout' in item.__str__():
                layout_eoi.addLayout(item)
            else:
                layout_eoi.addWidget(item)

        score_tab.setLayout(layout_score)

        eoi_tab.setLayout(layout_eoi)

        # ------- Convert tab for train/val splitting -----------
        convert_tab = QtWidgets.QWidget()
        convert_layout = QtWidgets.QVBoxLayout()

        convert_title = QtWidgets.QLabel("Convert Manifests to Train/Val Splits")
        convert_title.setStyleSheet("font-weight: bold; font-size: 12pt;")
        convert_layout.addWidget(convert_title)

        convert_help = QtWidgets.QLabel(
            "Load one or more manifest.csv files (from EOI exports) and split into train/val sets.\n"
            "Splits are done by subject to avoid data leakage."
        )
        convert_help.setWordWrap(True)
        convert_layout.addWidget(convert_help)

        # Manifest list
        manifest_label = QtWidgets.QLabel("Selected Manifests:")
        convert_layout.addWidget(manifest_label)

        self.manifest_list = QtWidgets.QListWidget()
        self.manifest_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        convert_layout.addWidget(self.manifest_list)

        # Buttons for manifest management
        manifest_btn_layout = QtWidgets.QHBoxLayout()
        self.add_manifest_btn = QtWidgets.QPushButton("Add Manifest(s)")
        self.add_manifest_btn.clicked.connect(self.addManifests)
        self.remove_manifest_btn = QtWidgets.QPushButton("Remove Selected")
        self.remove_manifest_btn.clicked.connect(self.removeManifests)
        self.clear_manifest_btn = QtWidgets.QPushButton("Clear All")
        self.clear_manifest_btn.clicked.connect(self.clearManifests)
        manifest_btn_layout.addWidget(self.add_manifest_btn)
        manifest_btn_layout.addWidget(self.remove_manifest_btn)
        manifest_btn_layout.addWidget(self.clear_manifest_btn)
        convert_layout.addLayout(manifest_btn_layout)

        # Split options
        options_group = QtWidgets.QGroupBox("Split Options")
        options_layout = QtWidgets.QFormLayout()

        self.val_fraction_spin = QtWidgets.QDoubleSpinBox()
        self.val_fraction_spin.setRange(0.1, 0.5)
        self.val_fraction_spin.setValue(0.2)
        self.val_fraction_spin.setSingleStep(0.05)
        self.val_fraction_spin.setDecimals(2)
        options_layout.addRow("Validation Fraction:", self.val_fraction_spin)

        self.random_seed_spin = QtWidgets.QSpinBox()
        self.random_seed_spin.setRange(0, 99999)
        self.random_seed_spin.setValue(42)
        options_layout.addRow("Random Seed:", self.random_seed_spin)

        self.stratified_check = QtWidgets.QCheckBox("Use stratified split (balance labels)")
        options_layout.addRow("", self.stratified_check)

        options_group.setLayout(options_layout)
        convert_layout.addWidget(options_group)

        # Output directory
        output_layout = QtWidgets.QHBoxLayout()
        output_label = QtWidgets.QLabel("Output Directory:")
        self.output_dir_edit = QtWidgets.QLineEdit()
        self.output_dir_edit.setPlaceholderText("Select output directory...")
        self.output_browse_btn = QtWidgets.QPushButton("Browse...")
        self.output_browse_btn.clicked.connect(self.browseOutputDir)
        output_layout.addWidget(output_label)
        output_layout.addWidget(self.output_dir_edit)
        output_layout.addWidget(self.output_browse_btn)
        convert_layout.addLayout(output_layout)

        # Convert button
        self.convert_btn = QtWidgets.QPushButton("Create Train/Val Split")
        self.convert_btn.clicked.connect(self.convertManifests)
        self.convert_btn.setStyleSheet("font-weight: bold; padding: 8px;")
        convert_layout.addWidget(self.convert_btn)

        # Status output
        self.convert_status = QtWidgets.QTextEdit()
        self.convert_status.setReadOnly(True)
        self.convert_status.setMaximumHeight(150)
        convert_layout.addWidget(self.convert_status)

        convert_tab.setLayout(convert_layout)

        # ------- Train tab for running training/export -----------
        train_tab = QtWidgets.QWidget()
        train_layout = QtWidgets.QVBoxLayout()

        train_title = QtWidgets.QLabel("Train Deep Learning Model")
        train_title.setStyleSheet("font-weight: bold; font-size: 12pt;")
        train_layout.addWidget(train_title)

        train_help = QtWidgets.QLabel(
            "Provide train/val manifests, adjust hyperparameters, and launch training."
        )
        train_help.setWordWrap(True)
        train_layout.addWidget(train_help)

        # File pickers for train/val
        train_form = QtWidgets.QFormLayout()

        self.train_manifest_edit = QtWidgets.QLineEdit()
        train_browse_btn = QtWidgets.QPushButton("Browse…")
        train_browse_btn.clicked.connect(self.browseTrainManifest)
        train_row = QtWidgets.QHBoxLayout()
        train_row.addWidget(self.train_manifest_edit)
        train_row.addWidget(train_browse_btn)
        train_form.addRow("Train manifest (CSV):", train_row)

        self.val_manifest_edit = QtWidgets.QLineEdit()
        val_browse_btn = QtWidgets.QPushButton("Browse…")
        val_browse_btn.clicked.connect(self.browseValManifest)
        val_row = QtWidgets.QHBoxLayout()
        val_row.addWidget(self.val_manifest_edit)
        val_row.addWidget(val_browse_btn)
        train_form.addRow("Val manifest (CSV):", val_row)

        # Hyperparameters
        self.model_type_combo = QtWidgets.QComboBox()
        self.model_type_combo.addItem("Simple 1D CNN", 1)
        self.model_type_combo.addItem("ResNet1D (Default)", 2)
        self.model_type_combo.addItem("InceptionTime", 3)
        self.model_type_combo.addItem("1D Transformer", 4)
        self.model_type_combo.addItem("2D Spectrogram CNN", 5)
        self.model_type_combo.addItem("2D CWT CNN (Scalogram)", 6)
        self.model_type_combo.setCurrentIndex(1)
        train_form.addRow("Model Architecture:", self.model_type_combo)

        self.epochs_spin = QtWidgets.QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(15)
        train_form.addRow("Epochs:", self.epochs_spin)

        self.batch_spin = QtWidgets.QSpinBox()
        self.batch_spin.setRange(1, 2048)
        self.batch_spin.setValue(64)
        train_form.addRow("Batch size:", self.batch_spin)

        self.lr_spin = QtWidgets.QDoubleSpinBox()
        self.lr_spin.setDecimals(6)
        self.lr_spin.setRange(1e-6, 1.0)
        self.lr_spin.setSingleStep(1e-4)
        self.lr_spin.setValue(1e-3)
        train_form.addRow("Learning rate:", self.lr_spin)

        self.weight_decay_spin = QtWidgets.QDoubleSpinBox()
        self.weight_decay_spin.setDecimals(6)
        self.weight_decay_spin.setRange(0.0, 1.0)
        self.weight_decay_spin.setSingleStep(1e-4)
        self.weight_decay_spin.setValue(1e-4)
        self.weight_decay_spin.setToolTip("L2 regularization strength (typical: 1e-4). Increase to 1e-3 or 1e-2 to reduce overfitting.")
        train_form.addRow("Weight decay:", self.weight_decay_spin)

        self.train_cwt_check = QtWidgets.QCheckBox("Use CWT (Scalogram) Preprocessing")
        self.train_cwt_check.setToolTip("Convert 1D EEG segments to 2D CWT Scalograms before training (requires 2D CNN model).")
        train_form.addRow("", self.train_cwt_check)

        self.train_out_dir_edit = QtWidgets.QLineEdit()
        train_out_btn = QtWidgets.QPushButton("Browse…")
        train_out_btn.clicked.connect(self.browseTrainOutDir)
        out_row = QtWidgets.QHBoxLayout()
        out_row.addWidget(self.train_out_dir_edit)
        out_row.addWidget(train_out_btn)
        train_form.addRow("Output dir:", out_row)

        train_layout.addLayout(train_form)

        # GUI visualization checkbox
        self.train_gui_check = QtWidgets.QCheckBox("Show training visualization GUI")
        self.train_gui_check.setToolTip("Opens a real-time visualization window showing loss curves and metrics during training")
        train_layout.addWidget(self.train_gui_check)

        # Train button
        self.train_btn = QtWidgets.QPushButton("Start Training")
        self.train_btn.setStyleSheet("font-weight: bold; padding: 8px;")
        self.train_btn.clicked.connect(self.startTraining)
        train_layout.addWidget(self.train_btn)

        # Export section
        export_title = QtWidgets.QLabel("Export Trained Model (TorchScript + ONNX)")
        export_title.setStyleSheet("font-weight: bold; margin-top: 12px;")
        train_layout.addWidget(export_title)

        export_form = QtWidgets.QFormLayout()

        self.ckpt_edit = QtWidgets.QLineEdit()
        ckpt_browse_btn = QtWidgets.QPushButton("Browse…")
        ckpt_browse_btn.clicked.connect(self.browseCheckpoint)
        ckpt_row = QtWidgets.QHBoxLayout()
        ckpt_row.addWidget(self.ckpt_edit)
        ckpt_row.addWidget(ckpt_browse_btn)
        export_form.addRow("Checkpoint (.pt):", ckpt_row)

        self.export_out_dir_edit = QtWidgets.QLineEdit()
        export_out_btn = QtWidgets.QPushButton("Browse…")
        export_out_btn.clicked.connect(self.browseExportOutDir)
        exp_out_row = QtWidgets.QHBoxLayout()
        exp_out_row.addWidget(self.export_out_dir_edit)
        exp_out_row.addWidget(export_out_btn)
        export_form.addRow("Export dir:", exp_out_row)

        train_layout.addLayout(export_form)

        self.export_btn = QtWidgets.QPushButton("Export Model")
        self.export_btn.clicked.connect(self.startExport)
        train_layout.addWidget(self.export_btn)

        # Logs
        self.train_log = QtWidgets.QTextEdit()
        self.train_log.setReadOnly(True)
        self.train_log.setMaximumHeight(180)
        train_layout.addWidget(self.train_log)

        train_tab.setLayout(train_layout)

        # ------- Help tab -----------
        help_tab = QtWidgets.QWidget()
        help_layout = QtWidgets.QVBoxLayout()

        help_title = QtWidgets.QLabel("Score Window Help")
        help_title.setStyleSheet("font-weight: bold; font-size: 12pt;")
        help_layout.addWidget(help_title)

        self.help_text = QtWidgets.QTextEdit()
        self.help_text.setReadOnly(True)
        self.help_text.setMinimumHeight(220)
        self.help_text.setHtml(
                        """
                        <h2>Score Window Help</h2>
                        <p><b>The Score window provides a complete workflow for HFO detection, labeling, and Deep Learning training:</b></p>
                        
                        <h3>📋 Table of Contents</h3>
                        <ul>
                            <li><a href="#overview">Overview</a></li>
                            <li><a href="#automdetect">Automatic Detection Tab</a></li>
                            <li><a href="#score">Score Tab</a></li>
                            <li><a href="#csvconvert">CSV Convert Tab</a></li>
                            <li><a href="#train">Train Tab</a></li>
                            <li><a href="#workflow-metrics">Workflow: HFO Metrics (Analysis)</a></li>
                            <li><a href="#workflow-dl">Workflow: Deep Learning Training</a></li>
                            <li><a href="#parameters">Detection Parameters & Defaults</a></li>
                        </ul>

                        <h3 id="overview">📌 Overview</h3>
                        <p>All start/stop times are shown and saved in <b>milliseconds (ms)</b>. The workflow consists of:</p>
                        <ol>
                            <li><b>Detection:</b> Run automated algorithms (Hilbert/STE/MNI/Consensus) to find candidate HFOs</li>
                            <li><b>Filtering & Labeling:</b> Apply region presets to filter by duration/behavior and classify as Ripple/Fast Ripple/Artifact</li>
                            <li><b>Export:</b> Export either HFO metrics (CSV) for analysis, or labeled segments for DL training</li>
                            <li><b>Training (optional):</b> Train a Deep Learning model on labeled data and deploy for future detection</li>
                        </ol>

                        <h3 id="automdetect">🔍 Automatic Detection Tab (First Tab)</h3>
                        <p><b>Detects candidate High-Frequency Oscillations (HFOs) using automated algorithms.</b></p>
                        
                        <h4>Detection Methods:</h4>
                        <ul>
                            <li><b>Hilbert:</b> Bandpass + Hilbert envelope analysis; detects via SD threshold and peak count. Best for ripples (80–250 Hz).</li>
                            <li><b>STE:</b> Short-Time Energy detector using windowed RMS thresholding. Detects broadband HFO energy.</li>
                            <li><b>MNI:</b> Percentile-based energy detector; compares baseline to current activity. Robust to slow variations.</li>
                            <li><b>Consensus:</b> Voting across Hilbert + STE + MNI. Choose voting mode:
                                <ul style="margin-left: 20px; margin-top: 5px;">
                                    <li><i>Strict:</i> All three detectors agree (highest specificity)</li>
                                    <li><i>Majority:</i> ≥2 detectors agree (balanced)</li>
                                    <li><i>Any:</i> ≥1 detector fires (highest sensitivity)</li>
                                </ul>
                            </li>
                            <li><b>Deep Learning:</b> Uses a pre-trained TorchScript model (.pt file) exported from the Train tab. Requires prior model training.</li>
                        </ul>

                        <h4>Key Actions:</h4>
                        <ul>
                            <li><b>Run Detection:</b> Click on a detection method and choose band (Ripple/Fast Ripple) or consensus voting strategy.</li>
                            <li><b>Add Selected EOI(s) to Score:</b> Transfers selected EOIs to the Score tab with automatic:
                                <ul style="margin-left: 20px; margin-top: 5px;">
                                    <li>Duration filtering (Ripple vs Fast Ripple classification)</li>
                                    <li>Behavioral gating (marks as 'rest' or 'active' if .pos speed data loaded)</li>
                                    <li>PSD-based band labeling (ripple-power vs fast-ripple-power comparison)</li>
                                </ul>
                            </li>
                            <li><b>Update EOI Region:</b> Modify an EOI's start/stop times using the current graph selection window (useful for refinement).</li>
                        </ul>

                        <h3 id="score">✏️ Score Tab (Second Tab)</h3>
                        <p><b>Manage detected and manually-scored HFOs. Apply region presets, label events, and export results.</b></p>
                        
                        <h4>Main Controls:</h4>
                        <ul>
                            <li><b>Region Preset:</b> Choose a brain region (LEC, Hippocampus, MEC) or None.
                                <ul style="margin-left: 20px; margin-top: 5px;">
                                    <li>Presets define: frequency bands, duration thresholds (ripple vs fast ripple ranges), behavioral gating (speed thresholds)</li>
                                    <li>When applied, automatically re-labels existing scores based on duration and recalculates behavioral states (rest/active) if missing</li>
                                </ul>
                            </li>
                            <li><b>Configure Preset:</b> Customize preset parameters (frequency ranges, duration thresholds, speed gates). Updates are applied when you click "Apply".</li>
                            <li><b>Add Score:</b> Manually add a single event using current graph selection times. Auto-filled with 'Auto' scorer and 'unknown' behavioral state.</li>
                            <li><b>Update Selected Scores:</b> Change label (Score, Scorer) of selected rows without modifying times.</li>
                            <li><b>Delete Selected Scores:</b> Remove selected rows permanently.</li>
                        </ul>

                        <h4>Score Columns:</h4>
                        <table border="1" cellpadding="8" style="margin: 10px 0;">
                            <tr style="background-color: #e0e0e0;">
                                <th>Column</th>
                                <th>Description</th>
                                <th>Editable</th>
                            </tr>
                            <tr>
                                <td><b>ID#</b></td>
                                <td>Unique identifier (e.g., 'M1', 'A2'). 'M'=Manual, 'A'=Auto-detected.</td>
                                <td>No</td>
                            </tr>
                            <tr>
                                <td><b>Score</b></td>
                                <td>Label: 'Ripple', 'Fast Ripple', 'None' (ambiguous), or custom. Auto-populated when preset applied.</td>
                                <td>Yes</td>
                            </tr>
                            <tr>
                                <td><b>Start Time (ms)</b></td>
                                <td>Event start in milliseconds.</td>
                                <td>No</td>
                            </tr>
                            <tr>
                                <td><b>Stop Time (ms)</b></td>
                                <td>Event stop in milliseconds.</td>
                                <td>No</td>
                            </tr>
                            <tr>
                                <td><b>Duration (ms)</b></td>
                                <td>Computed as (Stop - Start). Used for ripple vs fast ripple classification.</td>
                                <td>No</td>
                            </tr>
                            <tr>
                                <td><b>Behavioral State</b></td>
                                <td>'rest' (low speed), 'active' (high speed), or 'unknown' (no speed data). Auto-computed from .pos file if available.</td>
                                <td>No</td>
                            </tr>
                            <tr>
                                <td><b>Scorer</b></td>
                                <td>'Auto' (automated detection) or user name (manual scoring).</td>
                                <td>Yes</td>
                            </tr>
                            <tr>
                                <td><b>Settings File</b></td>
                                <td>Path to detection parameters file (JSON) used to detect this event.</td>
                                <td>No</td>
                            </tr>
                        </table>

                        <h4>Export Options:</h4>
                        <ul>
                            <li><b>Export HFO Metrics (CSV):</b> 
                                <ul style="margin-left: 20px; margin-top: 5px;">
                                    <li>Exports quantitative metrics for <b>Ripple-family events only</b> (Ripple + Fast Ripple)</li>
                                    <li>Includes: ripple rates (per minute), pathology scores, behavioral breakdown (% rest vs active)</li>
                                    <li>Output: Single CSV with one row per region/subject</li>
                                    <li>Use this for papers, statistics, and clinical analysis</li>
                                </ul>
                            </li>
                            <li><b>Export for DL Training:</b>
                                <ul style="margin-left: 20px; margin-top: 5px;">
                                    <li>Exports labeled <b>segments</b> (30 ms windows, overlapping) with labels: Ripple-family=1, Artifact=0</li>
                                    <li>Output: (1) Segment files in HDF5 format, (2) manifest.csv linking files to labels, (3) metrics summary</li>
                                    <li>Use this to prepare datasets for Deep Learning model training</li>
                                </ul>
                            </li>
                        </ul>

                        <h3 id="csvconvert">📊 CSV Convert Tab</h3>
                        <p><b>Create train/validation splits from manifests exported by "Export for DL Training".</b></p>
                        
                        <h4>Controls:</h4>
                        <ul>
                            <li><b>Validation Fraction:</b> Percentage of data reserved for validation (e.g., 0.20 = 20%). Rest goes to training.</li>
                            <li><b>Random Seed:</b> Ensures reproducible splits. Same seed always produces the same train/val split.</li>
                            <li><b>Stratified split:</b> When enabled, balances class distribution (Ripple vs Artifact counts) and avoids subject leakage (each subject appears in only one split).</li>
                            <li><b>Output Directory:</b> Folder where train.csv and val.csv are saved.</li>
                            <li><b>Status Panel:</b> Shows sample counts, class breakdown, and subject statistics for each split.</li>
                        </ul>

                        <h3 id="train">🎓 Train Tab</h3>
                        <p><b>Train a Convolutional Neural Network (CNN) on labeled segments to automatically detect HFOs.</b></p>
                        
                        <h4>Input Files:</h4>
                        <ul>
                            <li><b>Train manifest (CSV):</b> Path to train.csv from CSV Convert tab. Lists segment paths and labels.</li>
                            <li><b>Val manifest (CSV):</b> Path to val.csv from CSV Convert tab. Used to monitor validation performance.</li>
                        </ul>

                        <h4>Hyperparameters:</h4>
                        <ul>
                            <li><b>Epochs:</b> Full passes over the training set. Typical: 15–50. More epochs = slower but potentially better accuracy (watch for overfitting).</li>
                            <li><b>Batch size:</b> Segments processed per gradient update. Typical: 32–128.
                                <ul style="margin-left: 20px; margin-top: 5px;">
                                    <li>Larger batches: faster training, smoother loss curves</li>
                                    <li>Smaller batches: noisier updates, may escape local minima</li>
                                </ul>
                            </li>
                            <li><b>Learning rate:</b> Optimizer step size. Typical: 1e-3 to 5e-3.
                                <ul style="margin-left: 20px; margin-top: 5px;">
                                    <li>Start low (5e-4) for stability, increase if training is too slow</li>
                                    <li>If loss oscillates wildly, reduce learning rate</li>
                                </ul>
                            </li>
                            <li><b>Weight decay:</b> L2 regularization (penalty for large weights). Typical: 1e-4. Increase to 1e-3 or 1e-2 to reduce overfitting.</li>
                        </ul>

                        <h4>Training Controls:</h4>
                        <ul>
                            <li><b>Output dir:</b> Folder to save checkpoints. Best validation model is saved as best.pt.</li>
                            <li><b>Show training visualization:</b> Opens a real-time GUI showing training/validation loss curves and metrics.</li>
                            <li><b>Start Training:</b> Begins training. Logs appear in the text area below.</li>
                            <li><b>Export Model:</b> Select best.pt and export as TorchScript (.pt, required) and optionally ONNX (.onnx). TorchScript is used by the Deep Learning detector.</li>
                        </ul>

                        <h3 id="workflow-metrics">⚙️ Recommended Workflow: HFO Metrics (Analysis)</h3>
                        <p>Use this workflow to analyze HFO properties (rates, pathology, behavior) without training a Deep Learning model.</p>
                        <ol style="margin-left: 20px; background-color: #f0f0f0; padding: 10px; border-radius: 5px; line-height: 1.8;">
                            <li><b>Automatic Detection:</b> Run Consensus detection (Hilbert + STE + MNI with Majority voting recommended)</li>
                            <li><b>Review:</b> In Automatic Detection tab, review detected EOIs visually on the graph</li>
                            <li><b>Move to Score:</b> Select EOIs → click <b>"Add Selected EOI(s) to Score"</b> to transfer to Score tab</li>
                            <li><b>Manual Refinement (optional):</b> In Score tab, manually update labels or delete false positives</li>
                            <li><b>Select Region:</b> Choose a brain region preset (LEC, Hippocampus, MEC) in Score tab. This filters by duration and applies behavioral gating.</li>
                            <li><b>Export Metrics:</b> Click <b>"Export HFO Metrics (CSV)"</b> to get quantitative metrics (ripple rates, pathology scores, rest/active breakdown)</li>
                            <li><b>Analyze:</b> Use the exported CSV in your analysis pipeline (R, Python, GraphPad, etc.)</li>
                        </ol>

                        <h3 id="workflow-dl">⚙️ Recommended Workflow: Deep Learning Training & Deployment</h3>
                        <p>Use this workflow to train a custom detection model on your data.</p>
                        <ol style="margin-left: 20px; background-color: #f0f0f0; padding: 10px; border-radius: 5px; line-height: 1.8;">
                            <li><b>Automatic Detection:</b> Run detection and collect candidate EOIs</li>
                            <li><b>Move to Score:</b> Click <b>"Add Selected EOI(s) to Score"</b></li>
                            <li><b>Manual Labeling:</b> Manually label EOIs as Ripple/Fast Ripple/Artifact in Score tab. Add false positives manually if needed.</li>
                            <li><b>Select Region Preset:</b> Choose a brain region (or None if multi-region) in Score tab</li>
                            <li><b>Export for DL:</b> Click <b>"Export for DL Training"</b>. This creates segment files + manifest + metrics summary.</li>
                            <li><b>Create Splits:</b> Go to CSV Convert tab. Load the manifest, set validation fraction (0.20 typical), and click "Generate". Creates train.csv and val.csv.</li>
                            <li><b>Train Model:</b> Go to Train tab. Load train.csv and val.csv, set hyperparameters, and click "Start Training".</li>
                            <li><b>Monitor:</b> Watch the training visualization to check for overfitting (if val loss > train loss, consider more regularization).</li>
                            <li><b>Export Model:</b> After training, click "Export Model" and select best.pt to save as TorchScript (.pt).</li>
                            <li><b>Deploy:</b> In Settings, set the model path to your .pt file. Future detections will use this model.</li>
                        </ol>

                        <h3 id="parameters">⚙️ Detection Parameters & Literature Defaults</h3>
                        <p>Default parameters align with epilepsy literature standards for robust HFO detection:</p>
                        <table border="1" cellpadding="8" style="margin: 10px 0;">
                            <tr style="background-color: #e0e0e0;">
                                <th>Method</th>
                                <th>Parameter</th>
                                <th>Default</th>
                                <th>Justification</th>
                            </tr>
                            <tr>
                                <td rowspan="4"><b>Hilbert</b></td>
                                <td>Threshold (SD)</td>
                                <td>3.5</td>
                                <td>Detects 3.5σ above baseline envelope; balances sensitivity & specificity</td>
                            </tr>
                            <tr>
                                <td>Cycles (Ripple)</td>
                                <td>4</td>
                                <td>≥4 cycles @ 80–250 Hz confirms oscillatory content</td>
                            </tr>
                            <tr>
                                <td>Min Duration</td>
                                <td>10 ms</td>
                                <td>Shortest physiological HFO (~1 cycle @ 100 Hz)</td>
                            </tr>
                            <tr>
                                <td>Merge Window</td>
                                <td>25 ms</td>
                                <td>Post-detection merging to avoid over-fragmentation</td>
                            </tr>
                            <tr>
                                <td><b>STE</b></td>
                                <td>Threshold (RMS)</td>
                                <td>2.5</td>
                                <td>2.5× baseline RMS energy; robust to amplitude variations</td>
                            </tr>
                            <tr>
                                <td><b>MNI</b></td>
                                <td>Threshold (Percentile)</td>
                                <td>98th</td>
                                <td>Top 2% of energy distribution; reduces false detections</td>
                            </tr>
                            <tr>
                                <td rowspan="2"><b>Region Presets</b></td>
                                <td>Ripple Duration Range</td>
                                <td>10–150 ms</td>
                                <td>Ripple-specific temporal range; filters out faster oscillations</td>
                            </tr>
                            <tr>
                                <td>Fast Ripple Duration Range</td>
                                <td>10–50 ms</td>
                                <td>Shorter, faster oscillations typical of pathological activity</td>
                            </tr>
                        </table>
                        """
                )
        help_layout.addWidget(self.help_text)
        help_tab.setLayout(help_layout)

        tabs.addTab(eoi_tab, 'Automatic Detection')
        tabs.addTab(score_tab, 'Score')
        tabs.addTab(convert_tab, 'CSV convert')
        tabs.addTab(train_tab, 'Train')
        tabs.addTab(help_tab, 'How to Score')

        window_layout = QtWidgets.QVBoxLayout()
        for item in [source_layout, tabs]:
            if 'Layout' in item.__str__():
                window_layout.addLayout(item)
            else:
                window_layout.addWidget(item)

        self.hilbert_thread = QtCore.QThread()
        self.pyhfo_thread = QtCore.QThread()
        self.ste_thread = QtCore.QThread()
        self.mni_thread = QtCore.QThread()
        self.dl_thread = QtCore.QThread()
        self.consensus_thread = QtCore.QThread()
        self.setLayout(window_layout)

    def closeEvent(self, event):
        """Override close event to properly stop worker threads"""
        if hasattr(self, 'hilbert_thread') and self.hilbert_thread.isRunning():
            self.hilbert_thread.quit()
            self.hilbert_thread.wait(1000)  # Wait up to 1 second for thread to finish
        if hasattr(self, 'pyhfo_thread') and self.pyhfo_thread.isRunning():
            self.pyhfo_thread.quit()
            self.pyhfo_thread.wait(1000)
        if hasattr(self, 'ste_thread') and self.ste_thread.isRunning():
            self.ste_thread.quit()
            self.ste_thread.wait(1000)
        if hasattr(self, 'mni_thread') and self.mni_thread.isRunning():
            self.mni_thread.quit()
            self.mni_thread.wait(1000)
        if hasattr(self, 'dl_thread') and self.dl_thread.isRunning():
            self.dl_thread.quit()
            self.dl_thread.wait(1000)
        if hasattr(self, 'consensus_thread') and self.consensus_thread.isRunning():
            self.consensus_thread.quit()
            self.consensus_thread.wait(1000)
        super().closeEvent(event)

    def initialize_attributes(self):
        self.IDs = []
        self.train_process = None
        self.export_process = None

    def _on_detection_progress(self, message):
        """Handle detection progress updates; print to console."""
        print(f"[Detection] {message}")

    def _build_region_presets(self):
        """Define region-specific defaults for banding, durations, and export gating."""
        return {
            'LEC': {
                'bands': {
                    'ripple': [80, 250],
                    'fast_ripple': [250, 500],
                    'gamma': [30, 80],
                },
                'durations': {
                    'ripple_min_ms': 15,
                    'ripple_max_ms': 120,
                    'fast_min_ms': 10,
                    'fast_max_ms': 80,
                },
                'threshold_sd': 3.5,
                'epoch_s': 300,
                'behavior_gating': True,
                'speed_threshold_min_cm_s': 0.0,
                'speed_threshold_max_cm_s': 5.0,
                'dl_export': {
                    'filter_by_duration': True,
                    'annotate_band': True,
                    'behavior_gating': True,
                },
            },
            'Hippocampus': {
                'bands': {
                    'ripple': [100, 250],
                    'fast_ripple': [250, 500],
                    'gamma': [30, 80],
                },
                'durations': {
                    'ripple_min_ms': 15,
                    'ripple_max_ms': 120,
                    'fast_min_ms': 10,
                    'fast_max_ms': 80,
                },
                'threshold_sd': 4.0,
                'epoch_s': 300,
                'behavior_gating': True,
                'speed_threshold_min_cm_s': 0.0,
                'speed_threshold_max_cm_s': 5.0,
                'dl_export': {
                    'filter_by_duration': True,
                    'annotate_band': True,
                    'behavior_gating': True,
                },
            },
            'MEC': {
                'bands': {
                    'ripple': [80, 200],
                    'fast_ripple': [200, 500],
                    'gamma': [30, 80],
                },
                'durations': {
                    'ripple_min_ms': 15,
                    'ripple_max_ms': 120,
                    'fast_min_ms': 10,
                    'fast_max_ms': 80,
                },
                'threshold_sd': 3.5,
                'epoch_s': 300,
                'behavior_gating': True,
                'speed_threshold_min_cm_s': 0.0,
                'speed_threshold_max_cm_s': 5.0,
                'dl_export': {
                    'filter_by_duration': True,
                    'annotate_band': True,
                    'behavior_gating': True,
                },
            },
        }

    def _on_region_changed(self, region_name):
        self.current_region = region_name
        # Clear region profile if "None" is selected
        if region_name == "None":
            self.region_profile = {}
        else:
            self.region_profile = self.region_presets.get(region_name, {}).copy()

    def openRegionPresetDialog(self):
        """Open dialog to review and modify region preset before applying."""
        if self.current_region == "None":
            QtWidgets.QMessageBox.information(self, "No Preset", "No region preset is active. Select a brain region (LEC, Hippocampus, MEC) to configure presets.")
            return
        profile = self.region_presets.get(self.current_region, {}).copy()
        if not profile:
            QtWidgets.QMessageBox.warning(self, "Preset Missing", f"No preset found for region: {self.current_region}")
            return
        
        dialog = RegionPresetDialog(self, self.current_region, profile)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            modified_profile = dialog.get_modified_preset()
            self.applyRegionPreset(modified_profile)
    
    def applyRegionPreset(self, profile):
        """Apply the given profile to all detector parameter sets."""
        self.region_profile = profile
        self._update_hilbert_params_from_profile(profile)
        self._update_ste_params_from_profile(profile)
        self._update_mni_params_from_profile(profile)
        self._update_consensus_params_from_profile(profile)
        self._update_score_labels_from_profile(profile)  # Re-label existing scores
        QtWidgets.QMessageBox.information(
            self,
            "Region Preset Applied",
            f"Applied preset for {self.current_region}.\n"
            "Detector defaults updated; export will include duration and speed annotations.")

    def _update_hilbert_params_from_profile(self, profile):
        """Persist hilbert parameter defaults according to the region preset."""
        try:
            settings_file = os.path.join(self.mainWindow.SETTINGS_DIR, 'hilbert_params.json')
            bands = profile.get('bands', {})
            durations = profile.get('durations', {})
            params = {
                'Epoch(s):': str(profile.get('epoch_s', 300)),
                'Threshold(SD):': str(profile.get('threshold_sd', 3.5)),
                'Minimum Time(ms):': str(durations.get('ripple_min_ms', 10)),
                'Min Frequency(Hz):': str(bands.get('ripple', [80, 250])[0]),
                'Max Frequency(Hz):': str(bands.get('ripple', [80, 250])[1]),
                'Required Peaks:': '6',
                'Required Peak Threshold(SD):': '2',
                'Boundary Threshold(Percent)': '30',
            }
            with open(settings_file, 'w') as f:
                json.dump(params, f)
        except Exception:
            pass

    def _update_ste_params_from_profile(self, profile):
        try:
            settings_file = os.path.join(self.mainWindow.SETTINGS_DIR, 'ste_params.json')
            bands = profile.get('bands', {})
            params = {
                'threshold': 2.5,
                'window_size': 0.01,
                'overlap': 0.5,
                'min_freq': bands.get('ripple', [80, 250])[0],
                'max_freq': bands.get('fast_ripple', [250, 500])[1],
            }
            with open(settings_file, 'w') as f:
                json.dump(params, f)
        except Exception:
            pass

    def _update_mni_params_from_profile(self, profile):
        try:
            settings_file = os.path.join(self.mainWindow.SETTINGS_DIR, 'mni_params.json')
            bands = profile.get('bands', {})
            params = {
                'baseline_window': 10.0,
                'threshold_percentile': 98.0,
                'min_freq': bands.get('ripple', [80, 250])[0],
                'max_freq': bands.get('fast_ripple', [250, 500])[1],
            }
            with open(settings_file, 'w') as f:
                json.dump(params, f)
        except Exception:
            pass

    def _update_consensus_params_from_profile(self, profile):
        try:
            settings_file = os.path.join(self.mainWindow.SETTINGS_DIR, 'consensus_params.json')
            bands = profile.get('bands', {})
            params = {
                'voting_strategy': 'majority',
                'overlap_ms': 10.0,
                'hilbert_epoch': profile.get('epoch_s', 300),
                'hilbert_sd_num': profile.get('threshold_sd', 3.5),
                'hilbert_min_duration': profile.get('durations', {}).get('ripple_min_ms', 10),
                'hilbert_min_freq': bands.get('ripple', [80, 250])[0],
                'hilbert_max_freq': bands.get('fast_ripple', [250, 500])[1],
                'hilbert_required_peaks': 6,
                'hilbert_peak_sd': 2.0,
                'ste_threshold': 2.5,
                'ste_window_size': 0.01,
                'ste_overlap': 0.5,
                'ste_min_freq': bands.get('ripple', [80, 250])[0],
                'ste_max_freq': bands.get('fast_ripple', [250, 500])[1],
                'mni_baseline_window': 10.0,
                'mni_threshold_percentile': 98.0,
                'mni_min_freq': bands.get('ripple', [80, 250])[0],
                'mni_max_freq': bands.get('fast_ripple', [250, 500])[1],
            }
            with open(settings_file, 'w') as f:
                json.dump(params, f)
        except Exception:
            pass

    def _update_score_labels_from_profile(self, profile):
        """Re-evaluate Score labels and behavioral state for all existing Score items based on new duration and speed thresholds."""
        try:
            durations = profile.get('durations', {})
            r_min = durations.get('ripple_min_ms', 10)
            r_max = durations.get('ripple_max_ms', 150)
            fr_min = durations.get('fast_ripple_min_ms', 10)
            fr_max = durations.get('fast_ripple_max_ms', 50)
            
            # Get speed thresholds for behavioral gating
            speed_min = profile.get('speed_threshold_min_cm_s', 0.0)
            speed_max = profile.get('speed_threshold_max_cm_s', 5.0)
            
            score_col = self.score_headers.get('Score:', 1)
            duration_col = self.score_headers.get('Duration(ms):', 4)
            behavior_col = self.score_headers.get('Behavioral State:', 5)
            start_col = self.score_headers.get('Start Time(ms):', 2)
            stop_col = self.score_headers.get('Stop Time(ms):', 3)
            
            # Get speed signal for behavioral state re-computation
            speed_signal = self._get_speed_signal()
            
            # Iterate through all Score items and re-label based on duration and behavior
            for i in range(self.scores.topLevelItemCount()):
                item = self.scores.topLevelItem(i)
                if item:
                    try:
                        # Update Score label based on duration
                        duration_text = item.text(duration_col)
                        if duration_text:
                            duration = float(duration_text)
                            # Classify based on duration thresholds
                            if r_min <= duration <= r_max:
                                new_label = 'Ripple'
                            elif fr_min <= duration <= fr_max:
                                new_label = 'Fast Ripple'
                            else:
                                new_label = 'None'  # ambiguous
                            
                            item.setText(score_col, new_label)
                        
                        # Re-compute behavioral state only if currently "unknown" or empty
                        current_behavior = item.text(behavior_col) if behavior_col is not None else 'unknown'
                        if (current_behavior.lower() == 'unknown' or current_behavior.strip() == '') and speed_signal is not None:
                            try:
                                speed_trace, fs_speed = speed_signal
                                s_ms = float(item.text(start_col))
                                e_ms = float(item.text(stop_col))
                                
                                s_idx = int(max(0, np.floor(s_ms / 1000 * fs_speed)))
                                e_idx = int(min(len(speed_trace), np.ceil(e_ms / 1000 * fs_speed)))
                                
                                if e_idx > s_idx:
                                    seg_speed = speed_trace[s_idx:e_idx]
                                    if seg_speed.size > 0:
                                        mean_speed = float(np.nanmean(seg_speed))
                                        new_behavior = 'rest' if (speed_min <= mean_speed <= speed_max) else 'active'
                                        item.setText(behavior_col, new_behavior)
                            except Exception:
                                pass
                    except Exception:
                        pass
        except Exception:
            pass

    def _get_speed_signal(self):
        """Return (speed_trace, fs_speed) if a .pos-derived speed source is loaded."""
        try:
            for key, value in self.settingsWindow.loaded_sources.items():
                if key.endswith('.pos') or 'Speed' in key:
                    if isinstance(value, (list, tuple)) and len(value) >= 2:
                        speed = np.asarray(value[0]).flatten()
                        fs_speed = value[1]
                        return speed, fs_speed
        except Exception:
            return None
        return None

    def _filter_and_annotate_eois_for_export(self, eois, profile):
        """Annotate EOIs with band label (PSD-dominant; duration fallback) and behavioral state.

        Labeling priority:
        1) PSD-based: compare ripple vs fast ripple band power; choose dominant.
        2) If band powers within 10% (ambiguous), fall back to duration thresholds.
        3) If duration also ambiguous, label as 'ripple_fast_ripple'.
        """
        # Get brain region name if available
        brain_region_name = getattr(self, 'current_region', None)
        
        durations = profile.get('durations', {})
        r_min = durations.get('ripple_min_ms', 0)
        r_max = durations.get('ripple_max_ms', np.inf)
        fr_min = durations.get('fast_min_ms', 0)
        fr_max = durations.get('fast_max_ms', np.inf)
        bands = profile.get('bands', {})
        ripple_band = bands.get('ripple', [80, 250])
        fast_band = bands.get('fast_ripple', [250, 500])
        speed_min = profile.get('speed_threshold_min_cm_s', 0.0)
        speed_max = profile.get('speed_threshold_max_cm_s', 5.0)
        do_behavior = profile.get('behavior_gating', False)
        export_opts = profile.get('dl_export', {})
        filter_by_duration = export_opts.get('filter_by_duration', True)

        speed_signal = self._get_speed_signal() if do_behavior else None

        # Access raw data and sampling rate for PSD
        try:
            raw_data, Fs = self.settingsWindow.loaded_sources[self.source_filename]
            raw_data = np.asarray(raw_data).flatten()
        except Exception:
            raw_data, Fs = None, None

        filtered = []
        metadata = []
        for s_ms, e_ms in eois:
            try:
                s_ms = float(s_ms)
                e_ms = float(e_ms)
            except Exception:
                continue
            duration = e_ms - s_ms
            if duration <= 0:
                continue

            band_label = 'ripple_fast_ripple'  # default ambiguous

            # PSD-based classification when raw_data is available
            psd_decided = False
            if raw_data is not None and Fs is not None:
                s_idx_sig = int(max(0, np.floor(s_ms / 1000 * Fs)))
                e_idx_sig = int(min(len(raw_data), np.ceil(e_ms / 1000 * Fs)))
                if e_idx_sig > s_idx_sig:
                    seg = raw_data[s_idx_sig:e_idx_sig]
                    try:
                        nper = int(min(1024, max(64, len(seg))))
                        f, Pxx = welch(seg, fs=Fs, nperseg=nper, noverlap=nper//2, scaling='density')
                        rp_lo, rp_hi = float(ripple_band[0]), float(ripple_band[1])
                        fr_lo, fr_hi = float(fast_band[0]), float(fast_band[1])
                        rp_mask = (f >= rp_lo) & (f < rp_hi)
                        fr_mask = (f >= fr_lo) & (f < fr_hi)
                        ripple_power = float(np.nansum(Pxx[rp_mask])) if np.any(rp_mask) else 0.0
                        fast_power = float(np.nansum(Pxx[fr_mask])) if np.any(fr_mask) else 0.0
                        max_power = max(ripple_power, fast_power)
                        if max_power > 0:
                            rel_diff = abs(ripple_power - fast_power) / max_power
                            if rel_diff > 0.10:
                                band_label = 'ripple' if ripple_power > fast_power else 'fast_ripple'
                                psd_decided = True
                    except Exception:
                        psd_decided = False

            if not psd_decided:
                # Duration fallback
                if r_min <= duration <= r_max:
                    band_label = 'ripple'
                elif fr_min <= duration <= fr_max:
                    band_label = 'fast_ripple'
                else:
                    band_label = 'ripple_fast_ripple'

            # If strict duration filtering requested and duration is outside all ranges, drop
            if filter_by_duration and not (r_min <= duration <= r_max or fr_min <= duration <= fr_max):
                # Only drop if user enforces duration filter; keep ambiguous labels otherwise
                continue

            state = 'unknown'
            mean_speed = None
            if speed_signal is not None:
                speed_trace, fs_speed = speed_signal
                s_idx = int(max(0, np.floor(s_ms / 1000 * fs_speed)))
                e_idx = int(min(len(speed_trace), np.ceil(e_ms / 1000 * fs_speed)))
                if e_idx > s_idx:
                    seg_speed = speed_trace[s_idx:e_idx]
                    if seg_speed.size > 0:
                        mean_speed = float(np.nanmean(seg_speed))
                        # Speed within range [min, max] = 'rest', outside = 'active'
                        state = 'rest' if (speed_min <= mean_speed <= speed_max) else 'active'

            filtered.append([s_ms, e_ms])
            meta_dict = {
                'band_label': band_label,
                'duration_ms': duration,
                'state': state,
                'mean_speed_cm_s': mean_speed,
            }
            if brain_region_name:
                meta_dict['brain_region'] = brain_region_name
            metadata.append(meta_dict)

        return filtered, metadata

    def openSettings(self, index, source):

        if 'score' in source:
            for key, val in self.score_headers.items():
                if 'Settings' in key:
                    break
        else:
            for key, val in self.EOI_headers.items():
                if 'Settings' in key:
                    break

        settings_filename = index[val].data()
        if settings_filename != '' or settings_filename != 'N/A':

            if os.path.exists(settings_filename):
                self.setting_viewer_window = SettingsViewer(settings_filename)

    def copySettings(self, index, source):
        if 'score' in source:
            for key, val in self.score_headers.items():
                if 'Settings' in key:
                    break
        else:
            for key, val in self.EOI_headers.items():
                if 'Settings' in key:
                    break

        settings_filename = index[val].data()
        if settings_filename != '' or settings_filename != 'N/A':
            # pyperclip.copy(settings_filename)
            cb = QtWidgets.QApplication.clipboard()
            cb.clear(mode=cb.Clipboard)
            cb.setText(settings_filename, mode=cb.Clipboard)

    def openMenu(self, source, position):

        menu = QtWidgets.QMenu()

        if 'score' in source:
            indexes = self.scores.selectedIndexes()

            menu.addAction("Open Settings File", functools.partial(self.openSettings, indexes, source))
            menu.addAction("Copy Settings Filepath", functools.partial(self.copySettings, indexes, source))

            menu.exec_(self.scores.viewport().mapToGlobal(position))
        else:
            indexes = self.EOI.selectedIndexes()

            menu.addAction("Open Settings File", functools.partial(self.openSettings, indexes, source))
            menu.addAction("Copy Settings Filepath", functools.partial(self.copySettings, indexes, source))

            menu.exec_(self.EOI.viewport().mapToGlobal(position))

    def addScore(self):
        if self.mainWindow.score_x1 is None or self.mainWindow.score_x2 is None:
            return

        # Auto-fill Scorer field if empty
        scorer_name = self.scorer.text().strip()
        if scorer_name == '':
            scorer_name = 'Auto'
            self.scorer.setText(scorer_name)

        # new_item = QtWidgets.QTreeWidgetItem()
        new_item = TreeWidgetItem()
        id = self.createID('Manual')
        self.IDs.append(id)

        for key, value in self.score_headers.items():
            if 'ID' in key:
                new_item.setText(value, id)
            elif 'Score:' in key:
                new_item.setText(value, self.score.currentText())
            elif 'Start' in key:
                new_item.setText(value, str(self.mainWindow.score_x1))
            elif 'Stop' in key:
                new_item.setText(value, str(self.mainWindow.score_x2))
            elif 'Duration' in key:
                try:
                    dur_ms = float(self.mainWindow.score_x2) - float(self.mainWindow.score_x1)
                    if dur_ms > 0:
                        new_item.setText(value, f"{dur_ms:.3f}")
                    else:
                        new_item.setText(value, "")
                except Exception:
                    new_item.setText(value, "")
            elif 'Behavioral State' in key:
                new_item.setText(value, 'unknown')
            elif 'Settings File' in key:
                # there is no settings file involved in manual detection
                new_item.setText(value, 'N/A')
            elif 'Scorer' in key:
                new_item.setText(value, scorer_name)

        self.scores.addTopLevelItem(new_item)

    def add_item(self, item):
        """This is a method was created so that we could add QTreeWidgetItems
            from the main thread since it did not like that we were adding EOIs
            from a thread"""
        self.EOI.addTopLevelItem(item)

    def createID(self, source):
        """This method will create an ID for the newly added Score"""
        source = source.lower()

        existing_IDs = self.existingID(source)  # get the IDs with that existing source

        # Parse ID numbers, skipping any malformed IDs
        ID_numbers = []
        for ID in existing_IDs:
            try:
                num_part = ID[3:]
                if num_part:  # Skip empty strings
                    ID_numbers.append(int(num_part))
            except (ValueError, IndexError):
                # Skip malformed IDs
                continue
        
        ID_numbers = np.asarray(ID_numbers).flatten()

        if len(ID_numbers) != 0:
            return '%s%d' % (self.id_abbreviations[source], np.setdiff1d(np.arange(1, len(ID_numbers)+2), ID_numbers)[0])
        else:
            # there are no ID's,
            return '%s%d' % (self.id_abbreviations[source], 1)

    def existingID(self, source):

        source = source.lower()

        abbreviation = self.id_abbreviations[source]

        '''
        # this method involves iterating which will be more time consuming, instead I'll keep a list of the values
        # root = self.scores.invisibleRootItem()
        iterator = QtWidgets.QTreeWidgetItemIterator(self.scores)



        ID = []
        while iterator.value():
            item = iterator.value()
            current_id = item.data(0, 0)
            if abbreviation in current_id:
                ID.append(current_id)

            iterator += 1
        '''

        ID = [value for value in self.IDs if abbreviation in value]

        return ID

    def deleteScores(self):
        '''deletes the selected scores in the Scores Window's TreeWidget'''
        
        # Get selected items as a list (copy to avoid modification during iteration)
        selected_items = list(self.scores.selectedItems())
        if not selected_items:
            return

        for key, value in self.score_headers.items():
            if 'ID' in key:
                id_value = value
                break

        # Collect indices and IDs first
        indices_to_remove = []
        ids_to_remove = []
        for item in selected_items:
            index = self.scores.indexOfTopLevelItem(item)
            if index >= 0:
                indices_to_remove.append(index)
                ID = item.data(id_value, 0)
                if ID in self.IDs:
                    ids_to_remove.append(ID)

        # Remove IDs from tracking list
        for ID in ids_to_remove:
            self.IDs.pop(self.IDs.index(ID))

        # Remove items in reverse order by index
        for index in sorted(indices_to_remove, reverse=True):
            self.scores.takeTopLevelItem(index)

    def updateScores(self):
        '''updates the selected scores in the Score Window\'s TreeWidget'''
        root = self.scores.invisibleRootItem()

        for key, value in self.score_headers.items():
            if 'Score:' in key:
                score_value = value
            elif 'Start Time' in key:
                start_value = value
            elif 'Stop Time' in key:
                stop_value = value

        # Only update the score label, not the times
        for item in self.scores.selectedItems():
            item.setText(score_value, self.score.currentText())

    def updateEOIRegion(self):
        root = self.EOI.invisibleRootItem()

        for key, value in self.EOI_headers.items():
            if 'Start Time' in key:
                start_value = value
            elif 'Stop Time' in key:
                stop_value = value

        if hasattr(self.mainWindow, 'lr'):
            # getRegion() returns values in seconds, convert to milliseconds
            x1, x2 = self.mainWindow.lr.getRegion()
            x1_ms = x1 * 1000.0
            x2_ms = x2 * 1000.0
            for item in self.EOI.selectedItems():
                item.setText(start_value, f"{x1_ms:.10g}")
                item.setText(stop_value, f"{x2_ms:.10g}")
        else:
           pass

    def _calculateAndSetDuration(self, item, start_col, stop_col, duration_col):
        """Helper function to auto-calculate and set duration if missing"""
        try:
            start_time = float(item.text(start_col))
            stop_time = float(item.text(stop_col))
            duration = stop_time - start_time
            if duration > 0:
                item.setText(duration_col, f"{duration:.3f}")
        except (ValueError, IndexError):
            # If unable to calculate, leave duration empty
            pass

    def loadScores(self):

        # choose the filename
        save_filename = self.score_filename.text()

        if 'Please add a source!' in save_filename:
            return

        save_filename, save_fileextension = QtWidgets.QFileDialog.getOpenFileName(self, 'Load Scores',
                                                          save_filename,
                                                          'Text Files (*.txt)')
        if save_filename == '':
            return

        self.score_filename.setText(save_filename)

        if os.path.exists(save_filename):
            df = pd.read_csv(save_filename, delimiter='\t')
        else:
            self.mainWindow.choice = ''
            self.mainWindow.ErrorDialogue.myGUI_signal.emit("ScoreFileExistError:%s" % save_filename)

            while self.mainWindow.choice == '':
                time.sleep(0.1)

            return

        # add the scores

        N = len(df)

        # find the IDs of any existing EOIs
        iterator = QtWidgets.QTreeWidgetItemIterator(self.scores)

        score_IDs = []
        while iterator.value():
            item = iterator.value()
            score_IDs.append(item.data(0, 0))
            iterator += 1

        self.scores.clear()
        [self.IDs.pop(self.IDs.index(ID)) for ID in score_IDs]  # remove the id from the list of IDs

        ids_exist = any('ID' in column for column in df.columns)
        scorer_exists = any('Scorer' in column for column in df.columns)
        score_settings_file_exists = any('Settings File' in column for column in df.columns)
        duration_exists = any('Duration' in column for column in df.columns)
        brain_region_exists = any('Brain Region' in column for column in df.columns)

        id_value = None
        scorer_value = None
        brain_region_value = None
        settings_value = None
        start_value = None
        stop_value = None
        duration_value = None
        score_value = None

        for key, value in self.score_headers.items():
            if 'ID' in key:
                id_value = value

            elif 'Scorer' in key:
                scorer_value = value

            elif 'Brain Region' in key:
                brain_region_value = value

            elif 'Settings File' in key:
                settings_value = value

            elif 'Start' in key:
                start_value = value

            elif 'Stop' in key:
                stop_value = value

            elif 'Duration' in key:
                duration_value = value

            elif 'Score:' in key:
                score_value = value

        for score_index in range(N):
            # item = QtWidgets.QTreeWidgetItem()
            item = TreeWidgetItem()

            if not ids_exist and id_value is not None:
                ID = self.createID('Unknown')
                self.IDs.append(ID)
                item.setText(id_value, ID)

            if not scorer_exists and scorer_value is not None:
                item.setText(scorer_value, 'Auto')

            if not brain_region_exists and brain_region_value is not None:
                item.setText(brain_region_value, 'Unknown')

            if not score_settings_file_exists and settings_value is not None:
                item.setText(settings_value, 'N/A')

            for column in df.columns:

                if 'Unnamed' in column:
                    continue

                for key, value in self.score_headers.items():
                    if key == column:
                        if 'ID' in key:
                            ID = df[column][score_index]
                            self.IDs.append(ID)
                            item.setText(value, ID)

                        else:
                            item.setText(value, str(df[column][score_index]))

                    # these next statements are for the older files
                    elif 'Score:' in column and 'Score:' in key:
                        item.setText(value, str(df[column][score_index]))

                    elif 'Scorer:' in column and 'Scorer:' in key:
                        item.setText(value, str(df[column][score_index]))

                    elif 'Start' in column and 'Start' in key:
                        item.setText(value, str(df[column][score_index]))

                    elif 'Stop' in column and 'Stop' in key:
                        item.setText(value, str(df[column][score_index]))

            # Auto-calculate duration if not present in file
            if not duration_exists:
                self._calculateAndSetDuration(item, start_value, stop_value, duration_value)

            self.scores.addTopLevelItem(item)

            self.scores.sortItems(start_value, QtCore.Qt.AscendingOrder)

    def saveScores(self):
        """This method will save the scores into a text file"""
        # iterate through each item

        # choose the filename
        save_filename = self.score_filename.text()

        if 'Please add a source!' in save_filename:
            return

        save_filename, save_file_extension = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Scores',
                                                          save_filename,
                                                          'Text Files (*.txt)')

        if save_filename == '':
            return

        self.score_filename.setText(save_filename)

        scores = []
        start_times = []
        stop_times = []
        ids = []
        scorer = []

        for key, value in self.score_headers.items():
            if 'Score:' in key:
                score_value = value
            elif 'Start' in key:
                start_value = value
            elif 'Stop' in key:
                stop_value = value
            elif 'ID' in key:
                id_value = value
            elif 'Scorer:' in key:
                scorer_value = value

        for item_count in range(self.scores.topLevelItemCount()):
            item = self.scores.topLevelItem(item_count)

            ids.append(item.data(id_value, 0))
            scores.append(item.data(score_value, 0))
            start_times.append(item.data(start_value, 0))
            stop_times.append(item.data(stop_value, 0))
            scorer.append(item.data(scorer_value, 0))

        data_dict = {}
        for key, value in self.score_headers.items():
            if 'ID' in key:
                data_dict[key] = ids

            elif 'Score:' in key:
                data_dict[key] = pd.Series(scores)

            elif 'Start' in key:
                data_dict[key] = pd.Series(start_times)

            elif 'Stop' in key:
                data_dict[key] = pd.Series(stop_times)

            elif 'Scorer' in key:
                data_dict[key] = pd.Series(scorer)

        df = pd.DataFrame(data_dict)

        # make the directory name if it does not exists already
        if not os.path.exists(os.path.dirname(save_filename)):
            os.makedirs(os.path.dirname(save_filename))

        df.to_csv(save_filename, sep='\t')

    def updateActiveSources(self):
        """This method updates the source combobox depending on the sources that are within the QTreeWidget within
        the GraphSettingsWindow object"""

        active_sources = self.settingsWindow.getActiveSources()  # get the list of source names within the QTreeWidget

        # get the list of current sources that are listed in the QCombobox
        current_sources = [self.source.itemText(i) for i in range(self.source.count())]

        # add items that are in active_sources but not in current_sources

        add_items = []
        [add_items.append(item) for item in active_sources if item not in current_sources]

        for item in add_items:
            current_sources.append(item)
            self.source.addItem(item)

        # remove items that are in current_sources that are not in active_sources

        remove_items = []
        [remove_items.append(item) for item in current_sources if item not in active_sources]

        for item in remove_items:
            self.source.removeItem(self.source.findText(item))

    def changeSources(self):

        data_source = self.source.currentText()

        if not hasattr(self.mainWindow, 'current_set_filename'):
            return

        session_path, set_filename = os.path.split(self.mainWindow.current_set_filename)
        session = os.path.splitext(set_filename)[0]
        source_filename = os.path.join(session_path, '%s%s' % (session, data_source))

        if not os.path.exists(source_filename):
            self.source_filename = None
            return

        else:
            self.source_filename = source_filename
        pass

    def findEOIs(self):

        # find the IDs of any existing EOIs
        iterator = QtWidgets.QTreeWidgetItemIterator(self.EOI)

        auto_IDs = []
        while iterator.value():
            item = iterator.value()
            auto_IDs.append(item.data(0, 0))
            iterator += 1

        self.EOI.clear()
        [self.IDs.pop(self.IDs.index(ID)) for ID in auto_IDs]  # remove the id from the list of IDs

        if 'Hilbert' in self.eoi_method.currentText():
            # make sure to have the windows have a self. in front of them otherwise they will run and close
            self.hilbert_window = HilbertParametersWindow(self.mainWindow, self)
        elif 'STE' in self.eoi_method.currentText():
            self.ste_window = STEParametersWindow(self.mainWindow, self)
        elif 'MNI' in self.eoi_method.currentText():
            self.mni_window = MNIParametersWindow(self.mainWindow, self)
        elif 'Consensus' in self.eoi_method.currentText():
            self.consensus_window = ConsensusParametersWindow(self.mainWindow, self)
        elif 'Deep Learning' in self.eoi_method.currentText():
            self.dl_window = DLParametersWindow(self.mainWindow, self)

    def changeEventText(self, source):
        """This method will move the plot to the current selection"""
        if 'score' in source:
            # get the selected item

            for key, value in self.score_headers.items():
                if 'Start' in key:
                    start_value = value
                elif 'Stop' in key:
                    stop_value = value
            # root = self.scores.invisibleRootItem()
            try:
                item = self.scores.selectedItems()[0]
            except IndexError:
                return

            stop_time = float(item.data(stop_value, 0))
            start_time = float(item.data(start_value, 0))

        elif 'EOI' in source:

            for key, value in self.EOI_headers.items():
                if 'Start' in key:
                    start_value = value
                elif 'Stop' in key:
                    stop_value = value

            # get the selected item
            # root = self.EOI.invisibleRootItem()
            try:
                item = self.EOI.selectedItems()[0]
            except IndexError:
                # the item was probably deleted
                return

            stop_time = float(item.data(stop_value, 0))
            start_time = float(item.data(start_value, 0))

        self.customSignals.set_lr_signal.emit(str(start_time), str(stop_time))  # plots the lr at this time point

        time_value = np.round((stop_time + start_time) / 2 - self.mainWindow.windowsize / 2)

        # center the screen around the average of the stop_time and start_time
        # self.mainWindow.scrollbar.setValue(time_value / 1000 * self.mainWindow.SourceFs)

        self.mainWindow.current_time_object.setText(str(time_value))
        # self.mainWindow.setCurrentTime()

        # plot the start time

        self.mainWindow.start_time_object.setText(str(start_time))
        self.mainWindow.stop_time_object.setText(str(stop_time))

    def addEOI(self):
        '''This method will add the EOI values to the score list (supports multiple selections).
        
        If a region profile is active, applies brain region presets, behavioral gating, 
        and automatic labeling (same logic as "Export EOIs for DL Training").
        '''

        # Auto-fill Scorer field if empty
        scorer_name = self.scorer.text().strip()
        if scorer_name == '':
            scorer_name = 'Auto'
            self.scorer.setText(scorer_name)
        
        # Default brain region comes from current preset selection
        default_brain_region = getattr(self, 'current_region', None) or 'Unknown'

        # Get all selected items (supports multi-select) - make a copy to avoid modification during iteration
        selected_items = list(self.EOI.selectedItems())
        if not selected_items:
            return

        # Get column indices
        start_col = None
        stop_col = None
        for key, value in self.EOI_headers.items():
            if 'Start' in key:
                start_col = value
            elif 'Stop' in key:
                stop_col = value

        # Extract EOI times from selected items
        eoi_times = []
        for item in selected_items:
            try:
                s_ms = float(item.data(start_col, 0))
                e_ms = float(item.data(stop_col, 0))
                eoi_times.append([s_ms, e_ms])
            except Exception:
                continue

        if not eoi_times:
            return

        # Apply region-aware filtering and annotation if profile is active
        metadata_rows = None
        if hasattr(self, 'region_profile') and self.region_profile:
            try:
                filtered_eois, metadata_rows = self._filter_and_annotate_eois_for_export(
                    np.asarray(eoi_times), self.region_profile
                )
                
                if not filtered_eois:
                    QtWidgets.QMessageBox.warning(
                        self, "No EOIs After Filters", 
                        "All selected EOIs were excluded by duration/behavior filters from the region preset."
                    )
                    return
                
                # Update eoi_times to filtered set
                eoi_times = filtered_eois
                
            except Exception as e:
                QtWidgets.QMessageBox.warning(
                    self, "Filter Error", 
                    f"Error applying region preset filters: {e}\n\nProceeding without filtering."
                )
                metadata_rows = None

        # Collect indices to remove from EOI tree
        indices_to_remove = []
        for item in selected_items:
            index = self.EOI.indexOfTopLevelItem(item)
            if index >= 0:
                indices_to_remove.append(index)

        # Process each EOI (filtered if profile active)
        duplicates_skipped = 0
        for idx, (s_ms, e_ms) in enumerate(eoi_times):
            # Check for duplicates: skip if event with same start/stop already exists in scores
            is_duplicate = False
            for i in range(self.scores.topLevelItemCount()):
                existing_item = self.scores.topLevelItem(i)
                try:
                    for score_key, score_value in self.score_headers.items():
                        if 'Start' in score_key:
                            existing_start = float(existing_item.data(score_value, 0))
                        elif 'Stop' in score_key:
                            existing_stop = float(existing_item.data(score_value, 0))
                    # Allow small tolerance for floating point comparison (0.1 ms)
                    if abs(existing_start - s_ms) < 0.1 and abs(existing_stop - e_ms) < 0.1:
                        is_duplicate = True
                        break
                except Exception:
                    continue
            
            if is_duplicate:
                duplicates_skipped += 1
                continue  # Skip this duplicate event

            new_item = TreeWidgetItem()

            # Get metadata for this EOI if available
            metadata = metadata_rows[idx] if metadata_rows else None
            
            # Extract or compute behavior state
            behavioral_state = 'unknown'
            mean_speed = None
            if metadata and 'state' in metadata:
                behavioral_state = metadata['state']
                mean_speed = metadata.get('mean_speed_cm_s')
            else:
                # Recalculate behavior state from speed signal if available
                try:
                    speed_signal = self._get_speed_signal()
                    if speed_signal is not None:
                        speed_trace, fs_speed = speed_signal
                        speed_min = self.region_profile.get('speed_threshold_min_cm_s', 0.0) if hasattr(self, 'region_profile') else 0.0
                        speed_max = self.region_profile.get('speed_threshold_max_cm_s', 5.0) if hasattr(self, 'region_profile') else 5.0
                        
                        s_idx = int(max(0, np.floor(s_ms / 1000 * fs_speed)))
                        e_idx = int(min(len(speed_trace), np.ceil(e_ms / 1000 * fs_speed)))
                        if e_idx > s_idx:
                            seg_speed = speed_trace[s_idx:e_idx]
                            if seg_speed.size > 0:
                                mean_speed = float(np.nanmean(seg_speed))
                                behavioral_state = 'rest' if (speed_min <= mean_speed <= speed_max) else 'active'
                except Exception:
                    pass
            
            # Store behavior state in item as custom data
            new_item.setData(0, QtCore.Qt.UserRole, behavioral_state)  # Use role to store behavior state
            
            # Determine score label based on band_label from metadata
            score_label = self.EOI_score.currentText()
            if metadata and 'band_label' in metadata:
                band = metadata['band_label']
                if band == 'ripple':
                    score_label = 'Ripple'
                elif band == 'fast_ripple':
                    score_label = 'Fast Ripple'
                elif band == 'ripple_fast_ripple':
                    score_label = 'None'  # ambiguous
            
            # Get brain region from metadata (preset) or use default
            brain_region = metadata.get('brain_region', default_brain_region) if metadata else default_brain_region

            # Fill Score item columns
            for score_key, score_value in self.score_headers.items():
                if 'ID' in score_key:
                    new_id = self.createID('Manual')
                    self.IDs.append(new_id)
                    new_item.setText(score_value, new_id)
                elif 'Score:' in score_key:
                    new_item.setText(score_value, score_label)
                elif 'Scorer' in score_key:
                    new_item.setText(score_value, scorer_name)
                elif 'Brain Region' in score_key:
                    new_item.setText(score_value, brain_region)
                elif 'Start Time' in score_key:
                    new_item.setText(score_value, str(s_ms))
                elif 'Stop Time' in score_key:
                    new_item.setText(score_value, str(e_ms))
                elif 'Duration' in score_key:
                    try:
                        dur_ms = float(e_ms) - float(s_ms)
                        if dur_ms > 0:
                            new_item.setText(score_value, f"{dur_ms:.3f}")
                        else:
                            new_item.setText(score_value, "")
                    except Exception:
                        new_item.setText(score_value, "")
                elif 'Behavioral State' in score_key:
                    new_item.setText(score_value, behavioral_state)
                elif 'Settings File' in score_key:
                    # Try to copy from original EOI item if available
                    if idx < len(selected_items):
                        for eoi_key, eoi_value in self.EOI_headers.items():
                            if 'Settings' in eoi_key:
                                new_item.setText(score_value, selected_items[idx].text(eoi_value))
                                break

            # Add to scores tree
            self.scores.addTopLevelItem(new_item)

        # Remove items from EOI tree in reverse order (to maintain correct indices)
        for index in sorted(indices_to_remove, reverse=True):
            self.EOI.takeTopLevelItem(index)

        # Update events detected count
        self.events_detected.setText(str(self.EOI.topLevelItemCount()))
        
        # Notify user if filtering was applied and/or duplicates were skipped
        messages = []
        if metadata_rows:
            added = len(eoi_times) - duplicates_skipped
            selected = len(selected_items)
            if added < selected:
                messages.append(f"Added {added} of {selected} selected EOIs to Score tab.\n"
                               f"{selected - added} EOIs were filtered out by region preset criteria "
                               "(duration/behavior/speed thresholds).")
        
        if duplicates_skipped > 0:
            messages.append(f"Skipped {duplicates_skipped} duplicate event(s) already in Score tab.")
        
        if messages:
            QtWidgets.QMessageBox.information(
                self, "EOIs Added to Score",
                "\n\n".join(messages)
            )

    def deleteEOI(self):
        '''Delete selected EOIs (supports multiple selections)'''
        root = self.EOI.invisibleRootItem()
        selected_items = self.EOI.selectedItems()
        
        if not selected_items:
            return

        # Get ID column index
        id_column = None
        for key, id_col in self.EOI_headers.items():
            if 'ID' in key:
                id_column = id_col
                break

        # Delete items in reverse order to maintain proper indices
        # Convert to list to avoid modifying selection while iterating
        items_to_delete = list(selected_items)
        
        for item in items_to_delete:
            # Get the ID for removal from tracking list
            if id_column is not None:
                ID = item.data(id_column, 0)
                if ID in self.IDs:
                    self.IDs.pop(self.IDs.index(ID))
            
            # Remove the item from the tree
            (item.parent() or root).removeChild(item)
            
            # Update the events detected count
            new_detected_events = str(int(self.events_detected.text()) - 1)
            self.events_detected.setText(new_detected_events)

        # Select the next available item
        if self.EOI.topLevelItemCount() > 0:
            next_item = self.EOI.topLevelItem(0)
            if next_item is not None:
                next_item.setSelected(True)

    def get_automatic_detection_filename(self):

        set_basename = os.path.basename(os.path.splitext(self.mainWindow.current_set_filename)[0])
        save_directory = os.path.dirname(self.score_filename.text())

        self.id_abbreviations[self.eoi_method.currentText().lower()]
        detection_method = self.eoi_method.currentText()

        save_filename = os.path.join(save_directory, '%s_%s.txt' % (set_basename, detection_method))

        return save_filename

    def loadAutomaticEOIs(self):

        # self.settings_fname = ''
        # save_filename = self.get_automatic_detection_filename()

        save_filename = self.eoi_filename.text()

        if 'Please add a source!' in save_filename:
            return

        save_filename, save_string_ext = QtWidgets.QFileDialog.getOpenFileName(self, 'Load EOI\'s',
                                                 save_filename,
                                                 'Text Files (*.txt)')
        if save_filename == '':
            return

        self.eoi_filename.setText(save_filename)

        if os.path.exists(save_filename):
            # do you want to overwrite this file?
            pass

        if os.path.exists(save_filename):
            df = pd.read_csv(save_filename, delimiter='\t')
        else:
            return

        # add the scores

        N = len(df)

        # find the IDs of any existing EOIs
        iterator = QtWidgets.QTreeWidgetItemIterator(self.EOI)

        auto_IDs = []
        while iterator.value():
            item = iterator.value()
            auto_IDs.append(item.data(0, 0))
            iterator += 1

        self.EOI.clear()
        [self.IDs.pop(self.IDs.index(ID)) for ID in auto_IDs]  # remove the id from the list of IDs

        ids_exist = any('ID' in column for column in df.columns)

        settings_filename_exist = any('Settings File' in column for column in df.columns)
        duration_exists = any('Duration' in column for column in df.columns)

        for key, value in self.EOI_headers.items():
            if 'ID' in key:
                id_value = value
            elif 'Settings File' in key:
                settings_value = value
            elif 'Start Time' in key:
                start_value = value
            elif 'Stop Time' in key:
                stop_value = value
            elif 'Duration' in key:
                duration_value = value

        for eoi_index in range(N):
            # item = QtWidgets.QTreeWidgetItem()
            item = TreeWidgetItem()

            if not ids_exist:
                ID = self.createID('Unknown')
                self.IDs.append(ID)
                item.setText(id_value, ID)

            if not settings_filename_exist:
                item.setText(settings_value, 'Unknown')

            for column in df.columns:

                if 'Unnamed' in column:
                    continue

                for key, value in self.EOI_headers.items():

                    if key == column:
                        if 'ID' in key:
                            ID = df[column][eoi_index]
                            self.IDs.append(ID)
                            item.setText(value, ID)

                        else:
                            item.setText(value, str(df[column][eoi_index]))

                    # these next statements are for the older files
                    elif 'Score' in column and 'Score:' in key:
                        item.setText(value, str(df[column][eoi_index]))

                    elif 'Start' in column and 'Start' in key:
                        item.setText(value, str(df[column][eoi_index]))

                    elif 'Stop' in column and 'Stop' in key:
                        item.setText(value, str(df[column][eoi_index]))

            # Auto-calculate duration if not present in file
            if not duration_exists:
                self._calculateAndSetDuration(item, start_value, stop_value, duration_value)

            self.EOI.addTopLevelItem(item)

        self.EOI.sortItems(start_value, QtCore.Qt.AscendingOrder)

        self.events_detected.setText(str(len(df)))

    def setEOIfilename(self):

        if not hasattr(self.mainWindow, 'current_set_filename'):
            return

        set_directory = os.path.dirname(self.mainWindow.current_set_filename)
        set_basename = os.path.basename(os.path.splitext(self.mainWindow.current_set_filename)[0])
        method = self.eoi_method.currentText()
        method = self.id_abbreviations.get(method.lower(), 'UNK')
        filename = os.path.join(set_directory, 'HFOScores',
                                set_basename,
                                '%s_%s.txt' % (set_basename, method))

        self.eoi_filename.setText(filename)

    def saveAutomaticEOIs(self):

        save_filename = self.eoi_filename.text()

        if 'Please add a source!' in save_filename:
            return

        save_filename, save_extension = QtWidgets.QFileDialog.getSaveFileName(self, 'Save EOI\'s',
                                                 save_filename,
                                                 'Text Files (*.txt)')

        if save_filename == '':
            return

        self.eoi_filename.setText(save_filename)

        # iterate through each item
        start_times = []
        stop_times = []
        ids = []

        # get the column values for each of the parameters
        for key, value in self.EOI_headers.items():
            if 'ID' in key:
                id_value = value
            elif 'Start' in key:
                start_value = value
            elif 'Stop' in key:
                stop_value = value

        for item_count in range(self.EOI.topLevelItemCount()):
            item = self.EOI.topLevelItem(item_count)

            ids.append(item.data(id_value, 0))
            start_times.append(item.data(start_value, 0))
            stop_times.append(item.data(stop_value, 0))

        data_dict = {}
        for key, value in self.score_headers.items():
            if 'ID' in key:
                data_dict[key] = ids

            elif 'Start' in key:
                data_dict[key] = pd.Series(start_times)

            elif 'Stop' in key:
                data_dict[key] = pd.Series(stop_times)

        df = pd.DataFrame(data_dict)

        # get filename

        # make the directory name if it does not exists already
        if not os.path.exists(os.path.dirname(save_filename)):
            os.makedirs(os.path.dirname(save_filename))

        df.to_csv(save_filename, sep='\t')

    def exportEOIsForTraining(self):
        """Export current EOIs to .npy segments and manifest for DL training."""
        if not hasattr(self, 'source_filename') or not os.path.exists(self.source_filename):
            QtWidgets.QMessageBox.warning(self, "No Source", "Please load a data file first.")
            return

        if self.EOI.topLevelItemCount() == 0:
            QtWidgets.QMessageBox.warning(self, "No EOIs", "Please detect or add EOIs first.")
            return

        # Ask user for output directory
        out_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output directory for segments")
        if not out_dir:
            return

        try:
            # Get EOIs from tree
            eois = []
            for key, value in self.EOI_headers.items():
                if 'Start' in key:
                    start_col = value
                elif 'Stop' in key:
                    stop_col = value

            for item_count in range(self.EOI.topLevelItemCount()):
                item = self.EOI.topLevelItem(item_count)
                try:
                    s_ms = float(item.data(start_col, 0))
                    e_ms = float(item.data(stop_col, 0))
                    eois.append([s_ms, e_ms])
                except Exception:
                    continue

            if not eois:
                QtWidgets.QMessageBox.warning(self, "No Valid EOIs", "Could not extract EOI times.")
                return

            # Apply region-aware filtering and annotation
            metadata_rows = None
            if hasattr(self, 'region_profile') and self.region_profile:
                eois, metadata_rows = self._filter_and_annotate_eois_for_export(np.asarray(eois), self.region_profile)
                if not eois:
                    QtWidgets.QMessageBox.warning(self, "No EOIs After Filters", "All EOIs were excluded by duration/behavior filters.")
                    return

            # Get signal
            raw_data, Fs = self.settingsWindow.loaded_sources[self.source_filename]
            raw_data = np.asarray(raw_data, dtype=np.float32)

            # Export with annotations
            from core.eoi_exporter import export_eois_for_training
            manifest_path = export_eois_for_training(raw_data, Fs, np.asarray(eois), out_dir, metadata=metadata_rows)

            QtWidgets.QMessageBox.information(
                self, "Export Complete",
                f"Exported {len(eois)} EOIs to:\n{out_dir}\n\nManifest: {manifest_path}"
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export Error", f"Error exporting EOIs: {e}")
            import traceback
            traceback.print_exc()


    def exportHFOMetrics(self):
        """Export HFO metrics CSV with pathology scores and behavioral breakdown."""
        if not hasattr(self, 'source_filename') or not os.path.exists(self.source_filename):
            QtWidgets.QMessageBox.warning(self, "No Source", "Please load a data file first.")
            return

        if self.scores.topLevelItemCount() == 0:
            QtWidgets.QMessageBox.warning(self, "No Scores", "Please add scores first.")
            return

        out_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output directory for HFO metrics")
        if not out_dir:
            return

        # Map score labels to event type
        pos_labels = {'ripple', 'fast ripple', 'sharp wave ripple'}

        try:
            score_col = None
            start_col = None
            stop_col = None
            scorer_col = None
            brain_region_col = None
            
            for key, value in self.score_headers.items():
                if 'Score:' in key:
                    score_col = value
                elif 'Start' in key:
                    start_col = value
                elif 'Stop' in key:
                    stop_col = value
                elif 'Scorer:' in key:
                    scorer_col = value
                elif 'Brain Region:' in key:
                    brain_region_col = value

            eois = []
            band_labels = []
            behavior_states = []
            scorers = []
            
            for item_count in range(self.scores.topLevelItemCount()):
                item = self.scores.topLevelItem(item_count)
                try:
                    lbl_txt = str(item.data(score_col, 0)).strip().lower()
                    if lbl_txt not in pos_labels:
                        continue  # skip non-HFO labels for metrics export
                    s_ms = float(item.data(start_col, 0))
                    e_ms = float(item.data(stop_col, 0))
                    eois.append([s_ms, e_ms])
                    band_labels.append(lbl_txt)
                    
                    # Extract behavioral state from item (stored as custom data)
                    behavior_state = item.data(0, QtCore.Qt.UserRole) or 'unknown'
                    behavior_states.append(behavior_state)
                    
                    try:
                        scorer = str(item.data(scorer_col, 0)).strip()
                        scorers.append(scorer if scorer else 'Unknown')
                    except:
                        scorers.append('Unknown')
                except Exception:
                    continue

            if not eois:
                QtWidgets.QMessageBox.warning(self, "No HFO Scores", "No Ripple/Fast Ripple/SWR events found to export.")
                return

            raw_data, Fs = self.settingsWindow.loaded_sources[self.source_filename]
            raw_data = np.asarray(raw_data, dtype=np.float32)
            recording_minutes = (len(raw_data) / float(Fs)) / 60.0 if Fs else 0

            # Compute HFO metrics with co-occurrence detection
            from core.hfo_classifier import HFO_Classifier
            
            durations_ms = [(stop - start) for start, stop in eois]
            ripple_mask = [lbl in {'ripple', 'sharp wave ripple'} for lbl in band_labels]
            fr_mask = [lbl == 'fast ripple' for lbl in band_labels]

            ripple_list = []
            fr_list = []
            
            for i, (start, stop, lbl) in enumerate(zip([e[0] for e in eois], [e[1] for e in eois], band_labels)):
                if lbl in {'ripple', 'sharp wave ripple'}:
                    ripple_list.append({'start_ms': start, 'end_ms': stop, 'peak_freq': 150})
                elif lbl == 'fast ripple':
                    fr_list.append({'start_ms': start, 'end_ms': stop, 'peak_freq': 350})
            
            classifier = HFO_Classifier(fs=Fs)
            classified_events = classifier.classify_events(ripple_list, fr_list)
            summary_stats = classifier.compute_summary(classified_events)
            
            ripple_durs = [d for d, m in zip(durations_ms, ripple_mask) if m]
            fr_durs = [d for d, m in zip(durations_ms, fr_mask) if m]

            ripple_count = len(ripple_durs)
            fr_count = len(fr_durs)
            total_count = len(eois)

            ripple_rate = (ripple_count / recording_minutes) if recording_minutes else 0
            fr_rate = (fr_count / recording_minutes) if recording_minutes else 0
            fr_over_ripple = (fr_count / ripple_count) if ripple_count else 0

            long_thresh = 100.0  # ms
            long_ripple_count = len([d for d in ripple_durs if d > long_thresh])
            long_ripple_pct = (long_ripple_count / ripple_count * 100.0) if ripple_count else 0

            mean_ripple_dur = float(np.mean(ripple_durs)) if ripple_durs else 0
            mean_fr_dur = float(np.mean(fr_durs)) if fr_durs else 0
            
            # Behavioral state breakdown
            rest_mask = [st == 'rest' for st in behavior_states]
            active_mask = [st == 'active' for st in behavior_states]
            
            rest_ripple_rate = (sum([m1 and m2 for m1, m2 in zip(rest_mask, ripple_mask)]) / recording_minutes) if recording_minutes else 0
            active_ripple_rate = (sum([m1 and m2 for m1, m2 in zip(active_mask, ripple_mask)]) / recording_minutes) if recording_minutes else 0
            rest_fr_rate = (sum([m1 and m2 for m1, m2 in zip(rest_mask, fr_mask)]) / recording_minutes) if recording_minutes else 0
            active_fr_rate = (sum([m1 and m2 for m1, m2 in zip(active_mask, fr_mask)]) / recording_minutes) if recording_minutes else 0
            
            # Cooccurrence by state
            cooccur_rest = sum([m1 and m2 for m1, m2 in zip(rest_mask, [e['is_cooccurrence'] for e in classified_events] if classified_events else [])])
            cooccur_active = sum([m1 and m2 for m1, m2 in zip(active_mask, [e['is_cooccurrence'] for e in classified_events] if classified_events else [])])
            rest_events = sum(rest_mask)
            active_events = sum(active_mask)

            summary_rows = [
                ("total_hfo_events", total_count),
                ("ripple_count", ripple_count),
                ("fast_ripple_count", fr_count),
                ("recording_duration_minutes", recording_minutes),
                ("ripple_rate_per_min", ripple_rate),
                ("fast_ripple_rate_per_min", fr_rate),
                ("fr_to_ripple_ratio", fr_over_ripple),
                ("long_duration_ripple_count_gt100ms", long_ripple_count),
                ("long_duration_ripple_pct_gt100ms", long_ripple_pct),
                ("mean_ripple_duration_ms", mean_ripple_dur),
                ("mean_fast_ripple_duration_ms", mean_fr_dur),
                ("ripple_fast_ripple_cooccurrence_count", summary_stats['ripple_fast_ripple_cooccurrence']),
                ("cooccurrence_rate_pct", summary_stats['cooccurrence_rate'] * 100.0),
                ("mean_pathology_score", summary_stats['mean_pathology_score']),
                ("", ""),  # Blank separator
                ("behavioral_state_breakdown", ""),
                ("rest_events_count", rest_events),
                ("active_events_count", active_events),
                ("ripple_rate_rest_per_min", rest_ripple_rate),
                ("ripple_rate_active_per_min", active_ripple_rate),
                ("fast_ripple_rate_rest_per_min", rest_fr_rate),
                ("fast_ripple_rate_active_per_min", active_fr_rate),
                ("cooccurrence_rest_count", cooccur_rest),
                ("cooccurrence_active_count", cooccur_active),
                ("cooccurrence_rate_rest_pct", (cooccur_rest / rest_events * 100.0) if rest_events else 0),
                ("cooccurrence_rate_active_pct", (cooccur_active / active_events * 100.0) if active_events else 0),
            ]

            metrics_path = os.path.join(out_dir, "hfo_metrics.csv")
            with open(metrics_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["metric", "value"])
                writer.writerows(summary_rows)

            QtWidgets.QMessageBox.information(
                self, "Export Complete",
                f"Exported HFO metrics for {total_count} events to:\n{metrics_path}"
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export Error", f"Error exporting HFO metrics: {e}")
            import traceback
            traceback.print_exc()

    def exportForDLTraining(self):
        """Export scored events (Score tab) to labeled segments and manifest for DL training."""
        if not hasattr(self, 'source_filename') or not os.path.exists(self.source_filename):
            QtWidgets.QMessageBox.warning(self, "No Source", "Please load a data file first.")
            return

        if self.scores.topLevelItemCount() == 0:
            QtWidgets.QMessageBox.warning(self, "No Scores", "Please add scores first (e.g., Ripple vs Artifact).")
            return

        out_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output directory for DL training data")
        if not out_dir:
            return

        # Map score labels to binary classes
        pos_labels = {'ripple', 'fast ripple', 'sharp wave ripple'}
        neg_labels = {'artifact'}

        try:
            score_col = None
            start_col = None
            stop_col = None
            scorer_col = None
            brain_region_col = None
            
            for key, value in self.score_headers.items():
                if 'Score:' in key:
                    score_col = value
                elif 'Start' in key:
                    start_col = value
                elif 'Stop' in key:
                    stop_col = value
                elif 'Scorer:' in key:
                    scorer_col = value
                elif 'Brain Region:' in key:
                    brain_region_col = value

            eois = []
            labels = []
            band_labels = []  # keep raw label text for metrics (ripple vs fast ripple)
            behavior_states = []  # track behavioral state for each event
            metadata_rows = []
            for item_count in range(self.scores.topLevelItemCount()):
                item = self.scores.topLevelItem(item_count)
                try:
                    lbl_txt = str(item.data(score_col, 0)).strip().lower()
                    if lbl_txt in pos_labels:
                        lbl = 1
                    elif lbl_txt in neg_labels:
                        lbl = 0
                    else:
                        continue  # skip unlabeled/other
                    s_ms = float(item.data(start_col, 0))
                    e_ms = float(item.data(stop_col, 0))
                    eois.append([s_ms, e_ms])
                    labels.append(lbl)
                    band_labels.append(lbl_txt)
                    
                    # Extract behavioral state from item (stored as custom data)
                    behavior_state = item.data(0, QtCore.Qt.UserRole) or 'unknown'
                    behavior_states.append(behavior_state)
                    
                    # Collect metadata (optional fields)
                    metadata = {'behavioral_state': behavior_state}
                    try:
                        scorer = str(item.data(scorer_col, 0)).strip()
                        if scorer and scorer != 'Unknown':
                            metadata['scorer'] = scorer
                    except:
                        pass
                    metadata_rows.append(metadata if metadata else None)
                except Exception:
                    continue

            if not eois:
                QtWidgets.QMessageBox.warning(self, "No Labeled Scores", "Only Ripple/Fast Ripple/SWR (label=1) or Artifact (label=0) are exported.")
                return

            raw_data, Fs = self.settingsWindow.loaded_sources[self.source_filename]
            raw_data = np.asarray(raw_data, dtype=np.float32)

            from core.eoi_exporter import export_labeled_eois_for_training
            manifest_path = export_labeled_eois_for_training(raw_data, Fs, np.asarray(eois), labels, out_dir, metadata=metadata_rows)

            # ----------------------------
            # Lightweight metrics summary with co-occurrence detection & behavioral state breakdown
            # ----------------------------
            try:
                from core.hfo_classifier import HFO_Classifier
                
                recording_minutes = (len(raw_data) / float(Fs)) / 60.0 if Fs else 0
                durations_ms = [(stop - start) for start, stop in eois]
                ripple_mask = [lbl in {'ripple', 'sharp wave ripple'} for lbl in band_labels]
                fr_mask = [lbl == 'fast ripple' for lbl in band_labels]

                ripple_list = []
                fr_list = []
                
                # Build event lists for classifier
                for i, (start, stop, lbl) in enumerate(zip([e[0] for e in eois], [e[1] for e in eois], band_labels)):
                    if lbl in {'ripple', 'sharp wave ripple'}:
                        ripple_list.append({'start_ms': start, 'end_ms': stop, 'peak_freq': 150})  # placeholder freq
                    elif lbl == 'fast ripple':
                        fr_list.append({'start_ms': start, 'end_ms': stop, 'peak_freq': 350})  # placeholder freq
                
                # Classify events including co-occurrences
                classifier = HFO_Classifier(fs=Fs)
                classified_events = classifier.classify_events(ripple_list, fr_list)
                summary_stats = classifier.compute_summary(classified_events)
                
                ripple_durs = [d for d, m in zip(durations_ms, ripple_mask) if m]
                fr_durs = [d for d, m in zip(durations_ms, fr_mask) if m]

                ripple_count = len(ripple_durs)
                fr_count = len(fr_durs)
                total_count = len(eois)

                ripple_rate = (ripple_count / recording_minutes) if recording_minutes else 0
                fr_rate = (fr_count / recording_minutes) if recording_minutes else 0
                fr_over_ripple = (fr_count / ripple_count) if ripple_count else 0

                long_thresh = 100.0  # ms
                long_ripple_count = len([d for d in ripple_durs if d > long_thresh])
                long_ripple_pct = (long_ripple_count / ripple_count * 100.0) if ripple_count else 0

                mean_ripple_dur = float(np.mean(ripple_durs)) if ripple_durs else 0
                mean_fr_dur = float(np.mean(fr_durs)) if fr_durs else 0
                
                # Behavioral state breakdown
                rest_mask = [st == 'rest' for st in behavior_states]
                active_mask = [st == 'active' for st in behavior_states]
                
                rest_ripple_rate = (sum([m1 and m2 for m1, m2 in zip(rest_mask, ripple_mask)]) / recording_minutes) if recording_minutes else 0
                active_ripple_rate = (sum([m1 and m2 for m1, m2 in zip(active_mask, ripple_mask)]) / recording_minutes) if recording_minutes else 0
                rest_fr_rate = (sum([m1 and m2 for m1, m2 in zip(rest_mask, fr_mask)]) / recording_minutes) if recording_minutes else 0
                active_fr_rate = (sum([m1 and m2 for m1, m2 in zip(active_mask, fr_mask)]) / recording_minutes) if recording_minutes else 0
                
                # Cooccurrence by state
                cooccur_rest = sum([m1 and m2 for m1, m2 in zip(rest_mask, [e['is_cooccurrence'] for e in classified_events] if classified_events else [])])
                cooccur_active = sum([m1 and m2 for m1, m2 in zip(active_mask, [e['is_cooccurrence'] for e in classified_events] if classified_events else [])])
                rest_events = sum(rest_mask)
                active_events = sum(active_mask)

                summary_rows = [
                    ("total_events", total_count),
                    ("ripple_count", ripple_count),
                    ("fast_ripple_count", fr_count),
                    ("ripple_rate_per_min", ripple_rate),
                    ("fast_ripple_rate_per_min", fr_rate),
                    ("fr_to_ripple_ratio", fr_over_ripple),
                    ("long_duration_ripple_count_gt100ms", long_ripple_count),
                    ("long_duration_ripple_pct_gt100ms", long_ripple_pct),
                    ("mean_ripple_duration_ms", mean_ripple_dur),
                    ("mean_fast_ripple_duration_ms", mean_fr_dur),
                    ("ripple_fast_ripple_cooccurrence_count", summary_stats['ripple_fast_ripple_cooccurrence']),
                    ("cooccurrence_rate_pct", summary_stats['cooccurrence_rate'] * 100.0),
                    ("mean_pathology_score", summary_stats['mean_pathology_score']),
                    ("", ""),  # Blank separator
                    ("behavioral_state_breakdown", ""),
                    ("rest_events_count", rest_events),
                    ("active_events_count", active_events),
                    ("ripple_rate_rest_per_min", rest_ripple_rate),
                    ("ripple_rate_active_per_min", active_ripple_rate),
                    ("fast_ripple_rate_rest_per_min", rest_fr_rate),
                    ("fast_ripple_rate_active_per_min", active_fr_rate),
                    ("cooccurrence_rest_count", cooccur_rest),
                    ("cooccurrence_active_count", cooccur_active),
                    ("cooccurrence_rate_rest_pct", (cooccur_rest / rest_events * 100.0) if rest_events else 0),
                    ("cooccurrence_rate_active_pct", (cooccur_active / active_events * 100.0) if active_events else 0),
                ]

                summary_path = os.path.join(out_dir, "hfo_metrics_summary.csv")
                with open(summary_path, "w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["metric", "value"])
                    writer.writerows(summary_rows)
            except Exception as metric_err:
                print(f"Warning: could not write metrics summary: {metric_err}")
                import traceback
                traceback.print_exc()

            QtWidgets.QMessageBox.information(
                self, "Export Complete",
                f"Exported {len(eois)} labeled segments to:\n{out_dir}\n\nManifest: {manifest_path}\nMetrics: hfo_metrics_summary.csv"
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export Error", f"Error exporting labeled scores: {e}")
            import traceback
            traceback.print_exc()

    def exportScoresForTraining(self):
        """Deprecated: Use exportForDLTraining() instead. Kept for backward compatibility."""
        self.exportForDLTraining()
        """Export scored events (Score tab) to labeled segments and manifest for DL training."""
        if not hasattr(self, 'source_filename') or not os.path.exists(self.source_filename):
            QtWidgets.QMessageBox.warning(self, "No Source", "Please load a data file first.")
            return

        if self.scores.topLevelItemCount() == 0:
            QtWidgets.QMessageBox.warning(self, "No Scores", "Please add scores first (e.g., Ripple vs Artifact).")
            return

        out_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output directory for labeled segments")
        if not out_dir:
            return

        # Map score labels to binary classes
        pos_labels = {'ripple', 'fast ripple', 'sharp wave ripple'}
        neg_labels = {'artifact'}

        try:
            score_col = None
            start_col = None
            stop_col = None
            scorer_col = None
            brain_region_col = None
            
            for key, value in self.score_headers.items():
                if 'Score:' in key:
                    score_col = value
                elif 'Start' in key:
                    start_col = value
                elif 'Stop' in key:
                    stop_col = value
                elif 'Scorer:' in key:
                    scorer_col = value
                elif 'Brain Region:' in key:
                    brain_region_col = value

            eois = []
            labels = []
            band_labels = []  # keep raw label text for metrics (ripple vs fast ripple)
            behavior_states = []  # track behavioral state for each event
            metadata_rows = []
            for item_count in range(self.scores.topLevelItemCount()):
                item = self.scores.topLevelItem(item_count)
                try:
                    lbl_txt = str(item.data(score_col, 0)).strip().lower()
                    if lbl_txt in pos_labels:
                        lbl = 1
                    elif lbl_txt in neg_labels:
                        lbl = 0
                    else:
                        continue  # skip unlabeled/other
                    s_ms = float(item.data(start_col, 0))
                    e_ms = float(item.data(stop_col, 0))
                    eois.append([s_ms, e_ms])
                    labels.append(lbl)
                    band_labels.append(lbl_txt)
                    
                    # Extract behavioral state from item (stored as custom data)
                    behavior_state = item.data(0, QtCore.Qt.UserRole) or 'unknown'
                    behavior_states.append(behavior_state)
                    
                    # Collect metadata (optional fields)
                    metadata = {'behavioral_state': behavior_state}
                    try:
                        scorer = str(item.data(scorer_col, 0)).strip()
                        if scorer and scorer != 'Unknown':
                            metadata['scorer'] = scorer
                    except:
                        pass
                    metadata_rows.append(metadata if metadata else None)
                except Exception:
                    continue

            if not eois:
                QtWidgets.QMessageBox.warning(self, "No Labeled Scores", "Only Ripple/Fast Ripple/SWR (label=1) or Artifact (label=0) are exported.")
                return

            raw_data, Fs = self.settingsWindow.loaded_sources[self.source_filename]
            raw_data = np.asarray(raw_data, dtype=np.float32)

            from core.eoi_exporter import export_labeled_eois_for_training
            manifest_path = export_labeled_eois_for_training(raw_data, Fs, np.asarray(eois), labels, out_dir, metadata=metadata_rows)

            # ----------------------------
            # Lightweight metrics summary with co-occurrence detection & behavioral state breakdown
            # ----------------------------
            try:
                from core.hfo_classifier import HFO_Classifier
                
                recording_minutes = (len(raw_data) / float(Fs)) / 60.0 if Fs else 0
                durations_ms = [(stop - start) for start, stop in eois]
                ripple_mask = [lbl in {'ripple', 'sharp wave ripple'} for lbl in band_labels]
                fr_mask = [lbl == 'fast ripple' for lbl in band_labels]

                ripple_list = []
                fr_list = []
                
                # Build event lists for classifier
                for i, (start, stop, lbl) in enumerate(zip([e[0] for e in eois], [e[1] for e in eois], band_labels)):
                    if lbl in {'ripple', 'sharp wave ripple'}:
                        ripple_list.append({'start_ms': start, 'end_ms': stop, 'peak_freq': 150})  # placeholder freq
                    elif lbl == 'fast ripple':
                        fr_list.append({'start_ms': start, 'end_ms': stop, 'peak_freq': 350})  # placeholder freq
                
                # Classify events including co-occurrences
                classifier = HFO_Classifier(fs=Fs)
                classified_events = classifier.classify_events(ripple_list, fr_list)
                summary_stats = classifier.compute_summary(classified_events)
                
                ripple_durs = [d for d, m in zip(durations_ms, ripple_mask) if m]
                fr_durs = [d for d, m in zip(durations_ms, fr_mask) if m]

                ripple_count = len(ripple_durs)
                fr_count = len(fr_durs)
                total_count = len(eois)

                ripple_rate = (ripple_count / recording_minutes) if recording_minutes else 0
                fr_rate = (fr_count / recording_minutes) if recording_minutes else 0
                fr_over_ripple = (fr_count / ripple_count) if ripple_count else 0

                long_thresh = 100.0  # ms
                long_ripple_count = len([d for d in ripple_durs if d > long_thresh])
                long_ripple_pct = (long_ripple_count / ripple_count * 100.0) if ripple_count else 0

                mean_ripple_dur = float(np.mean(ripple_durs)) if ripple_durs else 0
                mean_fr_dur = float(np.mean(fr_durs)) if fr_durs else 0
                
                # Behavioral state breakdown
                rest_mask = [st == 'rest' for st in behavior_states]
                active_mask = [st == 'active' for st in behavior_states]
                
                rest_ripple_rate = (sum([m1 and m2 for m1, m2 in zip(rest_mask, ripple_mask)]) / recording_minutes) if recording_minutes else 0
                active_ripple_rate = (sum([m1 and m2 for m1, m2 in zip(active_mask, ripple_mask)]) / recording_minutes) if recording_minutes else 0
                rest_fr_rate = (sum([m1 and m2 for m1, m2 in zip(rest_mask, fr_mask)]) / recording_minutes) if recording_minutes else 0
                active_fr_rate = (sum([m1 and m2 for m1, m2 in zip(active_mask, fr_mask)]) / recording_minutes) if recording_minutes else 0
                
                # Cooccurrence by state
                cooccur_rest = sum([m1 and m2 for m1, m2 in zip(rest_mask, [e['is_cooccurrence'] for e in classified_events] if classified_events else [])])
                cooccur_active = sum([m1 and m2 for m1, m2 in zip(active_mask, [e['is_cooccurrence'] for e in classified_events] if classified_events else [])])
                rest_events = sum(rest_mask)
                active_events = sum(active_mask)

                summary_rows = [
                    ("total_events", total_count),
                    ("ripple_count", ripple_count),
                    ("fast_ripple_count", fr_count),
                    ("ripple_rate_per_min", ripple_rate),
                    ("fast_ripple_rate_per_min", fr_rate),
                    ("fr_to_ripple_ratio", fr_over_ripple),
                    ("long_duration_ripple_count_gt100ms", long_ripple_count),
                    ("long_duration_ripple_pct_gt100ms", long_ripple_pct),
                    ("mean_ripple_duration_ms", mean_ripple_dur),
                    ("mean_fast_ripple_duration_ms", mean_fr_dur),
                    ("ripple_fast_ripple_cooccurrence_count", summary_stats['ripple_fast_ripple_cooccurrence']),
                    ("cooccurrence_rate_pct", summary_stats['cooccurrence_rate'] * 100.0),
                    ("mean_pathology_score", summary_stats['mean_pathology_score']),
                    ("", ""),  # Blank separator
                    ("behavioral_state_breakdown", ""),
                    ("rest_events_count", rest_events),
                    ("active_events_count", active_events),
                    ("ripple_rate_rest_per_min", rest_ripple_rate),
                    ("ripple_rate_active_per_min", active_ripple_rate),
                    ("fast_ripple_rate_rest_per_min", rest_fr_rate),
                    ("fast_ripple_rate_active_per_min", active_fr_rate),
                    ("cooccurrence_rest_count", cooccur_rest),
                    ("cooccurrence_active_count", cooccur_active),
                    ("cooccurrence_rate_rest_pct", (cooccur_rest / rest_events * 100.0) if rest_events else 0),
                    ("cooccurrence_rate_active_pct", (cooccur_active / active_events * 100.0) if active_events else 0),
                ]

                summary_path = os.path.join(out_dir, "hfo_metrics_summary.csv")
                with open(summary_path, "w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["metric", "value"])
                    writer.writerows(summary_rows)
            except Exception as metric_err:
                print(f"Warning: could not write metrics summary: {metric_err}")
                import traceback
                traceback.print_exc()

            QtWidgets.QMessageBox.information(
                self, "Export Complete",
                f"Exported {len(eois)} labeled segments to:\n{out_dir}\n\nManifest: {manifest_path}\nMetrics: hfo_metrics_summary.csv"
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export Error", f"Error exporting labeled scores: {e}")
            import traceback
            traceback.print_exc()

    def addManifests(self):
        """Add manifest files to the list."""
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Select Manifest CSV Files", "", "CSV Files (*.csv);;All Files (*)"
        )
        for f in files:
            if f not in [self.manifest_list.item(i).text() for i in range(self.manifest_list.count())]:
                self.manifest_list.addItem(f)

    def removeManifests(self):
        """Remove selected manifests from the list."""
        for item in self.manifest_list.selectedItems():
            self.manifest_list.takeItem(self.manifest_list.row(item))

    def clearManifests(self):
        """Clear all manifests from the list."""
        self.manifest_list.clear()

    def browseOutputDir(self):
        """Browse for output directory."""
        out_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if out_dir:
            self.output_dir_edit.setText(out_dir)

    def convertManifests(self):
        """Convert manifests to train/val splits."""
        if self.manifest_list.count() == 0:
            QtWidgets.QMessageBox.warning(self, "No Manifests", "Please add at least one manifest file.")
            return

        if not self.output_dir_edit.text():
            QtWidgets.QMessageBox.warning(self, "No Output Directory", "Please select an output directory.")
            return

        try:
            from dl_training.manifest_splitter import load_manifests, subject_wise_split, stratified_subject_split
            from dl_training.manifest_splitter import print_statistics, check_class_balance, save_splits

            # Get manifest paths
            manifest_paths = [self.manifest_list.item(i).text() for i in range(self.manifest_list.count())]

            # Update status
            self.convert_status.clear()
            self.convert_status.append("Loading manifests...")
            QtWidgets.QApplication.processEvents()

            # Load and combine
            combined_df = load_manifests(manifest_paths)

            self.convert_status.append(f"✓ Loaded {len(combined_df)} events from {len(manifest_paths)} manifests\n")
            QtWidgets.QApplication.processEvents()

            # Print initial statistics
            import io
            import sys
            old_stdout = sys.stdout
            sys.stdout = buffer = io.StringIO()

            print_statistics(combined_df, "Combined Dataset")

            # Split
            val_frac = self.val_fraction_spin.value()
            seed = self.random_seed_spin.value()

            if self.stratified_check.isChecked():
                train_df, val_df = stratified_subject_split(combined_df, val_frac, seed)
            else:
                train_df, val_df = subject_wise_split(combined_df, val_frac, seed)

            # Print split statistics
            print_statistics(train_df, "Training Set")
            print_statistics(val_df, "Validation Set")
            check_class_balance(train_df, val_df)

            # Save
            output_dir = self.output_dir_edit.text()
            save_splits(train_df, val_df, output_dir, save_metadata=True)

            sys.stdout = old_stdout
            output = buffer.getvalue()

            self.convert_status.append(output)
            self.convert_status.append("\n✓ COMPLETE! Train and validation sets created successfully.")

            QtWidgets.QMessageBox.information(
                self, "Split Complete",
                f"Created train/val splits in:\n{output_dir}\n\n"
                f"Train: {len(train_df)} events\n"
                f"Val: {len(val_df)} events"
            )

        except Exception as e:
            self.convert_status.append(f"\n❌ ERROR: {e}")
            QtWidgets.QMessageBox.critical(self, "Conversion Error", f"Error creating splits: {e}")
            import traceback
            traceback.print_exc()

    # ---------------- Train tab helpers ----------------
    def browseTrainManifest(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select train manifest", "", "CSV Files (*.csv)")
        if path:
            self.train_manifest_edit.setText(path)

    def browseValManifest(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select val manifest", "", "CSV Files (*.csv)")
        if path:
            self.val_manifest_edit.setText(path)

    def browseTrainOutDir(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select training output directory")
        if path:
            self.train_out_dir_edit.setText(path)

    def browseCheckpoint(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select checkpoint (.pt)", "", "PyTorch checkpoint (*.pt)")
        if path:
            self.ckpt_edit.setText(path)

    def browseExportOutDir(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select export output directory")
        if path:
            self.export_out_dir_edit.setText(path)

    def _append_process_output(self, proc, widget):
        text = proc.readAllStandardOutput().data().decode(errors='ignore')
        if text:
            widget.append(text)
        text_err = proc.readAllStandardError().data().decode(errors='ignore')
        if text_err:
            widget.append(text_err)

    def startTraining(self):
        train_path = self.train_manifest_edit.text().strip()
        val_path = self.val_manifest_edit.text().strip()
        out_dir = self.train_out_dir_edit.text().strip()

        if not os.path.isfile(train_path):
            QtWidgets.QMessageBox.warning(self, "Missing train manifest", "Select a valid train manifest CSV.")
            return
        if not os.path.isfile(val_path):
            QtWidgets.QMessageBox.warning(self, "Missing val manifest", "Select a valid val manifest CSV.")
            return
        if not out_dir:
            QtWidgets.QMessageBox.warning(self, "Missing output dir", "Select an output directory for training artifacts.")
            return

        args = [
            sys.executable, "-m", "stlar.dl_training.train",
            "--train", train_path,
            "--val", val_path,
            "--epochs", str(self.epochs_spin.value()),
            "--batch-size", str(self.batch_spin.value()),
            "--lr", str(self.lr_spin.value()),
            "--weight-decay", str(self.weight_decay_spin.value()),
            "--out-dir", out_dir,
            "--model-type", str(self.model_type_combo.currentData()),
        ]
        
        if self.train_cwt_check.isChecked():
            args.append("--use-cwt")

        # Add GUI flag if checkbox is checked
        if self.train_gui_check.isChecked():
            args.append("--gui")

        # Prevent overlapping runs
        if self.train_process and self.train_process.state() == QtCore.QProcess.Running:
            QtWidgets.QMessageBox.information(self, "Training running", "Please wait for the current training to finish.")
            return

        self.train_log.clear()
        self.train_log.append("Starting training...\n")

        self.train_process = QtCore.QProcess(self)
        self.train_process.readyReadStandardOutput.connect(lambda: self._append_process_output(self.train_process, self.train_log))
        self.train_process.readyReadStandardError.connect(lambda: self._append_process_output(self.train_process, self.train_log))
        self.train_process.finished.connect(lambda code, status: self._onTrainFinished(code, status))
        self.train_process.start(args[0], args[1:])

    def _onTrainFinished(self, code, status):
        if code == 0:
            self.train_log.append("\n✓ Training complete.")
            QtWidgets.QMessageBox.information(self, "Training complete", "Training finished successfully.")
        else:
            self.train_log.append(f"\n❌ Training failed (code {code}).")
            QtWidgets.QMessageBox.critical(self, "Training error", f"Training failed with exit code {code}.")

    def startExport(self):
        ckpt = self.ckpt_edit.text().strip()
        out_dir = self.export_out_dir_edit.text().strip()

        if not os.path.isfile(ckpt):
            QtWidgets.QMessageBox.warning(self, "Missing checkpoint", "Select a valid .pt checkpoint file.")
            return
        if not out_dir:
            QtWidgets.QMessageBox.warning(self, "Missing export dir", "Select an output directory.")
            return

        os.makedirs(out_dir, exist_ok=True)
        
        # Build output paths
        onnx_path = os.path.join(out_dir, "model.onnx")
        ts_path = os.path.join(out_dir, "model_traced.pt")

        args = [
            sys.executable, "-m", "stlar.dl_training.export",
            "--ckpt", ckpt,
            "--onnx", onnx_path,
            "--ts", ts_path,
            "--model-type", str(self.model_type_combo.currentData()),
        ]

        if self.train_cwt_check.isChecked():
            args.append("--use-cwt")

        # Prevent overlapping runs
        if self.export_process and self.export_process.state() == QtCore.QProcess.Running:
            QtWidgets.QMessageBox.information(self, "Export running", "Please wait for the current export to finish.")
            return

        self.train_log.append("\nStarting export...\n")

        self.export_process = QtCore.QProcess(self)
        self.export_process.readyReadStandardOutput.connect(lambda: self._append_process_output(self.export_process, self.train_log))
        self.export_process.readyReadStandardError.connect(lambda: self._append_process_output(self.export_process, self.train_log))
        self.export_process.finished.connect(lambda code, status: self._onExportFinished(code, status))
        self.export_process.start(args[0], args[1:])

    def _onExportFinished(self, code, status):
        if code == 0:
            self.train_log.append("\n✓ Export complete (TorchScript + ONNX written).")
            QtWidgets.QMessageBox.information(self, "Export complete", "Export finished successfully.")
        else:
            self.train_log.append(f"\n❌ Export failed (code {code}).")
            QtWidgets.QMessageBox.critical(self, "Export error", f"Export failed with exit code {code}.")


def hilbert_detect_events(raw_data, Fs, *, epoch, sd_num, min_duration, min_freq, max_freq,
                          required_peak_number, required_peak_sd=None, boundary_fraction=0.3, verbose=False):
    """Run the Hilbert-based automatic detection and return an array of [start_ms, stop_ms] rows."""

    if max_freq != Fs / 2 and min_freq != 0:
        filtered_data = filt.iirfilt(
            bandtype='band', data=raw_data, Fs=Fs, Wp=min_freq, Ws=max_freq,
            order=3, automatic=0, Rp=3, As=60, filttype='butter', showresponse=0
        )
    elif max_freq == Fs / 2:
        filtered_data = filt.iirfilt(
            bandtype='high', data=raw_data, Fs=Fs, Wp=min_freq, Ws=[],
            order=3, automatic=0, Rp=3, As=60, filttype='butter', showresponse=0
        )
    elif min_freq == 0:
        filtered_data = filt.iirfilt(
            bandtype='low', data=raw_data, Fs=Fs, Wp=max_freq, Ws=[],
            order=3, automatic=0, Rp=3, As=60, filttype='butter', showresponse=0
        )
    else:
        filtered_data = raw_data.copy()

    filtered_data -= np.mean(filtered_data)
    t = (1000 / Fs) * np.arange(len(filtered_data))

    analytic_signal = hilbert(filtered_data)
    hilbert_envelope = np.abs(analytic_signal)
    rectified_signal = np.abs(filtered_data)

    epoch_window = int(epoch * Fs)
    i = 0
    EOIs = []

    if verbose:
        print('Epoch window (samples): %d' % epoch_window)

    while i <= len(hilbert_envelope):

        window_t = t[i:i + epoch_window + 1]
        if len(window_t) == 0:
            break

        if verbose:
            print('Analyzing times up to %f sec (%f percent of the data)' %
                  (window_t[-1] / 1000, 100 * window_t[-1] / t[-1]))

        window_data = hilbert_envelope[i:i + epoch_window + 1]

        window_mean = np.mean(window_data)
        window_std = np.std(window_data)
        threshold = window_mean + sd_num * window_std

        eoi_signal = np.where(window_data >= threshold)[0]
        
        # If no samples exceed threshold, skip this epoch
        if eoi_signal.size == 0:
            i += epoch_window + 1
            continue

        # Ensure indices are integers; if none found, advance to next epoch
        eoi_indices = [np.asarray(eoi, dtype=int) for eoi in find_consec(eoi_signal)]
        if len(eoi_indices) == 0:
            i += epoch_window + 1
            continue

        window_EOIs = np.zeros((len(eoi_indices), 2))

        rejected_eois = []

        peri_boundary_samples = int((200 / 1000) * Fs)

        eoi_find_start_indices = np.asarray(
            [np.arange(eoi[0] - peri_boundary_samples, eoi[0]) for eoi in eoi_indices], dtype=int
        )
        eoi_find_start_indices[eoi_find_start_indices < 0] = 0
        eoi_find_start_time = window_t[eoi_find_start_indices]

        row, col = np.where(window_data[eoi_find_start_indices] <= boundary_fraction * threshold)
        # enforce integer dtype for index arrays
        row = row.astype(int)
        col = col.astype(int)

        row_consec = np.asarray(find_same_consec(row), dtype=object)
        # ensure consec_value used for indexing is integer array
        eoi_starts = [col[np.asarray(consec_value, dtype=int)] for consec_value in row_consec]
        valid_rows = np.unique(row).astype(int)

        eoi_starts = [
            eoi_find_start_time[row_index, np.amax(eoi_starts[np.where(valid_rows == row_index)[0][0]])]
            for row_index in valid_rows
        ]

        window_EOIs[valid_rows, 0] = eoi_starts

        rejected_eois.extend(np.setdiff1d(np.arange(len(eoi_indices)), np.unique(row)))

        eoi_find_stop_indices = np.asarray(
            [np.arange(eoi[-1] + 1, eoi[-1] + peri_boundary_samples + 1) for eoi in eoi_indices], dtype=int
        )
        eoi_find_stop_indices[eoi_find_stop_indices > len(window_t) - 1] = len(window_t) - 1

        eoi_find_stop_time = window_t[eoi_find_stop_indices]

        row, col = np.where(window_data[eoi_find_stop_indices] <= boundary_fraction * threshold)
        # enforce integer dtype for index arrays
        row = row.astype(int)
        col = col.astype(int)
        row_consec = np.asarray(find_same_consec(row), dtype=object)
        # ensure consec_value used for indexing is integer array
        eoi_stops = [col[np.asarray(consec_value, dtype=int)] for consec_value in row_consec]
        valid_rows = np.unique(row).astype(int)

        eoi_stops = [
            eoi_find_stop_time[row_index, np.amin(eoi_stops[np.where(valid_rows == row_index)[0][0]])]
            for row_index in valid_rows
        ]

        window_EOIs[np.unique(row).astype(int), 1] = eoi_stops

        rejected_eois.extend(np.setdiff1d(np.arange(len(eoi_indices)), np.unique(row)))

        if rejected_eois != []:
            window_EOIs = np.delete(window_EOIs, rejected_eois, axis=0)

        if len(window_EOIs) == 0:
            i += epoch_window + 1
            continue

        latest_time = window_EOIs[0, -1]
        latest_index = 0
        rejected_eois = []

        for eoi_index, eoi in enumerate(window_EOIs):

            if eoi_index != 0:
                within_previous_bool = (eoi <= latest_time)
                if sum(within_previous_bool) == 2:
                    rejected_eois.append(eoi_index)

                elif sum(within_previous_bool) == 1:

                    window_EOIs[latest_index, 1] = eoi[-1]
                    rejected_eois.append(eoi_index)
                    latest_time = eoi[-1]

                elif sum(within_previous_bool) == 0:
                    latest_time = eoi[-1]
                    latest_index = eoi_index

        if rejected_eois != []:
            window_EOIs = np.delete(window_EOIs, rejected_eois, axis=0)

        if len(window_EOIs) == 0:
            i += epoch_window + 1
            continue

        latest_time = window_EOIs[0, -1]
        latest_index = 0
        rejected_eois = []

        for eoi_index, eoi in enumerate(window_EOIs):

            if eoi_index != 0:

                if eoi[0] - latest_time < 10:
                    latest_time = eoi[-1]
                    window_EOIs[latest_index, -1] = latest_time
                    rejected_eois.append(eoi_index)
                else:
                    latest_time = eoi[-1]
                    latest_index = eoi_index

        if rejected_eois != []:
            window_EOIs = np.delete(window_EOIs, rejected_eois, axis=0)

        if len(window_EOIs) == 0:
            i += epoch_window + 1
            continue

        rejected_eois = np.where(np.diff(window_EOIs) < min_duration)[0]
        if len(rejected_eois) > 0:
            window_EOIs = np.delete(window_EOIs, rejected_eois, axis=0)

        i += epoch_window + 1

        if len(window_EOIs) == 0:
            continue

        if required_peak_sd is None:
            required_peak_threshold = None
        else:
            required_peak_threshold = window_mean + required_peak_sd * window_std

        if len(EOIs) != 0:

            if window_EOIs[0, 0] - EOIs[-1, 1] < 10:
                EOIs[-1, 1] = window_EOIs[0, 0]
                window_EOIs = window_EOIs[1:, :]

                eoi_data = rectified_signal[int(Fs * EOIs[-1, 0] / 1000):int(Fs * EOIs[-1, 1] / 1000) + 1]

                peak_indices = detect_peaks(eoi_data, threshold=0)
                if not len(np.where(eoi_data[peak_indices] >= window_mean + 2 * window_std)[0]) >= 6:
                    EOIs = EOIs[:-1, :]

            window_EOIs = RejectEOIs(window_EOIs, rectified_signal, Fs, required_peak_threshold,
                                     required_peak_number)

            EOIs = np.vstack((EOIs, window_EOIs))
        else:

            EOIs = RejectEOIs(window_EOIs, rectified_signal, Fs, required_peak_threshold,
                              required_peak_number)

    if len(EOIs) == 0:
        return np.asarray([])

    return np.asarray(EOIs)


def HilbertDetection(self):
    try:
        if not hasattr(self, 'source_filename') or self.source_filename is None:
            self.progressSignal.progress.emit("Hilbert: Error - No source file loaded")
            return

        if not os.path.exists(self.source_filename):
            self.progressSignal.progress.emit("Hilbert: Error - Source file not found")
            return

        self.progressSignal.progress.emit("Hilbert: 10% - Loading data")
        raw_data, Fs = self.settingsWindow.loaded_sources[self.source_filename]

        self.progressSignal.progress.emit("Hilbert: 30% - Running detection")
        EOIs = hilbert_detect_events(
            raw_data,
            Fs,
            epoch=self.epoch,
            sd_num=self.sd_num,
            min_duration=self.min_duration,
            min_freq=self.min_freq,
            max_freq=self.max_freq,
            required_peak_number=self.required_peak_number,
            required_peak_sd=self.required_peak_sd,
            boundary_fraction=self.boundary_fraction,
        )

        if EOIs is None or len(EOIs) == 0:
            print('No EOIs were found!')
            self.progressSignal.progress.emit("Hilbert: Complete - 0 events")
            return

        self.progressSignal.progress.emit(f"Hilbert: 70% - Found {len(EOIs)} events")
        self.events_detected.setText(str(len(EOIs)))

        for key, value in self.EOI_headers.items():
            if 'ID' in key:
                ID_value = value
            elif 'Start' in key:
                start_value = value
            elif 'Stop' in key:
                stop_value = value
            elif 'Duration' in key:
                duration_value = value
            elif 'Settings' in key:
                settings_value = value

        for EOI in EOIs:
            EOI_item = TreeWidgetItem()

            new_id = self.createID(self.eoi_method.currentText())
            self.IDs.append(new_id)
            EOI_item.setText(ID_value, new_id)
            EOI_item.setText(start_value, str(EOI[0]))
            EOI_item.setText(stop_value, str(EOI[1]))
            try:
                dur_ms = float(EOI[1]) - float(EOI[0])
                if dur_ms > 0:
                    EOI_item.setText(duration_value, f"{dur_ms:.3f}")
                else:
                    EOI_item.setText(duration_value, "")
            except Exception:
                EOI_item.setText(duration_value, "")
            EOI_item.setText(settings_value, self.settings_fname)

            self.AddItemSignal.childAdded.emit(EOI_item)
        self.progressSignal.progress.emit("Hilbert: 100% - Complete")
    except KeyboardInterrupt:
        print('Hilbert detection was interrupted by user')
        self.progressSignal.progress.emit("Hilbert: Cancelled by user")
        return
    except Exception as e:
        print(f'Error during Hilbert detection: {e}')
        self.progressSignal.progress.emit(f"Hilbert: Error - {e}")
        import traceback
        traceback.print_exc()
        return


def _convert_pyhfo_results_to_eois(hfos, Fs):
    """Attempt to convert pyHFO results into Nx2 array of [start_ms, stop_ms].
    This handles a few common structures; falls back to empty on failure."""
    import numpy as np

    # Case 1: list of dicts with seconds
    if isinstance(hfos, (list, tuple)) and len(hfos) and isinstance(hfos[0], dict):
        start_keys = ['start', 'start_time', 't_start', 'onset']
        stop_keys = ['end', 'stop_time', 't_end', 'offset']
        rows = []
        for ev in hfos:
            s = None
            e = None
            for k in start_keys:
                if k in ev:
                    s = ev[k]
                    break
            for k in stop_keys:
                if k in ev:
                    e = ev[k]
                    break
            if s is None or e is None:
                continue
            # Many libs report seconds; if clearly too small, assume seconds
            if max(abs(s), abs(e)) < 1e6:  # not already in ms
                s_ms = float(s) * 1000.0
                e_ms = float(e) * 1000.0
            else:
                s_ms = float(s)
                e_ms = float(e)
            rows.append([s_ms, e_ms])
        if rows:
            return np.asarray(rows, dtype=float)

    # Case 1.5: tuple like (array, channel_name)
    if isinstance(hfos, tuple) and len(hfos) >= 1:
        hfos = hfos[0]

    # Case 2: dict with numpy arrays in samples
    if isinstance(hfos, dict):
        s = None
        e = None
        for k in ['start_samples', 'starts', 'start_idx']:
            if k in hfos:
                s = hfos[k]
                break
        for k in ['end_samples', 'ends', 'stop_idx']:
            if k in hfos:
                e = hfos[k]
                break
        if s is not None and e is not None:
            s = np.asarray(s)
            e = np.asarray(e)
            ms = 1000.0 * s / float(Fs)
            me = 1000.0 * e / float(Fs)
            return np.column_stack([ms, me]).astype(float)

    # Case 3: Nx2 numpy array in seconds or samples
    try:
        arr = np.asarray(hfos)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            first_two = arr[:, :2]
            # Heuristic: if values look like small (<1e6), could be seconds; if integers and large, samples
            if np.issubdtype(first_two.dtype, np.integer):
                ms = 1000.0 * first_two[:, 0] / float(Fs)
                me = 1000.0 * first_two[:, 1] / float(Fs)
                return np.column_stack([ms, me]).astype(float)
            else:
                # assume seconds
                return (first_two.astype(float) * 1000.0)
    except Exception:
        pass

    return np.asarray([])


def PyHFODetection(self):
    try:
        if not hasattr(self, 'source_filename'):
            return
        if not os.path.exists(self.source_filename):
            return

        raw_data, Fs = self.settingsWindow.loaded_sources[self.source_filename]

        # Use local Hilbert detector implementation
        try:
            # Get parameters from the window
            epoch_time = int(getattr(self, 'pyhfo_epoch', 10*60))
            sd_threshold = float(getattr(self, 'pyhfo_sd_num', 5))
            min_window = float(getattr(self, 'pyhfo_min_duration', 0.01))
            pass_band = int(getattr(self, 'pyhfo_min_freq', 80))
            stop_band = int(getattr(self, 'pyhfo_max_freq', 500))
            n_jobs = int(getattr(self, 'pyhfo_n_jobs', 1))

            # Build detector parameters
            args = ParamHIL(
                sample_freq=float(Fs),
                pass_band=pass_band,
                stop_band=stop_band,
                epoch_time=epoch_time,
                sd_threshold=sd_threshold,
                min_window=min_window,
                n_jobs=n_jobs,
            )

            print(f"pyHFO detection starting: {pass_band}-{stop_band} Hz, {n_jobs} cores, {epoch_time}s epochs...")

            detector = set_HIL_detector(args)

            # Ensure float data
            signal = np.asarray(raw_data, dtype=np.float32)

            # Run detection on single channel; HFODetector expects channel name as str
            detection_results = detector.detect(signal, 'chn1')

            # HFODetector returns (HFOs, channel_name); take the HFOs array
            if isinstance(detection_results, tuple) and len(detection_results) >= 1:
                detection_results = detection_results[0]

            # Convert results to EOI format [start_ms, stop_ms]
            EOIs = _convert_pyhfo_results_to_eois(detection_results, Fs)

        except AttributeError:
            # If the detector doesn't have a simple .detect() method, 
            # fall back to a simpler approach or notify user
            print('pyHFO detector interface not compatible with automated calling')
            self.mainWindow.ErrorDialogue.myGUI_signal.emit("PyHFOAPIError")
            return

        if EOIs is None or len(EOIs) == 0:
            print('No EOIs were found by pyHFO!')
            return

        print(f"pyHFO detection complete: {len(EOIs)} events found")
        self.events_detected.setText(str(len(EOIs)))

        for key, value in self.EOI_headers.items():
            if 'ID' in key:
                ID_value = value
            elif 'Start' in key:
                start_value = value
            elif 'Stop' in key:
                stop_value = value
            elif 'Duration' in key:
                duration_value = value
            elif 'Settings' in key:
                settings_value = value

        for EOI in EOIs:
            EOI_item = TreeWidgetItem()

            new_id = self.createID(self.eoi_method.currentText())
            self.IDs.append(new_id)
            EOI_item.setText(ID_value, new_id)
            EOI_item.setText(start_value, str(EOI[0]))
            EOI_item.setText(stop_value, str(EOI[1]))
            try:
                dur_ms = float(EOI[1]) - float(EOI[0])
                if dur_ms > 0:
                    EOI_item.setText(duration_value, f"{dur_ms:.3f}")
                else:
                    EOI_item.setText(duration_value, "")
            except Exception:
                EOI_item.setText(duration_value, "")
            EOI_item.setText(settings_value, getattr(self, 'settings_fname', 'N/A'))

            self.AddItemSignal.childAdded.emit(EOI_item)
    except KeyboardInterrupt:
        print('pyHFO detection was interrupted by user')
        return
    except Exception as e:
        print(f'Error during pyHFO detection: {e}')
        import traceback
        traceback.print_exc()
        return


def STEDetection(self):
    try:
        if not hasattr(self, 'source_filename') or self.source_filename is None or not os.path.exists(self.source_filename):
            self.progressSignal.progress.emit("STE: Error - No source file loaded")
            return

        self.progressSignal.progress.emit("STE: 10% - Loading data")
        raw_data, Fs = self.settingsWindow.loaded_sources[self.source_filename]

        self.progressSignal.progress.emit("STE: 30% - Running detection")
        EOIs = ste_detect_events(
            raw_data, Fs,
            threshold=self.ste_threshold,
            window_size=self.ste_window_size,
            overlap=self.ste_overlap,
            min_freq=self.ste_min_freq,
            max_freq=self.ste_max_freq
        )

        self.progressSignal.progress.emit(f"STE: 70% - Found {len(EOIs) if EOIs is not None and len(EOIs) > 0 else 0} events")
        _process_detection_results(self, EOIs)
        self.progressSignal.progress.emit("STE: 100% - Complete")

    except Exception as e:
        print(f'Error during STE detection: {e}')
        self.progressSignal.progress.emit(f"STE: Error - {e}")
        import traceback
        traceback.print_exc()


def MNIDetection(self):
    try:
        if not hasattr(self, 'source_filename') or self.source_filename is None or not os.path.exists(self.source_filename):
            self.progressSignal.progress.emit("MNI: Error - No source file loaded")
            return

        self.progressSignal.progress.emit("MNI: 10% - Loading data")
        raw_data, Fs = self.settingsWindow.loaded_sources[self.source_filename]

        self.progressSignal.progress.emit("MNI: 30% - Running detection")
        EOIs = mni_detect_events(
            raw_data, Fs,
            baseline_window=self.mni_baseline_window,
            threshold_percentile=self.mni_threshold_percentile,
            min_freq=self.mni_min_freq
        )

        self.progressSignal.progress.emit(f"MNI: 70% - Found {len(EOIs) if EOIs is not None and len(EOIs) > 0 else 0} events")
        _process_detection_results(self, EOIs)
        self.progressSignal.progress.emit("MNI: 100% - Complete")

    except Exception as e:
        print(f'Error during MNI detection: {e}')
        self.progressSignal.progress.emit(f"MNI: Error - {e}")
        import traceback
        traceback.print_exc()


def DLDetection(self):
    try:
        if not hasattr(self, 'source_filename') or self.source_filename is None or not os.path.exists(self.source_filename):
            self.progressSignal.progress.emit("DL: Error - No source file loaded")
            return

        self.progressSignal.progress.emit("DL: 10% - Loading data")
        raw_data, Fs = self.settingsWindow.loaded_sources[self.source_filename]

        # --- CWT / Scalogram Detection Path ---
        if getattr(self, 'dl_use_cwt', False):
            if CWT_InferenceDataset is None or torch is None or build_model is None:
                self.progressSignal.progress.emit("DL: Error - CWT inference module or torch not available")
                return

            self.progressSignal.progress.emit("DL: 20% - Initializing CWT Model")
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if device == 'cuda':
                cuda_device = torch.cuda.current_device()
                device_name = torch.cuda.get_device_name(cuda_device)
                self.progressSignal.progress.emit(f"DL: Using GPU ({device_name})")
            else:
                self.progressSignal.progress.emit("DL: Using CPU")
            
            # Instantiate HFO_2D_CNN from consolidated training module
            # Model type 6 = HFO_2D_CNN (2D CNN for scalogram input)
            model = build_model(model_type=6, num_classes=1).to(device)
            
            # Load weights
            if not hasattr(self, 'dl_model_path') or not os.path.exists(self.dl_model_path):
                self.progressSignal.progress.emit("DL: Error - Invalid model path")
                return
                
            try:
                # Try loading as TorchScript first (most efficient for inference)
                try:
                    model = torch.jit.load(self.dl_model_path, map_location=device)
                except Exception:
                    # Fallback: try loading as state dict
                    state_dict = torch.load(self.dl_model_path, map_location=device, weights_only=True)
                    model.load_state_dict(state_dict)
            except Exception as e:
                self.progressSignal.progress.emit(f"DL: Error loading weights - {e}")
                return
            
            model.eval()
            
            # Prepare sliding windows
            self.progressSignal.progress.emit("DL: 30% - Preparing Data segments")
            window_size = getattr(self, 'dl_window_size', 1.0) # seconds
            overlap = getattr(self, 'dl_overlap', 0.5)
            
            win_samp = int(window_size * Fs)
            step_samp = int(win_samp * (1 - overlap))
            
            if step_samp < 1: step_samp = 1
            
            segments = []
            start_times = []
            
            # Extract segments
            # Ensure raw_data is 1D
            sig = np.array(raw_data).flatten()
            
            for i in range(0, len(sig) - win_samp, step_samp):
                segments.append(sig[i : i + win_samp])
                start_times.append(i / Fs)
            
            if not segments:
                self.progressSignal.progress.emit("DL: Complete - Signal too short")
                return

            # Create Dataset and Loader
            # CWT_InferenceDataset handles the CWT scalogram conversion automatically
            # Optional: Enable debug mode by setting environment variable STLAR_DEBUG_CWT
            debug_cwt_dir = None
            if os.environ.get('STLAR_DEBUG_CWT'):
                debug_cwt_dir = os.environ.get('STLAR_DEBUG_CWT')
            
            dataset = CWT_InferenceDataset(segments, fs=Fs, debug_cwt_dir=debug_cwt_dir)
            batch_size = getattr(self, 'dl_batch_size', 32)
            
            # Use pad_collate_fn_2d for proper batching of 2D tensors with variable time length
            try:
                from dl_training.data import pad_collate_fn_2d
            except ImportError:
                from ..dl_training.data import pad_collate_fn_2d
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn_2d)
            
            self.progressSignal.progress.emit(f"DL: 40% - Running Inference on {len(segments)} segments")
            
            detected_events = []
            threshold = getattr(self, 'dl_threshold', 0.75)
            
            # Timing for progress estimation
            import time
            inference_start_time = time.time()
            batch_times = []
            
            global_idx = 0
            with torch.no_grad():
                for batch_idx, batch_data in enumerate(loader):
                    batch_start_time = time.time()
                    
                    # CWT_InferenceDataset returns only tensors (no labels for inference)
                    # pad_collate_fn_2d returns either tensors or (tensors, labels)
                    if isinstance(batch_data, tuple):
                        inputs, _ = batch_data
                    else:
                        inputs = batch_data
                    
                    inputs = inputs.to(device)
                    try:
                        outputs = model(inputs)
                    except RuntimeError as e:
                        if "shape" in str(e) and "invalid for input" in str(e):
                            raise RuntimeError(
                                f"Model input shape mismatch: {e}\n"
                                "You have 'Use CWT (Scalogram)' enabled, which feeds 2D images to the model.\n"
                                "If your model expects raw 1D signals (e.g. trained via train-dl), please UNCHECK 'Use CWT' in settings."
                            ) from e
                        raise e

                    if outputs.shape[1] > 1:
                        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                    else:
                        probs = torch.sigmoid(outputs).squeeze(1).cpu().numpy()
                    
                    for p in probs:
                        if p >= threshold:
                            t_start = start_times[global_idx]
                            t_stop = t_start + window_size
                            # Convert to ms for ScoreWindow
                            detected_events.append([t_start * 1000.0, t_stop * 1000.0])
                        global_idx += 1
                    
                    # Track batch processing time
                    batch_time = time.time() - batch_start_time
                    batch_times.append(batch_time)
                    
                    if batch_idx % 10 == 0:
                        progress = 40 + int(50 * (global_idx / len(segments)))
                        elapsed_time = time.time() - inference_start_time
                        
                        # Calculate ETA based on average batch time
                        if len(batch_times) > 10:  # Need at least 10 batches for reliable estimate
                            avg_batch_time = sum(batch_times[-10:]) / min(len(batch_times), 10)  # Use last 10 batches
                            total_batches = len(loader)
                            remaining_batches = total_batches - (batch_idx + 1)
                            eta_seconds = remaining_batches * avg_batch_time
                            
                            # Format elapsed and ETA
                            elapsed_str = f"{int(elapsed_time // 60)}m {int(elapsed_time % 60)}s"
                            if eta_seconds < 60:
                                eta_str = f"{int(eta_seconds)}s"
                            elif eta_seconds < 3600:
                                eta_str = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
                            else:
                                eta_str = f"{int(eta_seconds // 3600)}h {int((eta_seconds % 3600) // 60)}m"
                            
                            self.progressSignal.progress.emit(
                                f"DL: {progress}% - Inferencing ({elapsed_str} elapsed, ~{eta_str} remaining)..."
                            )
                        else:
                            elapsed_str = f"{int(elapsed_time // 60)}m {int(elapsed_time % 60)}s"
                            self.progressSignal.progress.emit(f"DL: {progress}% - Inferencing ({elapsed_str} elapsed)...")

            # Merge overlapping/contiguous detected windows into events with variable duration
            # This prevents all EOIs from being exactly window_size duration
            if detected_events:
                detected_events = np.array(detected_events)
                # Sort by start time
                detected_events = detected_events[detected_events[:, 0].argsort()]
                
                merged = []
                current_start = detected_events[0][0]
                current_stop = detected_events[0][1]
                
                for i in range(1, len(detected_events)):
                    next_start = detected_events[i][0]
                    next_stop = detected_events[i][1]
                    
                    # If windows overlap or are adjacent (within small gap), merge them
                    gap_ms = next_start - current_stop
                    merge_threshold_ms = window_size * 1000.0 * (1 - overlap) * 1.5  # 1.5x step size
                    
                    if gap_ms <= merge_threshold_ms:
                        # Merge: extend current event
                        current_stop = max(current_stop, next_stop)
                    else:
                        # Gap too large: save current event and start new one
                        merged.append([current_start, current_stop])
                        current_start = next_start
                        current_stop = next_stop
                
                # Don't forget the last event
                merged.append([current_start, current_stop])
                
                # Apply minimum duration threshold (default 50ms for HFO events)
                min_duration_ms = 50.0  # Minimum HFO duration
                filtered = []
                for event_start, event_stop in merged:
                    duration = event_stop - event_start
                    if duration < min_duration_ms:
                        # Extend short events to minimum duration (centered)
                        deficit = min_duration_ms - duration
                        event_start -= deficit / 2.0
                        event_stop += deficit / 2.0
                        # Ensure non-negative times
                        if event_start < 0:
                            event_stop += abs(event_start)
                            event_start = 0
                    filtered.append([event_start, event_stop])
                
                detected_events = filtered
            
            self.progressSignal.progress.emit(f"DL: 90% - Found {len(detected_events)} events (after merging)")
            
            # Populate Tree
            if detected_events:
                _process_detection_results(self, np.array(detected_events))
            
            self.progressSignal.progress.emit("DL: 100% - Complete")
            return

        # --- Existing DL Detection Path (Raw/1D) ---
        # If EOIs already exist, perform classification of those EOIs; otherwise run DL detection.
        existing_eois = []
        try:
            for key, value in self.EOI_headers.items():
                if 'Start' in key:
                    start_col = value
                elif 'Stop' in key:
                    stop_col = value
            iterator = QtWidgets.QTreeWidgetItemIterator(self.EOI)
            while iterator.value():
                item = iterator.value()
                try:
                    s_ms = float(item.data(start_col, 0))
                    e_ms = float(item.data(stop_col, 0))
                    existing_eois.append([s_ms, e_ms])
                except Exception:
                    pass
                iterator += 1
        except Exception:
            existing_eois = []

        if len(existing_eois) > 0:
            # Classify existing EOIs
            self.progressSignal.progress.emit(f"DL: 20% - Verifying model file")
            
            # Check model path
            if not hasattr(self, 'dl_model_path') or not self.dl_model_path:
                self.progressSignal.progress.emit("DL: Error - No model path set. Configure model in Settings → Deep Learning")
                return
            
            if not os.path.exists(self.dl_model_path):
                self.progressSignal.progress.emit(f"DL: Error - Model file not found: {self.dl_model_path}")
                return
            
            self.progressSignal.progress.emit(f"DL: 30% - Loading model from {os.path.basename(self.dl_model_path)}")
            
            from core.Detector import dl_classify_segments
            
            # Progress callback to update GUI during model loading
            def dl_progress(msg):
                if "timeout" in msg.lower() or "error" in msg.lower():
                    self.progressSignal.progress.emit(f"DL: {msg}")
                elif "torch.jit.load" in msg.lower() or "torch.load" in msg.lower():
                    self.progressSignal.progress.emit(f"DL: 30% - {msg}")
                elif "loaded successfully" in msg.lower():
                    self.progressSignal.progress.emit(f"DL: 40% - Model ready, starting inference")
            
            probs, labels = dl_classify_segments(
                raw_data, Fs, existing_eois,
                model_path=self.dl_model_path,
                threshold=self.dl_threshold,
                batch_size=self.dl_batch_size,
                progress_callback=dl_progress
            )
            
            self.progressSignal.progress.emit(f"DL: 50% - Model loaded, classifying {len(existing_eois)} EOIs")

            # Populate scores tree with classification results
            self.progressSignal.progress.emit("DL: 70% - Populating results")
            for idx, eoi in enumerate(existing_eois):
                new_item = TreeWidgetItem()
                id_val = self.createID('Deep Learning')
                self.IDs.append(id_val)

                for key, value in self.score_headers.items():
                    if 'ID' in key:
                        new_item.setText(value, id_val)
                    elif 'Score:' in key:
                        # Map positive/negative to Ripple/Artifact; adjust as needed
                        lbl = labels[idx] if idx < len(labels) else 'unknown'
                        new_item.setText(value, 'Ripple' if lbl == 'positive' else 'Artifact')
                    elif 'Start' in key:
                        new_item.setText(value, str(eoi[0]))
                    elif 'Stop' in key:
                        new_item.setText(value, str(eoi[1]))
                    elif 'Duration' in key:
                        try:
                            d_ms = float(eoi[1]) - float(eoi[0])
                            new_item.setText(value, f"{d_ms:.3f}")
                        except Exception:
                            new_item.setText(value, "")
                    elif 'Settings File' in key:
                        new_item.setText(value, getattr(self, 'settings_fname', 'N/A'))
                    elif 'Scorer' in key:
                        # Store probability in scorer field for quick review
                        p = probs[idx] if idx < len(probs) else 0.0
                        new_item.setText(value, f"DL(p={p:.3f})")
                self.scores.addTopLevelItem(new_item)
            self.progressSignal.progress.emit(f"DL: 70% - Processed {len(existing_eois)} EOIs")
            self.progressSignal.progress.emit("DL: 100% - Complete")
            return
        else:
            # No EOIs present; run DL detection as a fallback
            self.progressSignal.progress.emit("DL: 30% - Running detection")
            
            # Progress callback for model loading
            def dl_progress(msg):
                self.progressSignal.progress.emit(f"DL: {msg}")
            
            from core.Detector import dl_detect_events
            EOIs = dl_detect_events(
                raw_data, Fs,
                model_path=self.dl_model_path,
                threshold=self.dl_threshold,
                batch_size=self.dl_batch_size,
                window_size=getattr(self, 'dl_window_size', 1.0),
                overlap=getattr(self, 'dl_overlap', 0.5),
                progress_callback=dl_progress
            )
            if EOIs is None or len(EOIs) == 0:
                self.progressSignal.progress.emit("DL: Complete - 0 events")
            else:
                self.progressSignal.progress.emit(f"DL: 70% - Found {len(EOIs)} events")
            _process_detection_results(self, EOIs)
            self.progressSignal.progress.emit("DL: 100% - Complete")

    except Exception as e:
        self.progressSignal.progress.emit(f"DL: Error - {e}")
        print(f'Error during Deep Learning detection/classification: {e}')
        import traceback
        traceback.print_exc()


def ConsensusDetection(self):
    """Run consensus detection combining Hilbert, STE, and MNI."""
    try:
        if not hasattr(self, 'source_filename') or self.source_filename is None or not os.path.exists(self.source_filename):
            self.progressSignal.progress.emit("Consensus: Error - No source file loaded")
            return

        self.progressSignal.progress.emit("Consensus: 10% - Loading data")
        raw_data, Fs = self.settingsWindow.loaded_sources[self.source_filename]

        self.progressSignal.progress.emit("Consensus: 30% - Building parameters")
        # Build parameter dicts
        hilbert_params = {
            'epoch': self.hilbert_epoch,
            'sd_num': self.hilbert_sd_num,
            'min_duration': self.hilbert_min_duration,
            'min_freq': self.hilbert_min_freq,
            'max_freq': self.hilbert_max_freq,
            'required_peak_number': self.hilbert_required_peaks,
            'required_peak_sd': getattr(self, 'hilbert_peak_sd', 2.0),
            'boundary_fraction': 0.3
        }

        ste_params = {
            'threshold': self.ste_threshold,
            'window_size': self.ste_window_size,
            'overlap': self.ste_overlap,
            'min_freq': self.ste_min_freq,
            'max_freq': self.ste_max_freq
        }

        mni_params = {
            'baseline_window': self.mni_baseline_window,
            'threshold_percentile': self.mni_threshold_percentile,
            'min_freq': self.mni_min_freq,
            'max_freq': getattr(self, 'mni_max_freq', 500.0)
        }

        # Run consensus
        from core.Detector import consensus_detect_events
        self.progressSignal.progress.emit("Consensus: 50% - Running detectors")
        EOIs = consensus_detect_events(
            raw_data, Fs,
            hilbert_params=hilbert_params,
            ste_params=ste_params,
            mni_params=mni_params,
            voting_strategy=getattr(self, 'consensus_voting', 'majority'),
            overlap_threshold_ms=getattr(self, 'consensus_overlap_ms', 10.0)
        )

        if EOIs is None or len(EOIs) == 0:
            self.progressSignal.progress.emit("Consensus: Complete - 0 events")
        else:
            self.progressSignal.progress.emit(f"Consensus: 70% - Found {len(EOIs)} events")

        _process_detection_results(self, EOIs)
        self.progressSignal.progress.emit("Consensus: 100% - Complete")

    except Exception as e:
        self.progressSignal.progress.emit(f"Consensus: Error - {e}")
        print(f'Error during Consensus detection: {e}')
        import traceback
        traceback.print_exc()


def _process_detection_results(self, EOIs):
    """Helper to populate the EOI tree with detection results."""
    if EOIs is None or len(EOIs) == 0:
        print('No EOIs were found!')
        return

    self.events_detected.setText(str(len(EOIs)))

    for key, value in self.EOI_headers.items():
        if 'ID' in key:
            ID_value = value
        elif 'Start' in key:
            start_value = value
        elif 'Stop' in key:
            stop_value = value
        elif 'Duration' in key:
            duration_value = value
        elif 'Settings' in key:
            settings_value = value

    for EOI in EOIs:
        EOI_item = TreeWidgetItem()

        new_id = self.createID(self.eoi_method.currentText())
        self.IDs.append(new_id)
        EOI_item.setText(ID_value, new_id)
        EOI_item.setText(start_value, str(EOI[0]))
        EOI_item.setText(stop_value, str(EOI[1]))
        try:
            dur_ms = float(EOI[1]) - float(EOI[0])
            if dur_ms > 0:
                EOI_item.setText(duration_value, f"{dur_ms:.3f}")
            else:
                EOI_item.setText(duration_value, "")
        except Exception:
            EOI_item.setText(duration_value, "")
        EOI_item.setText(settings_value, getattr(self, 'settings_fname', 'N/A'))

        self.AddItemSignal.childAdded.emit(EOI_item)


def _save_generic_settings(self, method_tag, settings_dict):
    """Helper to save settings for generic detectors."""
    try:
        session_path, set_filename = os.path.split(self.mainWindow.current_set_filename)
        session = os.path.splitext(set_filename)[0]
        hfo_path = os.path.join(session_path, 'HFOScores', session)

        if not os.path.exists(hfo_path):
            os.makedirs(hfo_path)

        settings_name = '%s_%s_settings' % (session, method_tag)
        existing_settings_files = [os.path.join(hfo_path, file) for file in os.listdir(hfo_path) if settings_name in file]

        if len(existing_settings_files) >= 1:
            match = False
            for file in existing_settings_files:
                with open(file, 'r+') as f:
                    file_settings = json.load(f)
                    if len(file_settings.items() & settings_dict.items()) == len(file_settings.items()):
                        match = True
                        self.settings_fname = file
                        break
            if not match:
                version = len(existing_settings_files) + 1
                self.settings_fname = os.path.join(hfo_path, '%s_%d.txt' % (settings_name, version))
                with open(self.settings_fname, 'w') as f:
                    json.dump(settings_dict, f)
        else:
            self.settings_fname = os.path.join(hfo_path, '%s.txt' % (settings_name))
            with open(self.settings_fname, 'w') as f:
                json.dump(settings_dict, f)
    except Exception as e:
        print(f"Could not save settings: {e}")
        self.settings_fname = 'N/A'


def _ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def _session_paths_for_settings(self):
    session_path, set_filename = os.path.split(self.mainWindow.current_set_filename)
    session = os.path.splitext(set_filename)[0]
    hfo_path = os.path.join(session_path, 'HFOScores', session)
    _ensure_dir(hfo_path)
    return session, hfo_path


def _pyhfo_settings_filename(self):
    session, hfo_path = _session_paths_for_settings(self)
    settings_name = f"{session}_{self.id_abbreviations['pyhfo']}_settings"
    return os.path.join(hfo_path, f"{settings_name}.txt")


def _prepare_pyhfo_settings(self):
    """Record the fact that pyHFO auto settings were used, for traceability in UI."""
    try:
        import json
        fname = _pyhfo_settings_filename(self)
        settings = {
            'Detector': 'pyHFO',
            'use_artifact_model': True,
            'use_spike_model': True,
            'use_epileptogenic_model': True,
        }
        with open(fname, 'w') as f:
            json.dump(settings, f)
        self.settings_fname = fname
    except Exception:
        self.settings_fname = 'N/A'


def find_same_consec(data):
    if len(data) == 1:
        return [0]

    diff_data = np.diff(data)

    consec_switch = np.where(diff_data != 0)[0]

    if len(consec_switch) == 0:
        # then it never switched
        return [np.arange(len(data))]

    consecutive_values = []

    if consec_switch[0] > 0:
        consecutive_values.append(np.arange(consec_switch[0] + 1))
    else:
        consecutive_values.append([0])
        if np.sum(np.in1d(consec_switch, [0])) == 0:
            consecutive_values.append(np.arange(1, consec_switch[1] + 1))

    for i in range(1, len(consec_switch)):
        consecutive_values.append(np.arange(consec_switch[i - 1] + 1, consec_switch[i] + 1))

    if consec_switch[-1] != len(data) - 1:
        consecutive_values.append(np.arange(consec_switch[-1] + 1, len(data)))

    return consecutive_values


def RejectEOIs(EOIs, rectified_signal, Fs, threshold, required_peaks):
    # reject events that don't have the required_peaks above the designated threshold, if there is no threshold and
    # you just want an N number of peaks, then leave the threshold as None (or blank in the GUI)
    rejected_eois = []

    for k in range(EOIs.shape[0]):

        eoi_data = rectified_signal[int(Fs * EOIs[k, 0] / 1000):int(Fs * EOIs[k, 1] / 1000) + 1]

        peak_indices = detect_peaks(eoi_data, threshold=0)

        if threshold is not None:

            if not len(np.where(eoi_data[peak_indices] >= threshold)[0]) >= required_peaks:
                rejected_eois.append(k)

        else:

            if not len(peak_indices) >= required_peaks:
                rejected_eois.append(k)

    window_EOIs = np.delete(EOIs, rejected_eois, axis=0)  # removing rejected EOIs

    return window_EOIs


def findStop(stop_indices):
    # checks which indices are consecutive from the first index given and finds that largest consecutive value to be
    # the stop index

    stop_index = stop_indices[0]

    if len(stop_indices) == 1:
        return stop_index

    for i in range(1, len(stop_indices) + 1):
        if stop_indices[i] == stop_index + 1:
            stop_index = stop_indices[i]
        else:
            break
    return stop_index


def findStart(start_indices):
    # checks which indices are consecutive from the last index given and finds that smallest consecutive value to be
    # the start index

    start_index = start_indices[-1]

    if len(start_indices) == 1:
        return start_index

    for i in range(len(start_indices)-2, -1, -1):
        if start_indices[i] == start_index - 1:
            start_index = start_indices[i]
        else:
            break
    return start_index


class HilbertParametersWindow(QtWidgets.QWidget):

    def __init__(self, main, score):
        super(HilbertParametersWindow, self).__init__()


        self.mainWindow = main
        self.scoreWindow = score

        # background(self)
        # width = self.deskW / 6
        # height = self.deskH / 1.5

        self.setWindowTitle(
            os.path.splitext(os.path.basename(__file__))[0] + " - Hilbert Parameters Window")  # sets the title of the window

        # main_location = main.frameGeometry().getCoords()
        # self.setGeometry(main_location[2], main_location[1] + 30, width, height)

        self.HilbertParameters = ['Epoch(s):', '', 'Threshold(SD):',  '', 'Minimum Time(ms):', '',
                             'Min Frequency(Hz):', '', 'Max Frequency(Hz):', '', 'Required Peaks:', '',
                                  'Required Peak Threshold(SD):', '', 'Boundary Threshold(Percent)', '', '', '']

        self.hilbert_fields = {}
        self.Hilbert_field_positions = {}

        positions = [(i, j) for i in range(3) for j in range(6)]
        hilbert_parameter_layout = QtWidgets.QGridLayout()

        for (i, j), parameter in zip(positions, self.HilbertParameters):

            if parameter == '':
                continue
            else:
                self.Hilbert_field_positions[parameter] = (i, j)

                self.hilbert_fields[i, j] = QtWidgets.QLabel(parameter)
                self.hilbert_fields[i, j].setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

                self.hilbert_fields[i, j + 1] = QtWidgets.QLineEdit()
                self.hilbert_fields[i, j + 1].setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)

                if 'Epoch' in parameter:
                    ParameterText = str(5*60)  # 5 minute epochs
                    self.hilbert_fields[i, j + 1].setToolTip(
                        'The data is broken up into epochs of this time period, there is a different threshold per epoch.')
                elif 'Threshold' in parameter and not any(x in parameter for x in ['Required', 'Boundary']):
                    ParameterText = '3'  # mean + 3 SD's
                    self.hilbert_fields[i, j + 1].setToolTip(
                        'The threshold is set to the mean of the epoch + X standard deviations of that epoch')
                elif 'Minimum' in parameter:
                    ParameterText = '10'  # minimum time of 6 ms'
                    self.hilbert_fields[i, j + 1].setToolTip(
                        'The minimum duration that an EOI must have in order to not be discarded')
                elif 'Min Freq' in parameter:
                    ParameterText = '80'  # minimum frequency of 100 Hz
                    self.hilbert_fields[i, j + 1].setToolTip(
                        'The minimum frequency of the filtered signal that then undergoes the Hilbert transform to find the EOIs.')
                elif 'Max Freq' in parameter:
                    ParameterText = '500'  # minimum frequency of 100 Hz
                    self.hilbert_fields[i, j + 1].setToolTip(
                        'The maximum frequency of the filtered signal that then undergoes the Hilbert transform to find the EOIs.')
                elif 'Required Peaks' in parameter:
                    ParameterText = '6'
                    self.hilbert_fields[i, j + 1].setToolTip(
                        'The required peaks (above the required peak threshold) of the recitfied signal that the EOI must have to not get discarded.')
                elif 'Required Peak Threshold' in parameter:
                    ParameterText = '2'
                    self.hilbert_fields[i, j + 1].setToolTip(
                        'The threshold for the the required peaks (mean + X standard deviations).')

                elif 'Boundary Threshold(Percent)' in parameter:
                    ParameterText = '30'
                    self.hilbert_fields[i, j + 1].setToolTip(
                        'The percentage of the threshold that will be used to determine the beginning and end of the EOI.')

                self.hilbert_fields[i, j + 1].setText(ParameterText)

                parameter_layout = QtWidgets.QHBoxLayout()
                parameter_layout.addWidget(self.hilbert_fields[i, j])
                parameter_layout.addWidget(self.hilbert_fields[i, j + 1])
                hilbert_parameter_layout.addLayout(parameter_layout, *(i, j))

        # Load saved parameters after all fields are created
        self._load_hilbert_params()

        window_layout = QtWidgets.QVBoxLayout()

        Title = QtWidgets.QLabel("Automatic Detection - Hilbert")

        directions = QtWidgets.QLabel("Please ensure that the parameters listed below are correct. " +
                                  "if you are interested in Fast Ripples, I recommend bumping the " +
                                  "minimum frequency to 500Hz.")

        self.analyze_btn = QtWidgets.QPushButton("Analyze")
        self.analyze_btn.clicked.connect(self.analyze)

        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.close_app)

        self.reset_btn = QtWidgets.QPushButton("Reset to Defaults")
        self.reset_btn.clicked.connect(self.reset_to_defaults)

        button_layout = QtWidgets.QHBoxLayout()

        for button in [self.analyze_btn, self.reset_btn, self.cancel_btn]:
            button_layout.addWidget(button)

        for order in [Title, directions, hilbert_parameter_layout, button_layout]:
            if 'Layout' in order.__str__():
                window_layout.addLayout(order)
                window_layout.addStretch(1)
            else:
                window_layout.addWidget(order, 0, QtCore.Qt.AlignCenter)
                window_layout.addStretch(1)

        self.setLayout(window_layout)

        center(self)

        self.show()

    def analyze(self):

        if not hasattr(self.scoreWindow, 'source_filename'):
            print('You have not chosen a source yet! Please add a source in the Graph Settings window!')
            return

        if not os.path.exists(self.scoreWindow.source_filename):
            return

        # Save current parameters before analysis
        self._save_hilbert_params()

        for parameter, (i, j) in self.Hilbert_field_positions.items():

            if parameter == '':
                continue

            if 'Min Freq' in parameter:
                min_freq_i, min_freq_j = (i, j)
            elif 'Max Freq' in parameter:
                max_freq_i, max_freq_j = (i, j)

        if '.egf' in self.scoreWindow.source_filename:
            # ParameterText = '500'  # maximum frequency of 400 Hz
            self.hilbert_fields[max_freq_i, max_freq_j + 1].setText('500')
        else:
            # ParameterText = '125'
            self.hilbert_fields[max_freq_i, max_freq_j + 1].setText('125')

        settings = {}
        for parameter, (i, j) in self.Hilbert_field_positions.items():

            if parameter == '':
                continue

            parameter_object = self.hilbert_fields[i, j+1]
            try:
                if 'Epoch' in parameter:
                    self.scoreWindow.epoch = float(parameter_object.text())
                    settings[parameter] = self.scoreWindow.epoch
                elif 'Threshold' in parameter and not any(x in parameter for x in ['Required', 'Boundary']):
                    self.scoreWindow.sd_num = float(parameter_object.text())
                    settings[parameter] = self.scoreWindow.sd_num
                elif 'Minimum' in parameter:
                    self.scoreWindow.min_duration = float(parameter_object.text())
                    settings[parameter] = self.scoreWindow.min_duration
                elif 'Min Freq' in parameter:
                    self.scoreWindow.min_freq = float(parameter_object.text())
                    settings[parameter] = self.scoreWindow.min_freq
                elif 'Max Freq' in parameter:
                    self.scoreWindow.max_freq = float(parameter_object.text())
                    settings[parameter] = self.scoreWindow.max_freq
                elif 'Required Peaks' in parameter:
                    self.scoreWindow.required_peak_number = float(parameter_object.text())
                    settings[parameter] = self.scoreWindow.required_peak_number
                elif 'Required Peak Threshold' in parameter:
                    value = parameter_object.text()
                    if value == '':
                        self.scoreWindow.required_peak_sd = None
                        settings[parameter] = value
                    else:
                        self.scoreWindow.required_peak_sd = float(value)
                        settings[parameter] = self.scoreWindow.required_peak_sd
                elif 'Boundary Threshold(Percent)' in parameter:
                    self.scoreWindow.boundary_fraction = float(parameter_object.text())/100
                    settings[parameter] = self.scoreWindow.boundary_fraction
            except ValueError:

                self.mainWindow.choice = ''
                self.mainWindow.ErrorDialogue.myGUI_signal.emit("InvalidDetectionParam")

                while self.mainWindow.choice == '':
                    time.sleep(0.1)

                return

        # save the EOI parameters
        # find any settings fnames
        method_abbreviation = self.scoreWindow.id_abbreviations['hilbert']
        session_path, set_filename = os.path.split(self.mainWindow.current_set_filename)
        session = os.path.splitext(set_filename)[0]

        hfo_path = os.path.join(session_path, 'HFOScores', session)

        if not os.path.exists(hfo_path):
            os.makedirs(hfo_path)

        settings_name = '%s_%s_settings' % (session, method_abbreviation)

        existing_settings_files = [os.path.join(hfo_path, file) for file in os.listdir(hfo_path) if settings_name in file]

        if len(existing_settings_files) >= 1:

            # check if any of these files has your settings
            match = False
            for file in existing_settings_files:
                with open(file, 'r+') as f:
                    file_settings = json.load(f)
                    if len(file_settings.items() & settings.items()) == len(file_settings.items()):
                        match = True
                        self.scoreWindow.settings_fname = file
                        break

            if not match:
                version = [int(os.path.splitext(file)[0].split('_')[-1]) for file in existing_settings_files if
                           os.path.splitext(file)[0].split('_')[-1] != 'settings']
                if len(version) == 0:
                    version = 1
                else:
                    version = np.amax(np.asarray(version)) + 1

                self.scoreWindow.settings_fname = os.path.join(hfo_path, '%s_%d.txt' % (settings_name, version))
                with open(self.scoreWindow.settings_fname, 'w') as f:
                    json.dump(settings, f)

        else:
            # no settings file for this session
            self.scoreWindow.settings_fname = os.path.join(hfo_path, '%s.txt' % (settings_name))
            with open(self.scoreWindow.settings_fname, 'w') as f:
                json.dump(settings, f)

        # Check if thread is already running, stop it first
        if self.scoreWindow.hilbert_thread.isRunning():
            self.scoreWindow.hilbert_thread.quit()
            self.scoreWindow.hilbert_thread.wait()
        
        self.scoreWindow.hilbert_thread.start()
        self.scoreWindow.hilbert_thread_worker = Worker(HilbertDetection, self.scoreWindow)
        self.scoreWindow.hilbert_thread_worker.moveToThread(self.scoreWindow.hilbert_thread)
        self.scoreWindow.hilbert_thread_worker.start.emit("start")

        self.close()

    def reset_to_defaults(self):
        """Reset all Hilbert parameters to their default values"""
        default_values = {
            'Epoch(s):': str(5*60),
            'Threshold(SD):': '3',
            'Minimum Time(ms):': '10',
            'Min Frequency(Hz):': '80',
            'Max Frequency(Hz):': '500',
            'Required Peaks:': '6',
            'Required Peak Threshold(SD):': '2',
            'Boundary Threshold(Percent)': '30'
        }
        
        for parameter, value in default_values.items():
            if parameter in self.Hilbert_field_positions:
                i, j = self.Hilbert_field_positions[parameter]
                self.hilbert_fields[i, j + 1].setText(value)
        
        # Save these defaults to persistent storage
        self._save_hilbert_params()

    def _save_hilbert_params(self):
        """Save current Hilbert parameters to persistent storage"""
        import json
        try:
            settings_file = os.path.join(self.scoreWindow.mainWindow.SETTINGS_DIR, 'hilbert_params.json')
            params = {}
            for parameter, (i, j) in self.Hilbert_field_positions.items():
                if parameter != '':
                    params[parameter] = self.hilbert_fields[i, j + 1].text()
            with open(settings_file, 'w') as f:
                json.dump(params, f)
        except Exception:
            pass

    def _load_hilbert_params(self):
        """Load saved Hilbert parameters from persistent storage"""
        import json
        try:
            settings_file = os.path.join(self.scoreWindow.mainWindow.SETTINGS_DIR, 'hilbert_params.json')
            if os.path.exists(settings_file):
                with open(settings_file, 'r') as f:
                    params = json.load(f)
                    for parameter, value in params.items():
                        if parameter in self.Hilbert_field_positions:
                            i, j = self.Hilbert_field_positions[parameter]
                            self.hilbert_fields[i, j + 1].setText(value)
        except Exception:
            pass

    def close_app(self):

        self.close()


class PyHFOParametersWindow(QtWidgets.QWidget):

    def __init__(self, main, score):
        super(PyHFOParametersWindow, self).__init__()

        self.mainWindow = main
        self.scoreWindow = score

        self.setWindowTitle("pyHFO - Automatic Detection Parameters")

        # Default frequency range (user can adjust manually)
        self.default_min_freq = 80
        self.default_max_freq = 500

        # Define parameters for pyHFO
        self.PyHFOParameters = [
            'Epoch(s):', '', 'Threshold(SD):', '', 'Minimum Duration(ms):', '',
            'Min Frequency(Hz):', '', 'Max Frequency(Hz):', '', 'CPU Cores:', '', '', ''
        ]

        self.pyhfo_fields = {}
        self.PyHFO_field_positions = {}

        positions = [(i, j) for i in range(2) for j in range(6)]
        pyhfo_parameter_layout = QtWidgets.QGridLayout()

        for (i, j), parameter in zip(positions, self.PyHFOParameters):
            if parameter == '':
                continue
            else:
                self.PyHFO_field_positions[parameter] = (i, j)

                self.pyhfo_fields[i, j] = QtWidgets.QLabel(parameter)
                self.pyhfo_fields[i, j].setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

                self.pyhfo_fields[i, j + 1] = QtWidgets.QLineEdit()
                self.pyhfo_fields[i, j + 1].setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)

                if 'Epoch' in parameter:
                    ParameterText = str(10*60)  # 10 minute epochs (longer = faster)
                    self.pyhfo_fields[i, j + 1].setToolTip(
                        'Data is divided into epochs for threshold calculation. Longer epochs = faster processing.')
                elif 'Threshold' in parameter:
                    ParameterText = '5'  # mean + 5 SD's
                    self.pyhfo_fields[i, j + 1].setToolTip(
                        'Amplitude threshold: mean + X standard deviations per epoch.')
                elif 'Minimum Duration' in parameter:
                    ParameterText = '10'  # 10 ms minimum
                    self.pyhfo_fields[i, j + 1].setToolTip(
                        'Minimum event duration in milliseconds (rejects shorter events).')
                elif 'Min Freq' in parameter:
                    ParameterText = str(self.default_min_freq)
                    self.pyhfo_fields[i, j + 1].setToolTip(
                        'Lower cutoff of bandpass filter (Hz).')
                elif 'Max Freq' in parameter:
                    ParameterText = str(self.default_max_freq)
                    self.pyhfo_fields[i, j + 1].setToolTip(
                        'Upper cutoff of bandpass filter (Hz).')
                elif 'CPU Cores' in parameter:
                    import multiprocessing
                    default_cores = min(4, multiprocessing.cpu_count() // 2)
                    ParameterText = str(default_cores)
                    self.pyhfo_fields[i, j + 1].setToolTip(
                        f'Number of CPU cores for parallel processing (max: {multiprocessing.cpu_count()}).')

                self.pyhfo_fields[i, j + 1].setText(ParameterText)

                parameter_layout = QtWidgets.QHBoxLayout()
                parameter_layout.addWidget(self.pyhfo_fields[i, j])
                parameter_layout.addWidget(self.pyhfo_fields[i, j + 1])
                pyhfo_parameter_layout.addLayout(parameter_layout, *(i, j))

        # Load saved parameters after all fields are created
        self._load_pyhfo_params()

        window_layout = QtWidgets.QVBoxLayout()

        Title = QtWidgets.QLabel("Automatic Detection - pyHFO (Adjust frequency range for ripples/fast ripples)")

        directions = QtWidgets.QLabel(
            "Adjust parameters below. Higher CPU cores = faster processing.\n"
            "Longer epochs reduce threshold recalculations and speed up detection."
        )

        self.analyze_btn = QtWidgets.QPushButton("Analyze")
        self.analyze_btn.clicked.connect(self.analyze)

        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.close_app)

        self.reset_btn = QtWidgets.QPushButton("Reset to Defaults")
        self.reset_btn.clicked.connect(self.reset_to_defaults)

        button_layout = QtWidgets.QHBoxLayout()

        for button in [self.analyze_btn, self.reset_btn, self.cancel_btn]:
            button_layout.addWidget(button)

        for order in [Title, directions, pyhfo_parameter_layout, button_layout]:
            if 'Layout' in order.__str__():
                window_layout.addLayout(order)
                window_layout.addStretch(1)
            else:
                window_layout.addWidget(order, 0, QtCore.Qt.AlignCenter)
                window_layout.addStretch(1)

        self.setLayout(window_layout)

        center(self)

        self.show()

    def analyze(self):
        if not hasattr(self.scoreWindow, 'source_filename'):
            print('You have not chosen a source yet! Please add a source in the Graph Settings window!')
            return

        if not os.path.exists(self.scoreWindow.source_filename):
            return

        # Save current parameters before analysis
        self._save_pyhfo_params()

        # Extract parameters
        settings = {}
        for parameter, (i, j) in self.PyHFO_field_positions.items():
            if parameter == '':
                continue

            parameter_object = self.pyhfo_fields[i, j+1]
            try:
                if 'Epoch' in parameter:
                    self.scoreWindow.pyhfo_epoch = float(parameter_object.text())
                    settings[parameter] = self.scoreWindow.pyhfo_epoch
                elif 'Threshold' in parameter:
                    self.scoreWindow.pyhfo_sd_num = float(parameter_object.text())
                    settings[parameter] = self.scoreWindow.pyhfo_sd_num
                elif 'Minimum Duration' in parameter:
                    self.scoreWindow.pyhfo_min_duration = float(parameter_object.text()) / 1000.0  # Convert ms to seconds
                    settings[parameter] = parameter_object.text()
                elif 'Min Freq' in parameter:
                    self.scoreWindow.pyhfo_min_freq = float(parameter_object.text())
                    settings[parameter] = self.scoreWindow.pyhfo_min_freq
                elif 'Max Freq' in parameter:
                    self.scoreWindow.pyhfo_max_freq = float(parameter_object.text())
                    settings[parameter] = self.scoreWindow.pyhfo_max_freq
                elif 'CPU Cores' in parameter:
                    self.scoreWindow.pyhfo_n_jobs = int(parameter_object.text())
                    settings[parameter] = self.scoreWindow.pyhfo_n_jobs
            except ValueError:
                self.mainWindow.choice = ''
                self.mainWindow.ErrorDialogue.myGUI_signal.emit("InvalidDetectionParam")
                while self.mainWindow.choice == '':
                    time.sleep(0.1)
                return

        # Save settings to file
        self._save_settings_file(settings)

        # Use single pyHFO thread
        thread = self.scoreWindow.pyhfo_thread

        # Check if thread is already running, stop it first
        if thread.isRunning():
            thread.quit()
            thread.wait()

        thread.start()
        self.scoreWindow.pyhfo_thread_worker = Worker(PyHFODetection, self.scoreWindow)
        self.scoreWindow.pyhfo_thread_worker.moveToThread(thread)
        self.scoreWindow.pyhfo_thread_worker.start.emit("start")

        self.close()

    def _save_settings_file(self, settings):
        """Save settings to a file for traceability."""
        try:
            method_abbreviation = self.scoreWindow.id_abbreviations[self.scoreWindow.eoi_method.currentText().lower()]
            session_path, set_filename = os.path.split(self.mainWindow.current_set_filename)
            session = os.path.splitext(set_filename)[0]

            hfo_path = os.path.join(session_path, 'HFOScores', session)

            if not os.path.exists(hfo_path):
                os.makedirs(hfo_path)

            settings_name = '%s_%s_settings' % (session, method_abbreviation)

            existing_settings_files = [os.path.join(hfo_path, file) for file in os.listdir(hfo_path) if settings_name in file]

            if len(existing_settings_files) >= 1:
                # check if any of these files has your settings
                match = False
                for file in existing_settings_files:
                    with open(file, 'r+') as f:
                        file_settings = json.load(f)
                        if len(file_settings.items() & settings.items()) == len(file_settings.items()):
                            match = True
                            self.scoreWindow.settings_fname = file
                            break

                if not match:
                    version = [int(os.path.splitext(file)[0].split('_')[-1]) for file in existing_settings_files if
                               os.path.splitext(file)[0].split('_')[-1] != 'settings']
                    if len(version) == 0:
                        version = 1
                    else:
                        version = np.amax(np.asarray(version)) + 1

                    self.scoreWindow.settings_fname = os.path.join(hfo_path, '%s_%d.txt' % (settings_name, version))
                    with open(self.scoreWindow.settings_fname, 'w') as f:
                        json.dump(settings, f)

            else:
                # no settings file for this session
                self.scoreWindow.settings_fname = os.path.join(hfo_path, '%s.txt' % (settings_name))
                with open(self.scoreWindow.settings_fname, 'w') as f:
                    json.dump(settings, f)
        except Exception as e:
            print(f"Could not save settings: {e}")
            self.scoreWindow.settings_fname = 'N/A'

    def reset_to_defaults(self):
        """Reset all pyHFO parameters to their default values"""
        import multiprocessing
        default_cores = min(4, multiprocessing.cpu_count() // 2)
        
        default_values = {
            'Epoch(s):': str(10*60),
            'Threshold(SD):': '5',
            'Minimum Duration(ms):': '10',
            'Min Frequency(Hz):': str(self.default_min_freq),
            'Max Frequency(Hz):': str(self.default_max_freq),
            'CPU Cores:': str(default_cores)
        }
        
        for parameter, value in default_values.items():
            if parameter in self.PyHFO_field_positions:
                i, j = self.PyHFO_field_positions[parameter]
                self.pyhfo_fields[i, j + 1].setText(value)
        
        # Save these defaults to persistent storage
        self._save_pyhfo_params()

    def _save_pyhfo_params(self):
        """Save current pyHFO parameters to persistent storage"""
        try:
            settings_file = os.path.join(self.scoreWindow.mainWindow.SETTINGS_DIR, f'pyhfo_{self.detection_type}_params.json')
            params = {}
            for parameter, (i, j) in self.PyHFO_field_positions.items():
                if parameter != '':
                    params[parameter] = self.pyhfo_fields[i, j + 1].text()
            with open(settings_file, 'w') as f:
                json.dump(params, f)
        except Exception:
            pass

    def _load_pyhfo_params(self):
        """Load saved pyHFO parameters from persistent storage"""
        try:
            settings_file = os.path.join(self.scoreWindow.mainWindow.SETTINGS_DIR, f'pyhfo_{self.detection_type}_params.json')
            if os.path.exists(settings_file):
                with open(settings_file, 'r') as f:
                    params = json.load(f)
                    for parameter, value in params.items():
                        if parameter in self.PyHFO_field_positions:
                            i, j = self.PyHFO_field_positions[parameter]
                            self.pyhfo_fields[i, j + 1].setText(value)
        except Exception:
            pass

    def close_app(self):
        self.close()


class STEParametersWindow(QtWidgets.QWidget):
    def __init__(self, main, score):
        super(STEParametersWindow, self).__init__()
        self.mainWindow = main
        self.scoreWindow = score
        self.setWindowTitle("STE Detection Parameters")

        layout = QtWidgets.QFormLayout()

        params = self._load_ste_params()
        self.threshold_edit = QtWidgets.QLineEdit(str(params.get('threshold', 3.0)))
        self.window_size_edit = QtWidgets.QLineEdit(str(params.get('window_size', 0.01)))
        self.overlap_edit = QtWidgets.QLineEdit(str(params.get('overlap', 0.5)))
        self.min_freq_edit = QtWidgets.QLineEdit(str(params.get('min_freq', 80.0)))
        self.max_freq_edit = QtWidgets.QLineEdit(str(params.get('max_freq', 500.0)))

        layout.addRow("Threshold (RMS):", self.threshold_edit)
        layout.addRow("Window Size (s):", self.window_size_edit)
        layout.addRow("Overlap (0-1):", self.overlap_edit)
        layout.addRow("Min Frequency (Hz):", self.min_freq_edit)
        layout.addRow("Max Frequency (Hz):", self.max_freq_edit)

        self.analyze_btn = QtWidgets.QPushButton("Analyze")
        self.analyze_btn.clicked.connect(self.analyze)
        layout.addRow(self.analyze_btn)

        self.setLayout(layout)
        center(self)
        self.show()

    def _load_ste_params(self):
        import json
        try:
            fname = os.path.join(self.scoreWindow.mainWindow.SETTINGS_DIR, 'ste_params.json')
            if os.path.exists(fname):
                with open(fname, 'r') as f:
                    return json.load(f)
        except Exception:
            return {}
        return {}

    def analyze(self):
        try:
            self.scoreWindow.ste_threshold = float(self.threshold_edit.text())
            self.scoreWindow.ste_window_size = float(self.window_size_edit.text())
            self.scoreWindow.ste_overlap = float(self.overlap_edit.text())
            self.scoreWindow.ste_min_freq = float(self.min_freq_edit.text())
            self.scoreWindow.ste_max_freq = float(self.max_freq_edit.text())
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Invalid Input", "Please enter valid numeric values.")
            return

        settings = {
            'threshold': self.scoreWindow.ste_threshold,
            'window_size': self.scoreWindow.ste_window_size,
            'overlap': self.scoreWindow.ste_overlap,
            'min_freq': self.scoreWindow.ste_min_freq,
            'max_freq': self.scoreWindow.ste_max_freq
        }
        _save_generic_settings(self.scoreWindow, 'STE', settings)

        if self.scoreWindow.ste_thread.isRunning():
            self.scoreWindow.ste_thread.quit()
            self.scoreWindow.ste_thread.wait()

        self.scoreWindow.ste_thread.start()
        self.scoreWindow.ste_thread_worker = Worker(STEDetection, self.scoreWindow)
        self.scoreWindow.ste_thread_worker.moveToThread(self.scoreWindow.ste_thread)
        self.scoreWindow.ste_thread_worker.start.emit("start")
        self.close()


class MNIParametersWindow(QtWidgets.QWidget):
    def __init__(self, main, score):
        super(MNIParametersWindow, self).__init__()
        self.mainWindow = main
        self.scoreWindow = score
        self.setWindowTitle("MNI Detection Parameters")

        layout = QtWidgets.QFormLayout()

        params = self._load_mni_params()
        self.baseline_edit = QtWidgets.QLineEdit(str(params.get('baseline_window', 10.0)))
        self.percentile_edit = QtWidgets.QLineEdit(str(params.get('threshold_percentile', 99.0)))
        self.min_freq_edit = QtWidgets.QLineEdit(str(params.get('min_freq', 80.0)))

        layout.addRow("Baseline Window (s):", self.baseline_edit)
        layout.addRow("Threshold Percentile:", self.percentile_edit)
        layout.addRow("Min Frequency (Hz):", self.min_freq_edit)

        self.analyze_btn = QtWidgets.QPushButton("Analyze")
        self.analyze_btn.clicked.connect(self.analyze)
        layout.addRow(self.analyze_btn)

        self.setLayout(layout)
        center(self)
        self.show()

    def _load_mni_params(self):
        import json
        try:
            fname = os.path.join(self.scoreWindow.mainWindow.SETTINGS_DIR, 'mni_params.json')
            if os.path.exists(fname):
                with open(fname, 'r') as f:
                    return json.load(f)
        except Exception:
            return {}
        return {}

    def analyze(self):
        try:
            self.scoreWindow.mni_baseline_window = float(self.baseline_edit.text())
            self.scoreWindow.mni_threshold_percentile = float(self.percentile_edit.text())
            self.scoreWindow.mni_min_freq = float(self.min_freq_edit.text())
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Invalid Input", "Please enter valid numeric values.")
            return

        settings = {
            'baseline_window': self.scoreWindow.mni_baseline_window,
            'threshold_percentile': self.scoreWindow.mni_threshold_percentile,
            'min_freq': self.scoreWindow.mni_min_freq
        }
        _save_generic_settings(self.scoreWindow, 'MNI', settings)

        if self.scoreWindow.mni_thread.isRunning():
            self.scoreWindow.mni_thread.quit()
            self.scoreWindow.mni_thread.wait()

        self.scoreWindow.mni_thread.start()
        self.scoreWindow.mni_thread_worker = Worker(MNIDetection, self.scoreWindow)
        self.scoreWindow.mni_thread_worker.moveToThread(self.scoreWindow.mni_thread)
        self.scoreWindow.mni_thread_worker.start.emit("start")
        self.close()


class DLParametersWindow(QtWidgets.QWidget):
    def __init__(self, main, score):
        super(DLParametersWindow, self).__init__()
        self.mainWindow = main
        self.scoreWindow = score
        self.setWindowTitle("Deep Learning Detection Parameters")

        layout = QtWidgets.QFormLayout()

        self.model_path_edit = QtWidgets.QLineEdit()
        self.browse_btn = QtWidgets.QPushButton("Browse...")
        self.browse_btn.clicked.connect(self.browse_model)
        
        path_layout = QtWidgets.QHBoxLayout()
        path_layout.addWidget(self.model_path_edit)
        path_layout.addWidget(self.browse_btn)

        self.threshold_edit = QtWidgets.QLineEdit("0.75")
        self.batch_size_edit = QtWidgets.QLineEdit("32")
        self.window_size_edit = QtWidgets.QLineEdit("1.0")
        self.overlap_edit = QtWidgets.QLineEdit("0.5")
        
        self.cwt_check = QtWidgets.QCheckBox("Use CWT (Scalogram) Preprocessing")
        self.cwt_check.setToolTip("Enable if using the 2D CNN model trained on CWT Scalograms (model type: HFO_2D_CNN)")

        layout.addRow("Model Path:", path_layout)
        layout.addRow("Threshold (Prob):", self.threshold_edit)
        layout.addRow("Batch Size:", self.batch_size_edit)
        layout.addRow("Window Size (s):", self.window_size_edit)
        layout.addRow("Overlap (0-1):", self.overlap_edit)
        layout.addRow("", self.cwt_check)

        self.analyze_btn = QtWidgets.QPushButton("Classify EOIs")
        self.analyze_btn.clicked.connect(self.analyze)
        layout.addRow(self.analyze_btn)

        self.setLayout(layout)
        center(self)
        self.show()

    def browse_model(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Model File", "", "Model Files (*.h5 *.pt *.onnx *.pkl);;All Files (*)")
        if fname:
            self.model_path_edit.setText(fname)

    def analyze(self):
        model_path = self.model_path_edit.text()
        if not model_path or not os.path.exists(model_path):
            QtWidgets.QMessageBox.warning(self, "Invalid Model", "Please select a valid model file.")
            return

        try:
            self.scoreWindow.dl_model_path = model_path
            self.scoreWindow.dl_threshold = float(self.threshold_edit.text())
            self.scoreWindow.dl_batch_size = int(self.batch_size_edit.text())
            self.scoreWindow.dl_window_size = float(self.window_size_edit.text())
            self.scoreWindow.dl_overlap = float(self.overlap_edit.text())
            self.scoreWindow.dl_use_cwt = self.cwt_check.isChecked()
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Invalid Input", "Please enter valid numeric values.")
            return

        settings = {
            'model_path': self.scoreWindow.dl_model_path,
            'threshold': self.scoreWindow.dl_threshold,
            'batch_size': self.scoreWindow.dl_batch_size,
            'window_size': self.scoreWindow.dl_window_size,
            'overlap': self.scoreWindow.dl_overlap,
            'use_cwt': self.scoreWindow.dl_use_cwt
        }
        _save_generic_settings(self.scoreWindow, 'DL', settings)

        if self.scoreWindow.dl_thread.isRunning():
            self.scoreWindow.dl_thread.quit()
            self.scoreWindow.dl_thread.wait()

        self.scoreWindow.dl_thread.start()
        self.scoreWindow.dl_thread_worker = Worker(DLDetection, self.scoreWindow)
        self.scoreWindow.dl_thread_worker.moveToThread(self.scoreWindow.dl_thread)
        self.scoreWindow.dl_thread_worker.start.emit("start")
        self.close()


class ConsensusParametersWindow(QtWidgets.QWidget):
    """Parameters window for Consensus detection (combines Hilbert, STE, MNI)."""
    
    def __init__(self, main, score):
        super(ConsensusParametersWindow, self).__init__()
        self.mainWindow = main
        self.scoreWindow = score
        self.setWindowTitle("Consensus Detection Parameters (Hilbert + STE + MNI)")

        layout = QtWidgets.QVBoxLayout()

        # Title and help text
        title = QtWidgets.QLabel("Consensus Detection: Vote-based combination of 3 detectors")
        title.setStyleSheet("font-weight: bold;")
        layout.addWidget(title)

        help_text = QtWidgets.QLabel(
            "Consensus voting: Select how many detectors must agree.\n"
            "• Strict (3/3): Highest specificity, may miss marginal HFOs\n"
            "• Majority (2/3): Balanced; recommended for most applications\n"
            "• Any (1/3): Highest sensitivity, more false positives"
        )
        help_text.setWordWrap(True)
        layout.addWidget(help_text)

        # Voting strategy
        voting_layout = QtWidgets.QHBoxLayout()
        voting_label = QtWidgets.QLabel("Voting Strategy:")
        self.voting_combo = QtWidgets.QComboBox()
        self.voting_combo.addItems(['Majority (2/3)', 'Strict (3/3)', 'Any (1/3)'])
        voting_layout.addWidget(voting_label)
        voting_layout.addWidget(self.voting_combo)
        layout.addLayout(voting_layout)

        params = self._load_consensus_params()

        # Overlap threshold
        overlap_layout = QtWidgets.QHBoxLayout()
        overlap_label = QtWidgets.QLabel("Overlap Threshold (ms):")
        self.overlap_edit = QtWidgets.QLineEdit(str(params.get('overlap_ms', 10.0)))
        self.overlap_edit.setMaximumWidth(100)
        overlap_layout.addWidget(overlap_label)
        overlap_layout.addWidget(self.overlap_edit)
        overlap_layout.addStretch()
        layout.addLayout(overlap_layout)

        # Separator
        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.HLine)
        layout.addWidget(separator)

        # Hilbert parameters
        hilbert_group = QtWidgets.QGroupBox("Hilbert Parameters")
        hilbert_form = QtWidgets.QFormLayout()
        self.hilbert_epoch = QtWidgets.QLineEdit(str(params.get('hilbert_epoch', 300.0)))
        self.hilbert_sd = QtWidgets.QLineEdit(str(params.get('hilbert_sd_num', 3.5)))
        self.hilbert_min_dur = QtWidgets.QLineEdit(str(params.get('hilbert_min_duration', 10.0)))
        self.hilbert_min_freq = QtWidgets.QLineEdit(str(params.get('hilbert_min_freq', 80.0)))
        self.hilbert_max_freq = QtWidgets.QLineEdit(str(params.get('hilbert_max_freq', 500.0)))
        self.hilbert_peaks = QtWidgets.QLineEdit(str(params.get('hilbert_required_peaks', 6)))
        self.hilbert_peak_sd = QtWidgets.QLineEdit(str(params.get('hilbert_peak_sd', 2.0)))
        hilbert_form.addRow("Epoch (s):", self.hilbert_epoch)
        hilbert_form.addRow("Threshold (SD):", self.hilbert_sd)
        hilbert_form.addRow("Min Duration (ms):", self.hilbert_min_dur)
        hilbert_form.addRow("Min Frequency (Hz):", self.hilbert_min_freq)
        hilbert_form.addRow("Max Frequency (Hz):", self.hilbert_max_freq)
        hilbert_form.addRow("Required Peaks:", self.hilbert_peaks)
        hilbert_form.addRow("Peak Threshold (SD):", self.hilbert_peak_sd)
        hilbert_group.setLayout(hilbert_form)
        layout.addWidget(hilbert_group)

        # STE parameters
        ste_group = QtWidgets.QGroupBox("STE Parameters")
        ste_form = QtWidgets.QFormLayout()
        self.ste_threshold = QtWidgets.QLineEdit(str(params.get('ste_threshold', 2.5)))
        self.ste_window = QtWidgets.QLineEdit(str(params.get('ste_window_size', 0.01)))
        self.ste_overlap = QtWidgets.QLineEdit(str(params.get('ste_overlap', 0.5)))
        self.ste_min_freq = QtWidgets.QLineEdit(str(params.get('ste_min_freq', 80.0)))
        self.ste_max_freq = QtWidgets.QLineEdit(str(params.get('ste_max_freq', 500.0)))
        ste_form.addRow("Threshold (RMS):", self.ste_threshold)
        ste_form.addRow("Window Size (s):", self.ste_window)
        ste_form.addRow("Overlap (0-1):", self.ste_overlap)
        ste_form.addRow("Min Frequency (Hz):", self.ste_min_freq)
        ste_form.addRow("Max Frequency (Hz):", self.ste_max_freq)
        ste_group.setLayout(ste_form)
        layout.addWidget(ste_group)

        # MNI parameters
        mni_group = QtWidgets.QGroupBox("MNI Parameters")
        mni_form = QtWidgets.QFormLayout()
        self.mni_baseline = QtWidgets.QLineEdit(str(params.get('mni_baseline_window', 10.0)))
        self.mni_percentile = QtWidgets.QLineEdit(str(params.get('mni_threshold_percentile', 98.0)))
        self.mni_min_freq = QtWidgets.QLineEdit(str(params.get('mni_min_freq', 80.0)))
        self.mni_max_freq = QtWidgets.QLineEdit(str(params.get('mni_max_freq', 500.0)))
        mni_form.addRow("Baseline Window (s):", self.mni_baseline)
        mni_form.addRow("Threshold Percentile:", self.mni_percentile)
        mni_form.addRow("Min Frequency (Hz):", self.mni_min_freq)
        mni_form.addRow("Max Frequency (Hz):", self.mni_max_freq)
        mni_group.setLayout(mni_form)
        layout.addWidget(mni_group)

        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        self.analyze_btn = QtWidgets.QPushButton("Analyze (Run Consensus)")
        self.analyze_btn.clicked.connect(self.analyze)
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.close)
        button_layout.addWidget(self.analyze_btn)
        button_layout.addWidget(self.cancel_btn)
        layout.addLayout(button_layout)

        self.setLayout(layout)
        self.resize(600, 900)
        center(self)
        self.show()

    def _load_consensus_params(self):
        import json
        try:
            fname = os.path.join(self.scoreWindow.mainWindow.SETTINGS_DIR, 'consensus_params.json')
            if os.path.exists(fname):
                with open(fname, 'r') as f:
                    return json.load(f)
        except Exception:
            return {}
        return {}

    def analyze(self):
        try:
            # Voting strategy mapping
            voting_text = self.voting_combo.currentText()
            if 'Strict' in voting_text:
                voting_strategy = 'strict'
            elif 'Any' in voting_text:
                voting_strategy = 'any'
            else:
                voting_strategy = 'majority'

            # Store parameters on score window
            self.scoreWindow.consensus_voting = voting_strategy
            self.scoreWindow.consensus_overlap_ms = float(self.overlap_edit.text())

            self.scoreWindow.hilbert_epoch = float(self.hilbert_epoch.text())
            self.scoreWindow.hilbert_sd_num = float(self.hilbert_sd.text())
            self.scoreWindow.hilbert_min_duration = float(self.hilbert_min_dur.text())
            self.scoreWindow.hilbert_min_freq = float(self.hilbert_min_freq.text())
            self.scoreWindow.hilbert_max_freq = float(self.hilbert_max_freq.text())
            self.scoreWindow.hilbert_required_peaks = int(self.hilbert_peaks.text())
            self.scoreWindow.hilbert_peak_sd = float(self.hilbert_peak_sd.text())

            self.scoreWindow.ste_threshold = float(self.ste_threshold.text())
            self.scoreWindow.ste_window_size = float(self.ste_window.text())
            self.scoreWindow.ste_overlap = float(self.ste_overlap.text())
            self.scoreWindow.ste_min_freq = float(self.ste_min_freq.text())
            self.scoreWindow.ste_max_freq = float(self.ste_max_freq.text())

            self.scoreWindow.mni_baseline_window = float(self.mni_baseline.text())
            self.scoreWindow.mni_threshold_percentile = float(self.mni_percentile.text())
            self.scoreWindow.mni_min_freq = float(self.mni_min_freq.text())
            self.scoreWindow.mni_max_freq = float(self.mni_max_freq.text())

        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Invalid Input", "Please enter valid numeric values.")
            return

        settings = {
            'voting_strategy': self.scoreWindow.consensus_voting,
            'overlap_ms': self.scoreWindow.consensus_overlap_ms,
            'hilbert_epoch': self.scoreWindow.hilbert_epoch,
            'hilbert_sd_num': self.scoreWindow.hilbert_sd_num,
            'hilbert_min_duration': self.scoreWindow.hilbert_min_duration,
            'hilbert_min_freq': self.scoreWindow.hilbert_min_freq,
            'hilbert_max_freq': self.scoreWindow.hilbert_max_freq,
            'hilbert_required_peaks': self.scoreWindow.hilbert_required_peaks,
            'ste_threshold': self.scoreWindow.ste_threshold,
            'ste_window_size': self.scoreWindow.ste_window_size,
            'ste_overlap': self.scoreWindow.ste_overlap,
            'ste_min_freq': self.scoreWindow.ste_min_freq,
            'ste_max_freq': self.scoreWindow.ste_max_freq,
            'mni_baseline_window': self.scoreWindow.mni_baseline_window,
            'mni_threshold_percentile': self.scoreWindow.mni_threshold_percentile,
            'mni_min_freq': self.scoreWindow.mni_min_freq,
            'mni_max_freq': self.scoreWindow.mni_max_freq
        }
        _save_generic_settings(self.scoreWindow, 'Consensus', settings)

        if self.scoreWindow.consensus_thread.isRunning():
            self.scoreWindow.consensus_thread.quit()
            self.scoreWindow.consensus_thread.wait()

        self.scoreWindow.consensus_thread.start()
        self.scoreWindow.consensus_thread_worker = Worker(ConsensusDetection, self.scoreWindow)
        self.scoreWindow.consensus_thread_worker.moveToThread(self.scoreWindow.consensus_thread)
        self.scoreWindow.consensus_thread_worker.start.emit("start")
        self.close()


class SettingsViewer(QtWidgets.QWidget):

    def __init__(self, filename):
        super(SettingsViewer, self).__init__()
        background(self)
        width = int(self.deskW / 3)
        height = int(self.deskH / 3)
        self.setGeometry(0, 0, width, height)

        self.setWindowTitle("Settings Viewer Window")

        setting_fname_label = QtWidgets.QLabel("Settings Filename:")
        self.setting_filename = QtWidgets.QLineEdit()
        self.setting_filename.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        self.setting_filename.setText(filename)

        settings_fname_layout = QtWidgets.QHBoxLayout()
        settings_fname_layout.addWidget(setting_fname_label)
        settings_fname_layout.addWidget(self.setting_filename)

        with open(filename, 'r+') as f:
            settings = json.load(f)

        parameter_label = QtWidgets.QLabel("Settings Parameters")
        # self.parameters = QtWidgets.QTextEdit()

        self.parameters = QtWidgets.QTreeWidget()

        self.parameters_headers = {'Parameter:': 0, "Value:": 1}
        for key, value in self.parameters_headers.items():
            self.parameters.headerItem().setText(value, key)

        for key, value in settings.items():
            # text = '%s\t%s' % (str(key), str(value))
            # self.parameters.append(text)
            # new_item = QtWidgets.QTreeWidgetItem()
            new_item = TreeWidgetItem()

            new_item.setText(self.parameters_headers['Parameter:'], str(key))
            new_item.setText(self.parameters_headers['Value:'], str(value))
            self.parameters.addTopLevelItem(new_item)

        parameters_layout = QtWidgets.QVBoxLayout()
        parameters_layout.addWidget(parameter_label)
        parameters_layout.addWidget(self.parameters)

        self.close_btn = QtWidgets.QPushButton("Close")
        self.close_btn.clicked.connect(self.close_app)

        settings_layout = QtWidgets.QVBoxLayout()

        for order in [settings_fname_layout, parameters_layout, self.close_btn]:
            if 'Layout' in order.__str__():
                settings_layout.addLayout(order)
                # layout_score.addStretch(1)
            else:
                # layout_score.addWidget(order, 0, QtCore.Qt.AlignCenter)
                settings_layout.addWidget(order)
                # layout_score.addStretch(1)

        self.setLayout(settings_layout)

        center(self)

        self.show()

    def close_app(self):

        self.close()
