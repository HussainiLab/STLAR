# Multi-Select EOI Feature Implementation

## Overview
Enhanced the Automatic Detection tab to support multiple EOI (Events of Interest) selection, allowing batch operations on detected events.

## Changes Made

### 1. Modified EOI Tree Widget Selection Mode
**File:** `hfoGUI/core/Score.py` (Line 518)

Changed the EOI tree widget to use `ExtendedSelection` mode to support multi-select:
```python
self.EOI.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
```

**Benefits:**
- Users can select multiple EOIs using Ctrl+Click (individual) or Shift+Click (range)
- All existing single-select functionality remains compatible

### 2. Updated `addEOI()` Method
**File:** `hfoGUI/core/Score.py` (Lines 1676-1721)

Enhanced to handle multiple selected EOIs:
- Changed to iterate through all `self.EOI.selectedItems()` instead of just one
- Processes each selected EOI and adds it to the Score tab
- Automatically removes added EOIs from the Automatic Detection tab
- Updates the "events detected" counter for each removed EOI

**Key Behavior:**
- Users can now select multiple EOIs and move them all to the Score tab in one action
- The button label updated to "Add Selected EOI(s) to Score" for clarity

### 3. Updated `deleteEOI()` Method
**File:** `hfoGUI/core/Score.py` (Lines 1738-1779)

Refactored to support batch deletion:
- Safely handles multiple selected items by collecting them first
- Deletes items in proper order without index issues
- Removes IDs from tracking list (`self.IDs`)
- Updates the "events detected" counter for each deleted EOI
- Auto-selects the next available item after deletion

**Key Behavior:**
- Users can select multiple EOIs and delete them all at once
- The button label updated to "Remove Selected EOI(s)" for clarity
- Properly maintains data integrity with ID tracking

### 4. Added Batch Frequency Marking Feature
**File:** `hfoGUI/core/Score.py` (Lines 555-572, 1781-1827)

New UI components added:
- Label: "Mark Selected As:"
- Dropdown menu with frequency options:
  - Ripples
  - Fast Ripples
  - HFOs
  - Spindles
- Button: "Apply to Selected"

New method `applyBatchFrequencyAction()`:
- Validates that a frequency type was selected
- Validates that at least one EOI is selected
- Stores the frequency type in each selected EOI's UserRole data
- Appends frequency type to the Settings File column for visual indication
- Resets the dropdown after action
- Provides user feedback via status bar

**Example Usage:**
1. Select 5 ripple-like EOIs in the automatic detection results
2. Select "Ripples" from the dropdown
3. Click "Apply to Selected"
4. All 5 EOIs are marked as Ripples and display "[Ripples]" in the Settings File column

### 5. Updated Button Labels
**File:** `hfoGUI/core/Score.py` (Lines 544-548)

- "Remove Selected EOI" → "Remove Selected EOI(s)"
- "Add Selected EOI to Score" → "Add Selected EOI(s) to Score"

**Impact:** Clarifies to users that these buttons work with multiple selections

## User Workflow

### Moving Multiple EOIs to Score Tab
1. In the Automatic Detection tab, select multiple EOIs (Ctrl+Click for individual, Shift+Click for ranges)
2. Click "Add Selected EOI(s) to Score"
3. Selected EOIs move to the Score tab for manual review/scoring
4. Event counter updates automatically

### Deleting Multiple EOIs
1. Select multiple EOIs to remove (Ctrl+Click or Shift+Click)
2. Click "Remove Selected EOI(s)"
3. All selected EOIs are deleted
4. Next available EOI is automatically selected
5. Event counter updates

### Marking Multiple EOIs with Frequency Type
1. Select multiple EOIs that appear to be a specific type (e.g., all Ripples)
2. Choose the frequency type from the "Mark Selected As:" dropdown
3. Click "Apply to Selected"
4. Status bar shows: "Marked X EOI(s) as [FrequencyType]"
5. Each EOI shows the frequency marker in the Settings File column

## Backward Compatibility

- All changes are backward compatible
- Single-select operations work exactly as before
- Existing data structures and tracking lists unchanged
- No breaking changes to other methods or components

## Data Persistence

- Frequency type metadata is stored in each item's UserRole data
- Can be enhanced to persist to disk by saving frequency info when exporting/saving detection results
- Visual indicator appended to Settings File column for reference

## Future Enhancements

Potential improvements:
1. Save frequency markings to detection result files
2. Export frequency classifications with EOI data
3. Add keyboard shortcuts (Ctrl+Shift+D for delete, Ctrl+Shift+A for add, etc.)
4. Add context menu for batch operations
5. Add "Select All", "Select None", "Select by Type" filters
6. Undo/Redo functionality for batch operations
7. Color coding for different frequency types in the tree widget
