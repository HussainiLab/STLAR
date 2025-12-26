"""
Multi-Manifest Train/Val Splitter for PyHFO Training
Combines multiple manifest.csv files and splits by subject/session for proper validation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import List, Tuple
import json


def load_manifests(manifest_paths: List[str], subject_ids: List[str] = None) -> pd.DataFrame:
    """
    Load multiple manifest.csv files and combine them.
    
    Parameters:
    -----------
    manifest_paths : list of str
        Paths to manifest.csv files
    subject_ids : list of str, optional
        Subject IDs corresponding to each manifest. If None, auto-generates IDs.
    
    Returns:
    --------
    pd.DataFrame with columns: segment_path, label, subject_id, manifest_source
    """
    all_data = []
    
    for idx, manifest_path in enumerate(manifest_paths):
        df = pd.read_csv(manifest_path)
        
        # Validate required columns
        if 'segment_path' not in df.columns or 'label' not in df.columns:
            raise ValueError(f"Manifest {manifest_path} must have 'segment_path' and 'label' columns")
        
        # Add subject ID
        if subject_ids and idx < len(subject_ids):
            subject_id = subject_ids[idx]
        else:
            # Auto-generate from manifest filename or index
            manifest_name = Path(manifest_path).stem
            subject_id = f"subject_{manifest_name}" if manifest_name != "manifest" else f"subject_{idx+1:03d}"
        
        df['subject_id'] = subject_id
        df['manifest_source'] = manifest_path
        
        all_data.append(df)
        print(f"✓ Loaded {len(df)} events from {manifest_path} (Subject: {subject_id})")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\n{'='*60}")
    print(f"Total: {len(combined_df)} events from {len(manifest_paths)} manifests")
    print(f"{'='*60}\n")
    
    return combined_df


def print_statistics(df: pd.DataFrame, name: str = "Dataset"):
    """Print detailed statistics about the dataset."""
    print(f"\n{name} Statistics:")
    print(f"{'─'*60}")
    print(f"Total events: {len(df)}")
    print(f"Subjects: {df['subject_id'].nunique()}")
    print(f"\nLabel distribution:")
    label_counts = df['label'].value_counts()
    for label, count in label_counts.items():
        pct = (count / len(df)) * 100
        print(f"  {str(label):20s}: {count:4d} ({pct:5.1f}%)")
    
    print(f"\nEvents per subject:")
    subject_counts = df['subject_id'].value_counts().sort_index()
    for subject, count in subject_counts.items():
        print(f"  {str(subject):20s}: {count:4d} events")
    print(f"{'─'*60}")


def subject_wise_split(df: pd.DataFrame, 
                       val_fraction: float = 0.2,
                       random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data by subject - no events from the same subject in both train and val.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Combined dataframe with subject_id column
    val_fraction : float
        Fraction of subjects to use for validation (0.0 to 1.0)
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    train_df, val_df : tuple of DataFrames
    """
    # Get unique subjects
    subjects = df['subject_id'].unique()
    n_subjects = len(subjects)
    n_val = max(1, int(n_subjects * val_fraction))  # At least 1 subject for val
    
    # Shuffle subjects
    np.random.seed(random_state)
    shuffled_subjects = np.random.permutation(subjects)
    
    # Split subjects
    val_subjects = shuffled_subjects[:n_val]
    train_subjects = shuffled_subjects[n_val:]
    
    # Create train/val dataframes
    train_df = df[df['subject_id'].isin(train_subjects)].copy()
    val_df = df[df['subject_id'].isin(val_subjects)].copy()
    
    print(f"\n{'='*60}")
    print(f"SUBJECT-WISE SPLIT (No Data Leakage)")
    print(f"{'='*60}")
    print(f"Train subjects ({len(train_subjects)}): {sorted(train_subjects)}")
    print(f"Val subjects ({len(val_subjects)}): {sorted(val_subjects)}")
    
    return train_df, val_df


def stratified_subject_split(df: pd.DataFrame,
                             val_fraction: float = 0.2,
                             random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split subjects while trying to maintain label distribution balance.
    Useful when subjects have very different label distributions.
    """
    from sklearn.model_selection import train_test_split
    
    # Calculate label distribution per subject
    subject_labels = df.groupby(['subject_id', 'label']).size().unstack(fill_value=0)
    
    # Get dominant label per subject for stratification
    dominant_labels = subject_labels.idxmax(axis=1)
    
    # Split subjects
    subjects = df['subject_id'].unique()
    
    try:
        train_subjects, val_subjects = train_test_split(
            subjects,
            test_size=val_fraction,
            random_state=random_state,
            stratify=dominant_labels
        )
    except ValueError:
        # Fall back to regular split if stratification fails
        print("⚠️  Stratification failed (too few subjects per class), using regular subject split")
        return subject_wise_split(df, val_fraction, random_state)
    
    train_df = df[df['subject_id'].isin(train_subjects)].copy()
    val_df = df[df['subject_id'].isin(val_subjects)].copy()
    
    print(f"\n{'='*60}")
    print(f"STRATIFIED SUBJECT-WISE SPLIT")
    print(f"{'='*60}")
    print(f"Train subjects ({len(train_subjects)}): {sorted(train_subjects)}")
    print(f"Val subjects ({len(val_subjects)}): {sorted(val_subjects)}")
    
    return train_df, val_df


def check_class_balance(train_df: pd.DataFrame, val_df: pd.DataFrame, min_samples: int = 10):
    """
    Check if both train and val have sufficient samples per class.
    Warn if imbalanced.
    """
    print(f"\n{'='*60}")
    print("CLASS BALANCE CHECK")
    print(f"{'='*60}")
    
    train_counts = train_df['label'].value_counts()
    val_counts = val_df['label'].value_counts()
    
    all_labels = set(train_counts.index) | set(val_counts.index)
    
    warnings = []
    for label in sorted(all_labels):
        train_n = train_counts.get(label, 0)
        val_n = val_counts.get(label, 0)
        
        status = "✓"
        if train_n < min_samples or val_n < min_samples:
            status = "⚠️"
            warnings.append(f"{label}: train={train_n}, val={val_n}")
        
        print(f"{status} {str(label):20s}: Train={train_n:4d}, Val={val_n:4d}")
    
    if warnings:
        print(f"\n⚠️  WARNING: Some classes have fewer than {min_samples} samples:")
        for w in warnings:
            print(f"   {w}")
        print("   Consider collecting more data or using data augmentation")
    else:
        print(f"\n✓ All classes have at least {min_samples} samples in both sets")
    
    print(f"{'='*60}")


def save_splits(train_df: pd.DataFrame, 
                val_df: pd.DataFrame,
                output_dir: str,
                save_metadata: bool = True):
    """
    Save train/val splits to CSV files.
    
    Parameters:
    -----------
    train_df, val_df : DataFrames
        Train and validation data
    output_dir : str
        Directory to save outputs
    save_metadata : bool
        Whether to save split metadata (subject assignments, statistics)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save train/val CSVs (only segment_path and label columns)
    train_file = output_path / "train.csv"
    val_file = output_path / "val.csv"
    
    train_df[['segment_path', 'label']].to_csv(train_file, index=False)
    val_df[['segment_path', 'label']].to_csv(val_file, index=False)
    
    print(f"\n{'='*60}")
    print(f"SAVED SPLITS")
    print(f"{'='*60}")
    print(f"✓ Train: {train_file} ({len(train_df)} events)")
    print(f"✓ Val:   {val_file} ({len(val_df)} events)")
    
    # Save metadata
    if save_metadata:
        metadata = {
            'train': {
                'n_events': len(train_df),
                'n_subjects': train_df['subject_id'].nunique(),
                'subjects': sorted(train_df['subject_id'].unique().tolist()),
                'label_counts': train_df['label'].value_counts().to_dict()
            },
            'val': {
                'n_events': len(val_df),
                'n_subjects': val_df['subject_id'].nunique(),
                'subjects': sorted(val_df['subject_id'].unique().tolist()),
                'label_counts': val_df['label'].value_counts().to_dict()
            }
        }
        
        metadata_file = output_path / "split_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Metadata: {metadata_file}")
    
    # Save full dataframes with subject info (for reference)
    train_full = output_path / "train_full.csv"
    val_full = output_path / "val_full.csv"
    train_df.to_csv(train_full, index=False)
    val_df.to_csv(val_full, index=False)
    print(f"✓ Full train (with metadata): {train_full}")
    print(f"✓ Full val (with metadata): {val_full}")
    
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Combine multiple manifest.csv files and split into train/val sets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with auto-detected subject IDs
  python split_manifests.py manifest1.csv manifest2.csv manifest3.csv
  
  # Specify custom subject IDs
  python split_manifests.py manifest1.csv manifest2.csv --subjects subj_A subj_B
  
  # Custom validation fraction and output directory
  python split_manifests.py *.csv --val-frac 0.3 --output splits/
  
  # Use stratified splitting (maintains label distribution)
  python split_manifests.py *.csv --stratified
  
  # Split AND train models in one command
  python split_manifests.py *.csv --train --epochs 50 --batch-size 32
  
  # Train only artifact and spike models (not ehfo)
  python split_manifests.py *.csv --train --model-types artifact spike
        """
    )
    
    parser.add_argument('manifests', nargs='+', 
                       help='Paths to manifest.csv files')
    parser.add_argument('--subjects', nargs='+', default=None,
                       help='Subject IDs (optional, auto-generated if not provided)')
    parser.add_argument('--val-frac', type=float, default=0.2,
                       help='Validation fraction (default: 0.2 = 20%%)')
    parser.add_argument('--output', '-o', default='./splits',
                       help='Output directory (default: ./splits)')
    parser.add_argument('--stratified', action='store_true',
                       help='Use stratified splitting (maintains label distribution)')
    parser.add_argument('--min-samples', type=int, default=10,
                       help='Minimum samples per class for balance warning (default: 10)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    # Training arguments
    parser.add_argument('--train', action='store_true',
                       help='Train deep learning models after splitting')
    parser.add_argument('--model-types', nargs='+', 
                       default=['artifact', 'spike', 'ehfo'],
                       choices=['artifact', 'spike', 'ehfo'],
                       help='Model types to train (default: all)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size (default: 32)')
    parser.add_argument('--no-onnx', action='store_true',
                       help='Skip ONNX export (only save .pt files)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.val_frac <= 0 or args.val_frac >= 1:
        raise ValueError("--val-frac must be between 0 and 1")
    
    if len(args.manifests) < 2:
        print("⚠️  WARNING: Only 1 manifest provided. Consider using temporal split instead.")
        print("   Multiple manifests (subjects/sessions) provide better validation.")
    
    # Load and combine manifests
    print(f"\n{'='*60}")
    print(f"LOADING MANIFESTS")
    print(f"{'='*60}")
    combined_df = load_manifests(args.manifests, args.subjects)
    
    # Print combined statistics
    print_statistics(combined_df, "Combined Dataset")
    
    # Perform split
    if args.stratified:
        train_df, val_df = stratified_subject_split(combined_df, args.val_frac, args.seed)
    else:
        train_df, val_df = subject_wise_split(combined_df, args.val_frac, args.seed)
    
    # Print split statistics
    print_statistics(train_df, "Training Set")
    print_statistics(val_df, "Validation Set")
    
    # Check class balance
    check_class_balance(train_df, val_df, args.min_samples)
    
    # Save outputs
    save_splits(train_df, val_df, args.output)
    
    # Train models if requested
    if args.train:
        train_csv = Path(args.output) / "train.csv"
        val_csv = Path(args.output) / "val.csv"
        models_dir = Path(args.output) / "models"
        
        train_models(
            train_csv=str(train_csv),
            val_csv=str(val_csv),
            output_dir=str(models_dir),
            model_types=args.model_types,
            epochs=args.epochs,
            batch_size=args.batch_size,
            export_onnx=not args.no_onnx
        )
    else:
        print("\n✓ Done! Ready for PyHFO training:")
        print(f"  Option 1 - Use this script to train:")
        print(f"    python split_manifests.py {' '.join(args.manifests)} --train")
        print(f"  Option 2 - Use PyHFO directly:")
        print(f"    python -m pyhfo.train --train {args.output}/train.csv --val {args.output}/val.csv")


def train_models(train_csv: str, 
                val_csv: str,
                output_dir: str,
                model_types: List[str] = ['artifact', 'spike', 'ehfo'],
                epochs: int = 50,
                batch_size: int = 32,
                export_onnx: bool = True):
    """
    Train PyHFO deep learning models and export to .pt and .onnx formats.
    
    Parameters:
    -----------
    train_csv : str
        Path to training CSV
    val_csv : str
        Path to validation CSV
    output_dir : str
        Directory to save trained models
    model_types : list of str
        Models to train: 'artifact', 'spike', 'ehfo'
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training
    export_onnx : bool
        Whether to export ONNX format (in addition to PyTorch .pt)
    """
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import Dataset, DataLoader
        import numpy as np
        from pathlib import Path
        import json
        from tqdm import tqdm
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Install with: pip install torch numpy tqdm")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("TRAINING DEEP LEARNING MODELS")
    print(f"{'='*60}")
    print(f"Train data: {train_csv}")
    print(f"Val data: {val_csv}")
    print(f"Output: {output_dir}")
    print(f"Models: {', '.join(model_types)}")
    print(f"{'='*60}\n")
    
    # Load data
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    
    # Define HFO Dataset
    class HFODataset(Dataset):
        def __init__(self, csv_path, label_map=None):
            self.data = pd.read_csv(csv_path)
            
            # Create label mapping
            if label_map is None:
                unique_labels = sorted(self.data['label'].unique())
                self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
            else:
                self.label_map = label_map
            
            self.reverse_map = {v: k for k, v in self.label_map.items()}
            print(f"  Label mapping: {self.label_map}")
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            row = self.data.iloc[idx]
            
            # Load signal data
            try:
                signal = np.load(row['segment_path'])
            except Exception as e:
                print(f"⚠️  Error loading {row['segment_path']}: {e}")
                # Return zeros if file not found
                signal = np.zeros(1000)
            
            # Normalize signal
            signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
            
            # Convert to tensor
            signal_tensor = torch.FloatTensor(signal).unsqueeze(0)  # Add channel dim
            
            # Get label
            label = self.label_map[row['label']]
            
            return signal_tensor, label
    
    # Define CNN Model for HFO Classification
    class HFONet(nn.Module):
        def __init__(self, input_length=1000, num_classes=3):
            super(HFONet, self).__init__()
            
            self.features = nn.Sequential(
                # Conv block 1
                nn.Conv1d(1, 32, kernel_size=7, padding=3),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(0.2),
                
                # Conv block 2
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(0.2),
                
                # Conv block 3
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(0.3),
                
                # Conv block 4
                nn.Conv1d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1)
            )
            
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, num_classes)
            )
        
        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x
    
    # Training function
    def train_model(train_loader, val_loader, num_classes, model_name, epochs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n{'─'*60}")
        print(f"Training {model_name}")
        print(f"Device: {device}")
        print(f"{'─'*60}")
        
        model = HFONet(num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
        
        best_val_loss = float('inf')
        best_model_state = None
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for signals, labels in pbar:
                signals, labels = signals.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(signals)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Validation
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for signals, labels in val_loader:
                    signals, labels = signals.to(device), labels.to(device)
                    outputs = model(signals)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            # Calculate metrics
            train_loss = train_loss / len(train_loader)
            val_loss = val_loss / len(val_loader)
            train_acc = 100. * train_correct / train_total
            val_acc = 100. * val_correct / val_total
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                print(f"  ✓ New best model (val_loss={val_loss:.4f})")
            
            scheduler.step(val_loss)
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        return model, history
    
    # Train each model type
    for model_type in model_types:
        print(f"\n{'='*60}")
        print(f"MODEL TYPE: {model_type.upper()}")
        print(f"{'='*60}")
        
        # Create datasets based on model type
        if model_type == 'artifact':
            # Binary: Artifact vs Real HFO
            print("Task: Artifact rejection (Artifact vs Real HFO)")
            # Remap labels: Artifact=0, everything else=1
            train_ds = HFODataset(train_csv)
            val_ds = HFODataset(val_csv, label_map=train_ds.label_map)
            
            # Binary remap for artifact detection
            def remap_artifact(label):
                return 0 if label == 'Artifact' else 1
            
            train_ds.data['label'] = train_ds.data['label'].apply(remap_artifact)
            val_ds.data['label'] = val_ds.data['label'].apply(remap_artifact)
            num_classes = 2
            
        elif model_type == 'spike':
            # Binary: Spike-HFO vs Pure HFO
            print("Task: Spike-HFO detection (Spike vs Pure HFO)")
            train_ds = HFODataset(train_csv)
            val_ds = HFODataset(val_csv, label_map=train_ds.label_map)
            
            def remap_spike(label):
                return 0 if 'spike' in label.lower() else 1
            
            train_ds.data['label'] = train_ds.data['label'].apply(remap_spike)
            val_ds.data['label'] = val_ds.data['label'].apply(remap_spike)
            num_classes = 2
            
        elif model_type == 'ehfo':
            # Multi-class: Classify HFO types (Ripple, Fast Ripple, etc.)
            print("Task: Epileptogenic HFO classification (Multi-class)")
            train_ds = HFODataset(train_csv)
            val_ds = HFODataset(val_csv, label_map=train_ds.label_map)
            num_classes = len(train_ds.label_map)
        
        else:
            print(f"⚠️  Unknown model type: {model_type}, skipping")
            continue
        
        # Create data loaders
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        
        print(f"Training samples: {len(train_ds)}")
        print(f"Validation samples: {len(val_ds)}")
        print(f"Number of classes: {num_classes}")
        
        # Train model
        model, history = train_model(train_loader, val_loader, num_classes, model_type, epochs)
        
        # Save PyTorch model (.pt)
        pt_path = output_path / f"{model_type}_model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'label_map': train_ds.label_map,
            'num_classes': num_classes,
            'history': history
        }, pt_path)
        print(f"\n✓ Saved PyTorch model: {pt_path}")
        
        # Export to ONNX
        if export_onnx:
            try:
                onnx_path = output_path / f"{model_type}_model.onnx"
                
                # Create dummy input
                dummy_input = torch.randn(1, 1, 1000)
                
                # Export
                torch.onnx.export(
                    model.cpu(),
                    dummy_input,
                    onnx_path,
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={
                        'input': {0: 'batch_size'},
                        'output': {0: 'batch_size'}
                    }
                )
                print(f"✓ Saved ONNX model: {onnx_path}")
                
            except Exception as e:
                print(f"⚠️  ONNX export failed: {e}")
        
        # Save training history
        history_path = output_path / f"{model_type}_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"✓ Saved training history: {history_path}")
    
    # Save metadata
    metadata = {
        'train_csv': str(train_csv),
        'val_csv': str(val_csv),
        'models_trained': model_types,
        'epochs': epochs,
        'batch_size': batch_size,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    metadata_path = output_path / "training_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Models saved to: {output_path}")
    print("Files created:")
    for model_type in model_types:
        print(f"  • {model_type}_model.pt")
        if export_onnx:
            print(f"  • {model_type}_model.onnx")
        print(f"  • {model_type}_history.json")
    print(f"  • training_metadata.json")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
