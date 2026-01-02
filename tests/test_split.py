from hfoGUI.dl_training.manifest_splitter import load_manifests, subject_wise_split

df = load_manifests(['E:/DATA/Ephys/AD-LFP-Project/Experiment-1/Group-1/A10/DL2-01/manifest.csv'])
print(f"Subjects: {df['subject_id'].unique()}")
print(f"Total events: {len(df)}")

train, val = subject_wise_split(df, 0.2, 42)
print(f"\nResult:")
print(f"Train: {len(train)} events")
print(f"Val: {len(val)} events")
