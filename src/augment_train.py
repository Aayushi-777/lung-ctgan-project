import pandas as pd
import pathlib

# Path to processed data
p = pathlib.Path("data/processed")

# Load original train and synthetic minority data
tr = pd.read_csv(p / "train.csv")
syn = pd.read_csv(p / "synthetic_minority.csv")

# Align synthetic columns with train.csv
syn = syn.reindex(columns=tr.columns, fill_value=0)

# Combine and shuffle
aug = pd.concat([tr, syn], ignore_index=True).sample(frac=1, random_state=42)

# Save
out_path = p / "train_augmented.csv"
aug.to_csv(out_path, index=False)

print(f"Created {out_path}, shape={aug.shape}")
