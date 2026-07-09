import pandas as pd
from pathlib import Path

path = Path("data/processed/customer360_segmentation_dataset.parquet")
df = pd.read_parquet(path)

print("segmentation dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nNull counts:")
print(df.isnull().sum())

print("\nSummary stats:")
print(df.describe())

print("\nSample rows:")
print(df.head(10))