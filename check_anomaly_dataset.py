import pandas as pd
from pathlib import Path

path = Path("data/processed/customer360_anomaly_dataset.parquet")

df = pd.read_parquet(path)

print("Dataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nNull values:")
print(df.isnull().sum())

print("\nSummary:")
print(df.describe())

print("\nSample:")
print(df.head(10))