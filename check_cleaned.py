from pathlib import Path
import pandas as pd

base = Path("data/interim")

files = [
    "customers_clean.parquet",
    "orders_clean.parquet",
    "order_items_clean.parquet",
    "payments_clean.parquet",
    "reviews_clean.parquet",
    "products_clean.parquet",
]

for f in files:
    df = pd.read_parquet(base / f)
    print(f"{f}: {df.shape}")