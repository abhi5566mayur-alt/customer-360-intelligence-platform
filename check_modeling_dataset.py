from pathlib import Path
import pandas as pd

path = Path("data/processed/customer360_modeling_dataset.parquet")
df = pd.read_parquet(path)

print("modeling dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nRows per cutoff:")
print(df["cutoff_date"].value_counts().sort_index())

print("\nTarget value counts:")
print(df["target_churn_90d"].value_counts(dropna=False))

print("\nChurn ratio by cutoff:")
print(df.groupby("cutoff_date")["target_churn_90d"].mean())

print("\nCLV summary:")
print(df["target_clv_90d"].describe())

print("\nSample rows:")
print(
    df[
        [
            "customer_unique_id",
            "cutoff_date",
            "frequency_orders",
            "monetary_total_spend",
            "orders_last_90d",
            "spend_last_90d",
            "target_churn_90d",
            "target_clv_90d",
        ]
    ].head(10)
)
