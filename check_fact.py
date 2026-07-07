from pathlib import Path
import pandas as pd

fact_path = Path("data/interim/fact_orders_enriched.parquet")
fact = pd.read_parquet(fact_path)

print("fact_orders_enriched shape:", fact.shape)
print("\nColumns:")
print(fact.columns.tolist())

print("\nSample rows:")
print(
    fact[
        [
            "order_id",
            "customer_unique_id",
            "order_purchase_timestamp",
            "product_id",
            "product_category_name_english",
            "price",
            "payment_type",
            "payment_value",
            "review_score",
        ]
    ].head(10)
)