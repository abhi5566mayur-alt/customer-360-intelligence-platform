import pandas as pd

from src.config.paths import INTERIM_DIR, PROCESSED_DIR
from src.utils.io import save_parquet

from src.features.customer_aggregations import (
    build_customer_base_features,
    build_customer_review_features,
    build_customer_payment_features,
    build_customer_category_features,
)


def main():
    fact = pd.read_parquet(INTERIM_DIR / "fact_orders_enriched.parquet")

    # build feature blocks
    customer_base = build_customer_base_features(fact)
    customer_reviews = build_customer_review_features(fact)
    customer_payments = build_customer_payment_features(fact)
    customer_categories = build_customer_category_features(fact)

    # merge all feature blocks
    customer360 = customer_base.merge(customer_reviews, on="customer_unique_id", how="left")
    customer360 = customer360.merge(customer_payments, on="customer_unique_id", how="left")
    customer360 = customer360.merge(customer_categories, on="customer_unique_id", how="left")

    # fill numeric nulls for feature columns where appropriate
    numeric_fill_zero = [
        "orders_last_30d",
        "spend_last_30d",
        "orders_last_90d",
        "spend_last_90d",
        "num_reviews",
        "negative_review_ratio",
        "num_payment_types_used",
        "avg_installments",
        "credit_card_ratio",
        "num_unique_categories",
        "favorite_category_ratio",
    ]

    for col in numeric_fill_zero:
        if col in customer360.columns:
            customer360[col] = customer360[col].fillna(0)

    output_path = PROCESSED_DIR / "customer360_master.parquet"
    save_parquet(customer360, output_path)

    print("=== CUSTOMER360 MASTER CREATED ===")
    print("Shape:", customer360.shape)
    print("Saved to:", output_path)


if __name__ == "__main__":
    main()