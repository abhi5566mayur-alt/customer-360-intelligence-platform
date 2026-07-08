import pandas as pd

from src.config.paths import INTERIM_DIR, PROCESSED_DIR
from src.utils.io import save_parquet
from src.features.target_builders import (
    build_customer_features_before_cutoff,
    build_targets_90d,
)
from src.features.customer_aggregations import (
    build_customer_review_features,
    build_customer_payment_features,
    build_customer_category_features,
)

CUTOFF_DATES = [
    "2017-12-01",
    "2018-02-01",
    "2018-04-01",
    "2018-06-01",
]


def build_snapshot_dataset(fact: pd.DataFrame, cutoff_date: str) -> pd.DataFrame:
    cutoff = pd.Timestamp(cutoff_date)

    # historical data only
    hist_fact = fact[fact["order_purchase_timestamp"] < cutoff].copy()

    # feature blocks
    customer_base = build_customer_features_before_cutoff(fact, cutoff_date)
    customer_reviews = build_customer_review_features(hist_fact)
    customer_payments = build_customer_payment_features(hist_fact)
    customer_categories = build_customer_category_features(hist_fact)

    snapshot_df = customer_base.merge(customer_reviews, on="customer_unique_id", how="left")
    snapshot_df = snapshot_df.merge(customer_payments, on="customer_unique_id", how="left")
    snapshot_df = snapshot_df.merge(customer_categories, on="customer_unique_id", how="left")

    # targets
    targets = build_targets_90d(fact, cutoff_date)
    snapshot_df = snapshot_df.merge(targets, on="customer_unique_id", how="left")

    # customers with no future activity => churn=1, clv=0
    snapshot_df["target_clv_90d"] = snapshot_df["target_clv_90d"].fillna(0)
    snapshot_df["target_churn_90d"] = snapshot_df["target_churn_90d"].fillna(1).astype(int)

    # fill feature nulls
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
        if col in snapshot_df.columns:
            snapshot_df[col] = snapshot_df[col].fillna(0)

    snapshot_df["cutoff_date"] = cutoff
    return snapshot_df


def main():
    fact = pd.read_parquet(INTERIM_DIR / "fact_orders_enriched.parquet")
    fact["order_purchase_timestamp"] = pd.to_datetime(
        fact["order_purchase_timestamp"], errors="coerce"
    )

    all_snapshots = []

    for cutoff_date in CUTOFF_DATES:
        snapshot_df = build_snapshot_dataset(fact, cutoff_date)
        print(f"Built snapshot for cutoff {cutoff_date}: {snapshot_df.shape}")
        all_snapshots.append(snapshot_df)

    modeling_df = pd.concat(all_snapshots, ignore_index=True)

    output_path = PROCESSED_DIR / "customer360_modeling_dataset.parquet"
    save_parquet(modeling_df, output_path)

    print("\n=== ROLLING MODELING DATASET CREATED ===")
    print("Cutoffs:", CUTOFF_DATES)
    print("Shape:", modeling_df.shape)
    print("Saved to:", output_path)

    print("\nOverall target summary:")
    print(modeling_df[["target_churn_90d", "target_clv_90d"]].describe(include="all"))

    print("\nChurn distribution by cutoff:")
    print(
        modeling_df.groupby("cutoff_date")["target_churn_90d"]
        .value_counts(normalize=True)
        .rename("ratio")
    )


if __name__ == "__main__":
    main()