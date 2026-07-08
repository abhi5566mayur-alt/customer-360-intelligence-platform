
import numpy as np
import pandas as pd


def build_customer_features_before_cutoff(fact: pd.DataFrame, cutoff_date: str) -> pd.DataFrame:
    """
    Build customer-level features using only data BEFORE the cutoff date.
    This is the feature snapshot used for supervised learning.
    """
    df = fact.copy()
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"], errors="coerce")

    cutoff = pd.Timestamp(cutoff_date)

    # historical data only
    hist = df[df["order_purchase_timestamp"] < cutoff].copy()
    hist = hist.dropna(subset=["customer_unique_id", "order_id", "order_purchase_timestamp"])

    # ---------------------------------------------------
    # Order-level aggregation first
    # ---------------------------------------------------
    order_level = (
        hist.groupby(["customer_unique_id", "order_id"], as_index=False)
        .agg(
            order_purchase_timestamp=("order_purchase_timestamp", "first"),
            order_value=("payment_value", "max"),
            item_count=("order_item_id", "nunique"),
        )
    )

    snapshot_date = cutoff

    # ---------------------------------------------------
    # Base customer aggregation
    # ---------------------------------------------------
    customer = (
        order_level.groupby("customer_unique_id", as_index=False)
        .agg(
            first_order_date=("order_purchase_timestamp", "min"),
            last_order_date=("order_purchase_timestamp", "max"),
            frequency_orders=("order_id", "nunique"),
            monetary_total_spend=("order_value", "sum"),
            avg_order_value=("order_value", "mean"),
            total_items_bought=("item_count", "sum"),
        )
    )

    customer["snapshot_date"] = snapshot_date
    customer["recency_days"] = (snapshot_date - customer["last_order_date"]).dt.days
    customer["customer_tenure_days"] = (
        customer["last_order_date"] - customer["first_order_date"]
    ).dt.days

    # ---------------------------------------------------
    # Recent windows
    # ---------------------------------------------------
    last_30_cutoff = snapshot_date - pd.Timedelta(days=30)
    last_90_cutoff = snapshot_date - pd.Timedelta(days=90)

    orders_30 = (
        order_level[order_level["order_purchase_timestamp"] >= last_30_cutoff]
        .groupby("customer_unique_id", as_index=False)
        .agg(
            orders_last_30d=("order_id", "nunique"),
            spend_last_30d=("order_value", "sum"),
        )
    )

    orders_90 = (
        order_level[order_level["order_purchase_timestamp"] >= last_90_cutoff]
        .groupby("customer_unique_id", as_index=False)
        .agg(
            orders_last_90d=("order_id", "nunique"),
            spend_last_90d=("order_value", "sum"),
        )
    )

    customer = customer.merge(orders_30, on="customer_unique_id", how="left")
    customer = customer.merge(orders_90, on="customer_unique_id", how="left")

    for col in ["orders_last_30d", "spend_last_30d", "orders_last_90d", "spend_last_90d"]:
        customer[col] = customer[col].fillna(0)

    # ---------------------------------------------------
    # New feature block 1: average items per order
    # ---------------------------------------------------
    customer["avg_items_per_order"] = (
        customer["total_items_bought"] / customer["frequency_orders"]
    ).replace([np.inf, -np.inf], 0)

    # ---------------------------------------------------
    # New feature block 2: cadence / order spacing
    # ---------------------------------------------------
    # Compute average gap between consecutive orders per customer
    order_level_sorted = order_level.sort_values(
        ["customer_unique_id", "order_purchase_timestamp"]
    ).copy()

    order_level_sorted["prev_order_ts"] = (
        order_level_sorted.groupby("customer_unique_id")["order_purchase_timestamp"].shift(1)
    )

    order_level_sorted["days_since_prev_order"] = (
        order_level_sorted["order_purchase_timestamp"] - order_level_sorted["prev_order_ts"]
    ).dt.days

    cadence = (
        order_level_sorted.groupby("customer_unique_id", as_index=False)
        .agg(
            avg_days_between_orders=("days_since_prev_order", "mean"),
        )
    )

    customer = customer.merge(cadence, on="customer_unique_id", how="left")

    # For one-order customers, avg_days_between_orders will be NaN
    customer["avg_days_between_orders"] = customer["avg_days_between_orders"].fillna(-1)

    # ---------------------------------------------------
    # New feature block 3: normalized order/spend intensity
    # ---------------------------------------------------
    # avoid division by zero for tenure=0
    tenure_den = customer["customer_tenure_days"].replace(0, 1)

    customer["orders_per_tenure_day"] = customer["frequency_orders"] / tenure_den
    customer["spend_per_tenure_day"] = customer["monetary_total_spend"] / tenure_den

    # ---------------------------------------------------
    # New feature block 4: recent activity ratios
    # ---------------------------------------------------
    spend_den = customer["monetary_total_spend"].replace(0, 1)
    freq_den = customer["frequency_orders"].replace(0, 1)

    customer["spend_last_30d_ratio"] = customer["spend_last_30d"] / spend_den
    customer["spend_last_90d_ratio"] = customer["spend_last_90d"] / spend_den
    customer["orders_last_90d_ratio"] = customer["orders_last_90d"] / freq_den

    # clip possible weird numeric issues
    ratio_cols = [
        "spend_last_30d_ratio",
        "spend_last_90d_ratio",
        "orders_last_90d_ratio",
    ]
    for col in ratio_cols:
        customer[col] = customer[col].replace([np.inf, -np.inf], 0).fillna(0)

    return customer

def build_targets_90d(fact: pd.DataFrame, cutoff_date: str) -> pd.DataFrame:
    """
    Build churn and CLV targets using the 90-day window AFTER cutoff_date.

    target_churn_90d = 1 if no purchase in next 90d, else 0
    target_clv_90d = total spend in next 90d
    """
    df = fact.copy()
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"], errors="coerce")

    cutoff = pd.Timestamp(cutoff_date)
    future_end = cutoff + pd.Timedelta(days=90)

    # future window
    future = df[
        (df["order_purchase_timestamp"] >= cutoff)
        & (df["order_purchase_timestamp"] < future_end)
    ].copy()

    future = future.dropna(subset=["customer_unique_id", "order_id"])

    # order-level future aggregation
    future_order_level = (
        future.groupby(["customer_unique_id", "order_id"], as_index=False)
        .agg(
            future_order_value=("payment_value", "max"),
        )
    )

    # customer-level future targets
    future_customer = (
        future_order_level.groupby("customer_unique_id", as_index=False)
        .agg(
            future_orders_90d=("order_id", "nunique"),
            target_clv_90d=("future_order_value", "sum"),
        )
    )

    # churn target
    future_customer["target_churn_90d"] = (
        future_customer["future_orders_90d"] == 0
    ).astype(int)

    return future_customer[["customer_unique_id", "target_clv_90d", "target_churn_90d"]]