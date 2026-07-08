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

    # keep only required rows
    hist = hist.dropna(subset=["customer_unique_id", "order_id", "order_purchase_timestamp"])

    # order-level aggregation first to avoid duplicate inflation from joins
    order_level = (
        hist.groupby(["customer_unique_id", "order_id"], as_index=False)
        .agg(
            order_purchase_timestamp=("order_purchase_timestamp", "first"),
            order_value=("payment_value", "max"),
            item_count=("order_item_id", "nunique"),
        )
    )

    snapshot_date = cutoff

    # base customer features
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

    # recent windows
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