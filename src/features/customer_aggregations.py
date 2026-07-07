import numpy as np
import pandas as pd


import pandas as pd


def build_customer_base_features(fact: pd.DataFrame) -> pd.DataFrame:
    """
    Build customer-level core RFM + order/spend features from fact_orders_enriched.
    One row per customer_unique_id.

    Important:
    We aggregate to order level first to reduce duplication from joins.
    For payment_value, we take the max payment total observed per order row group
    rather than summing across exploded rows.
    """
    df = fact.copy()

    # keep only rows with customer id and purchase timestamp
    df = df.dropna(subset=["customer_unique_id", "order_purchase_timestamp"])

    # ensure datetime
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"], errors="coerce")

    # snapshot date = one day after latest purchase
    snapshot_date = df["order_purchase_timestamp"].max() + pd.Timedelta(days=1)

    # -----------------------------
    # Build a safer order-level table
    # -----------------------------
    # For each order, we want:
    # - one purchase timestamp
    # - one customer
    # - one order_value
    # - item count
    #
    # Because fact_orders_enriched can contain repeated rows per order due to:
    # - multiple items
    # - multiple payments
    # - review joins
    #
    # We use:
    # - payment_value max at order level as a safer proxy than summing exploded rows
    # - unique item count
    order_level = (
        df.groupby(["customer_unique_id", "order_id"], as_index=False)
        .agg(
            order_purchase_timestamp=("order_purchase_timestamp", "first"),
            order_value=("payment_value", "max"),
            item_count=("order_item_id", "nunique"),
        )
    )

    # customer-level aggregation
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

    # recency / tenure
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

    # fill nulls for recent windows
    for col in ["orders_last_30d", "spend_last_30d", "orders_last_90d", "spend_last_90d"]:
        customer[col] = customer[col].fillna(0)

    return customer


def build_customer_review_features(fact: pd.DataFrame) -> pd.DataFrame:
    df = fact.copy()

    df = df.dropna(subset=["customer_unique_id"])

    review_df = (
        df.groupby("customer_unique_id", as_index=False)
        .agg(
            num_reviews=("review_score", lambda x: x.notna().sum()),
            avg_review_score=("review_score", "mean"),
            min_review_score=("review_score", "min"),
        )
    )

    # negative review ratio = score <= 2
    neg_ratio = (
        df.assign(is_negative_review=(df["review_score"] <= 2).astype(int))
        .groupby("customer_unique_id", as_index=False)
        .agg(
            negative_review_ratio=("is_negative_review", "mean")
        )
    )

    review_df = review_df.merge(neg_ratio, on="customer_unique_id", how="left")
    return review_df


def build_customer_payment_features(fact: pd.DataFrame) -> pd.DataFrame:
    df = fact.copy()

    df = df.dropna(subset=["customer_unique_id"])

    payment_df = (
        df.groupby("customer_unique_id", as_index=False)
        .agg(
            num_payment_types_used=("payment_type", "nunique"),
            avg_installments=("payment_installments", "mean"),
        )
    )

    # credit card ratio
    credit_ratio = (
        df.assign(is_credit_card=(df["payment_type"] == "credit_card").astype(int))
        .groupby("customer_unique_id", as_index=False)
        .agg(
            credit_card_ratio=("is_credit_card", "mean")
        )
    )

    payment_df = payment_df.merge(credit_ratio, on="customer_unique_id", how="left")
    return payment_df


def build_customer_category_features(fact: pd.DataFrame) -> pd.DataFrame:
    df = fact.copy()

    df = df.dropna(subset=["customer_unique_id"])

    # unique category count
    category_df = (
        df.groupby("customer_unique_id", as_index=False)
        .agg(
            num_unique_categories=("product_category_name_english", "nunique")
        )
    )

    # favorite category by frequency
    fav = (
        df.dropna(subset=["product_category_name_english"])
        .groupby(["customer_unique_id", "product_category_name_english"])
        .size()
        .reset_index(name="cnt")
        .sort_values(["customer_unique_id", "cnt"], ascending=[True, False])
    )

    fav_top = fav.groupby("customer_unique_id", as_index=False).first()
    fav_top = fav_top.rename(
        columns={
            "product_category_name_english": "favorite_category",
            "cnt": "favorite_category_count",
        }
    )

    # total category events per customer for ratio
    total_counts = (
        df.dropna(subset=["product_category_name_english"])
        .groupby("customer_unique_id", as_index=False)
        .size()
        .rename(columns={"size": "total_category_events"})
    )

    fav_top = fav_top.merge(total_counts, on="customer_unique_id", how="left")
    fav_top["favorite_category_ratio"] = (
        fav_top["favorite_category_count"] / fav_top["total_category_events"]
    )

    category_df = category_df.merge(
        fav_top[["customer_unique_id", "favorite_category", "favorite_category_ratio"]],
        on="customer_unique_id",
        how="left",
    )

    return category_df