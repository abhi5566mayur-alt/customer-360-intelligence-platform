import pandas as pd


DATE_COLS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]


def clean_orders(orders: pd.DataFrame) -> pd.DataFrame:
    df = orders.copy()

    # standardize column names
    df.columns = [col.strip().lower() for col in df.columns]

    # drop exact duplicates
    df = df.drop_duplicates()

    # parse datetime columns
    for col in DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    keep_cols = [
        "order_id",
        "customer_id",
        "order_status",
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    df = df[keep_cols]

    # remove rows missing required ids
    df = df.dropna(subset=["order_id", "customer_id"])

    return df
