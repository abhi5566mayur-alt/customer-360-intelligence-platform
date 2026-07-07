import pandas as pd


def clean_order_items(order_items: pd.DataFrame) -> pd.DataFrame:
    df = order_items.copy()

    # standardize column names
    df.columns = [col.strip().lower() for col in df.columns]

    # drop duplicates
    df = df.drop_duplicates()

    keep_cols = [
        "order_id",
        "order_item_id",
        "product_id",
        "seller_id",
        "shipping_limit_date",
        "price",
        "freight_value",
    ]
    df = df[keep_cols]

    # parse datetime
    if "shipping_limit_date" in df.columns:
        df["shipping_limit_date"] = pd.to_datetime(df["shipping_limit_date"], errors="coerce")

    # remove rows missing essential keys
    df = df.dropna(subset=["order_id", "product_id"])

    return df