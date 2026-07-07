import pandas as pd


def clean_payments(payments: pd.DataFrame) -> pd.DataFrame:
    df = payments.copy()

    # standardize column names
    df.columns = [col.strip().lower() for col in df.columns]

    # drop duplicates
    df = df.drop_duplicates()

    keep_cols = [
        "order_id",
        "payment_sequential",
        "payment_type",
        "payment_installments",
        "payment_value",
    ]
    df = df[keep_cols]

    # remove rows missing join key
    df = df.dropna(subset=["order_id"])

    return df