import pandas as pd


def clean_customers(customers: pd.DataFrame) -> pd.DataFrame:
    df = customers.copy()

    # standardize column names
    df.columns = [col.strip().lower() for col in df.columns]

    # drop exact duplicates
    df = df.drop_duplicates()

    # keep only required columns for V1
    keep_cols = [
        "customer_id",
        "customer_unique_id",
        "customer_zip_code_prefix",
        "customer_city",
        "customer_state",
    ]
    df = df[keep_cols]

    # remove rows missing key identifiers
    df = df.dropna(subset=["customer_id", "customer_unique_id"])

    return df
