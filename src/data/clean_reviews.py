import pandas as pd


DATE_COLS = [
    "review_creation_date",
    "review_answer_timestamp",
]


def clean_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    df = reviews.copy()

    # standardize column names
    df.columns = [col.strip().lower() for col in df.columns]

    # drop duplicates
    df = df.drop_duplicates()

    # parse datetime columns
    for col in DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    keep_cols = [
        "review_id",
        "order_id",
        "review_score",
        "review_comment_title",
        "review_comment_message",
        "review_creation_date",
        "review_answer_timestamp",
    ]
    df = df[keep_cols]

    # remove rows missing order_id
    df = df.dropna(subset=["order_id"])

    return df