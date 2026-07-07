import pandas as pd


def clean_products(products: pd.DataFrame, category_translation: pd.DataFrame) -> pd.DataFrame:
    df = products.copy()
    cat = category_translation.copy()

    # standardize column names
    df.columns = [col.strip().lower() for col in df.columns]
    cat.columns = [col.strip().lower() for col in cat.columns]

    # drop duplicates
    df = df.drop_duplicates()
    cat = cat.drop_duplicates()

    # merge english category names
    df = df.merge(cat, on="product_category_name", how="left")

    keep_cols = [
        "product_id",
        "product_category_name",
        "product_category_name_english",
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ]

    # keep only columns that exist
    keep_cols = [col for col in keep_cols if col in df.columns]
    df = df[keep_cols]

    df = df.dropna(subset=["product_id"])

    return df