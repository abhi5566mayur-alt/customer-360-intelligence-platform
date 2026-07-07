import pandas as pd

from src.config.paths import RAW_DIR
from src.utils.io import read_csv_safe


def load_customers() -> pd.DataFrame:
    return read_csv_safe(RAW_DIR / "olist_customers_dataset.csv")


def load_orders() -> pd.DataFrame:
    return read_csv_safe(RAW_DIR / "olist_orders_dataset.csv")


def load_order_items() -> pd.DataFrame:
    return read_csv_safe(RAW_DIR / "olist_order_items_dataset.csv")


def load_payments() -> pd.DataFrame:
    return read_csv_safe(RAW_DIR / "olist_order_payments_dataset.csv")


def load_reviews() -> pd.DataFrame:
    return read_csv_safe(RAW_DIR / "olist_order_reviews_dataset.csv")


def load_products() -> pd.DataFrame:
    return read_csv_safe(RAW_DIR / "olist_products_dataset.csv")


def load_category_translation() -> pd.DataFrame:
    return read_csv_safe(RAW_DIR / "product_category_name_translation.csv")


def load_all_raw_data() -> dict:
    data = {
        "customers": load_customers(),
        "orders": load_orders(),
        "order_items": load_order_items(),
        "payments": load_payments(),
        "reviews": load_reviews(),
        "products": load_products(),
        "category_translation": load_category_translation(),
    }
    return data


if __name__ == "__main__":
    data = load_all_raw_data()

    print("=== RAW DATA LOADED SUCCESSFULLY ===")
    for name, df in data.items():
        print(f"{name}: {df.shape}")
        