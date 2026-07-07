from src.data.load_raw_data import load_all_raw_data
from src.data.clean_customers import clean_customers
from src.data.clean_orders import clean_orders
from src.data.clean_order_items import clean_order_items
from src.data.clean_payments import clean_payments
from src.data.clean_reviews import clean_reviews
from src.data.clean_products import clean_products

from src.config.paths import INTERIM_DIR
from src.utils.io import save_parquet


def main():
    data = load_all_raw_data()

    customers_clean = clean_customers(data["customers"])
    orders_clean = clean_orders(data["orders"])
    order_items_clean = clean_order_items(data["order_items"])
    payments_clean = clean_payments(data["payments"])
    reviews_clean = clean_reviews(data["reviews"])
    products_clean = clean_products(data["products"], data["category_translation"])

    # save parquet files
    save_parquet(customers_clean, INTERIM_DIR / "customers_clean.parquet")
    save_parquet(orders_clean, INTERIM_DIR / "orders_clean.parquet")
    save_parquet(order_items_clean, INTERIM_DIR / "order_items_clean.parquet")
    save_parquet(payments_clean, INTERIM_DIR / "payments_clean.parquet")
    save_parquet(reviews_clean, INTERIM_DIR / "reviews_clean.parquet")
    save_parquet(products_clean, INTERIM_DIR / "products_clean.parquet")

    print("=== CLEANING PIPELINE COMPLETED ===")
    print("Saved cleaned parquet files to:", INTERIM_DIR)


if __name__ == "__main__":
    main()