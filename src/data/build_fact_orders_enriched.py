import pandas as pd

from src.config.paths import INTERIM_DIR
from src.utils.io import save_parquet


def main():
    # Load cleaned parquet files
    customers = pd.read_parquet(INTERIM_DIR / "customers_clean.parquet")
    orders = pd.read_parquet(INTERIM_DIR / "orders_clean.parquet")
    order_items = pd.read_parquet(INTERIM_DIR / "order_items_clean.parquet")
    payments = pd.read_parquet(INTERIM_DIR / "payments_clean.parquet")
    reviews = pd.read_parquet(INTERIM_DIR / "reviews_clean.parquet")
    products = pd.read_parquet(INTERIM_DIR / "products_clean.parquet")

    # 1) orders + customers
    fact = orders.merge(
        customers,
        on="customer_id",
        how="left"
    )

    # 2) add order items
    fact = fact.merge(
        order_items,
        on="order_id",
        how="left"
    )

    # 3) add payments
    # payments can have multiple rows per order, so row count may increase
    fact = fact.merge(
        payments,
        on="order_id",
        how="left"
    )

    # 4) add reviews
    fact = fact.merge(
        reviews,
        on="order_id",
        how="left"
    )

    # 5) add product info
    fact = fact.merge(
        products,
        on="product_id",
        how="left"
    )

    # Save
    output_path = INTERIM_DIR / "fact_orders_enriched.parquet"
    save_parquet(fact, output_path)

    print("=== FACT ORDERS ENRICHED CREATED ===")
    print("Shape:", fact.shape)
    print("Saved to:", output_path)


if __name__ == "__main__":
    main()