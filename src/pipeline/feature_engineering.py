import copy


def add_engineered_features(customer_features: dict) -> dict:
    """
    Generate engineered features required by the trained models.
    """

    features = copy.deepcopy(customer_features)

    frequency_orders = max(features["frequency_orders"], 1)
    tenure_days = max(features["customer_tenure_days"], 1)
    total_spend = max(features["monetary_total_spend"], 1)

    features["avg_items_per_order"] = (
        features["total_items_bought"] / frequency_orders
    )

    if frequency_orders > 1:
        features["avg_days_between_orders"] = (
            tenure_days / (frequency_orders - 1)
        )
    else:
        features["avg_days_between_orders"] = -1

    features["orders_per_tenure_day"] = (
        features["frequency_orders"] / tenure_days
    )

    features["spend_per_tenure_day"] = (
        features["monetary_total_spend"] / tenure_days
    )

    features["spend_last_30d_ratio"] = (
        features["spend_last_30d"] / total_spend
    )

    features["spend_last_90d_ratio"] = (
        features["spend_last_90d"] / total_spend
    )

    features["orders_last_90d_ratio"] = (
        features["orders_last_90d"] / frequency_orders
    )

    # Default values for inference
    features["cutoff_month"] = 6
    features["cutoff_quarter"] = 2

    return features