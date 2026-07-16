"""
Validation utilities for Customer360 Service.

These functions validate user inputs before they are passed
to machine learning models.
"""

from typing import Dict, List


# ==========================================================
# CUSTOMER FEATURES
# ==========================================================

def validate_customer_features(
    customer_features: Dict,
    required_features: List[str],
):

    if not isinstance(customer_features, dict):
        raise ValueError(
            "customer_features must be a dictionary."
        )

    missing = [
        feature
        for feature in required_features
        if feature not in customer_features
    ]

    if missing:
        raise ValueError(
            f"Missing required features: {missing}"
        )

    return True


# ==========================================================
# PRODUCT ID
# ==========================================================

def validate_product_id(
    product_id,
    product_index,
):

    if product_id is None:
        raise ValueError(
            "product_id cannot be None."
        )

    if product_id not in product_index:
        raise ValueError(
            f"Unknown product_id: {product_id}"
        )

    return True


# ==========================================================
# REVIEW TEXT
# ==========================================================

def validate_review_text(review_text):

    if review_text is None:
        raise ValueError(
            "Review text cannot be None."
        )

    if not isinstance(review_text, str):
        raise ValueError(
            "Review text must be a string."
        )

    if review_text.strip() == "":
        raise ValueError(
            "Review text cannot be empty."
        )

    return True
