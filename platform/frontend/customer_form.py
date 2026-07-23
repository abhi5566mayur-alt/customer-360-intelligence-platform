import streamlit as st


def customer_feature_form():
    """
    Reusable customer feature form.
    Returns:
        dict: customer features
        bool: explain
    """

    with st.expander("🛒 Purchase Behavior", expanded=True):
        frequency_orders = st.number_input(
            "Frequency Orders",
            min_value=0,
            value=5,
        )

        monetary_total_spend = st.number_input(
            "Total Spend",
            min_value=0.0,
            value=1850.75,
        )

        avg_order_value = st.number_input(
            "Average Order Value",
            min_value=0.0,
            value=370.15,
        )

        total_items_bought = st.number_input(
            "Total Items Bought",
            min_value=0,
            value=12,
        )

    with st.expander("⏰ Customer Activity", expanded=True):
        recency_days = st.number_input(
            "Recency (Days)",
            min_value=0,
            value=18,
        )

        customer_tenure_days = st.number_input(
            "Customer Tenure (Days)",
            min_value=0,
            value=420,
        )

        orders_last_30d = st.number_input(
            "Orders Last 30 Days",
            min_value=0,
            value=2,
        )

        spend_last_30d = st.number_input(
            "Spend Last 30 Days",
            min_value=0.0,
            value=520.0,
        )

        orders_last_90d = st.number_input(
            "Orders Last 90 Days",
            min_value=0,
            value=4,
        )

        spend_last_90d = st.number_input(
            "Spend Last 90 Days",
            min_value=0.0,
            value=980.0,
        )

    with st.expander("⭐ Customer Reviews"):
        num_reviews = st.number_input(
            "Number of Reviews",
            min_value=0,
            value=4,
        )

        avg_review_score = st.slider(
            "Average Review Score",
            1.0,
            5.0,
            4.5,
            0.1,
        )

        min_review_score = st.slider(
            "Minimum Review Score",
            1,
            5,
            3,
        )

        negative_review_ratio = st.slider(
            "Negative Review Ratio",
            0.0,
            1.0,
            0.10,
            0.01,
        )

    with st.expander("💳 Payment Information"):
        num_payment_types_used = st.number_input(
            "Payment Types Used",
            min_value=1,
            value=1,
        )

        avg_installments = st.number_input(
            "Average Installments",
            min_value=1,
            value=3,
        )

        credit_card_ratio = st.slider(
            "Credit Card Ratio",
            0.0,
            1.0,
            1.0,
            0.01,
        )

    with st.expander("🛍 Product Preferences"):
        num_unique_categories = st.number_input(
            "Unique Categories",
            min_value=1,
            value=4,
        )

        favorite_category_ratio = st.slider(
            "Favorite Category Ratio",
            0.0,
            1.0,
            0.55,
            0.01,
        )

    explain = st.checkbox(
        "Generate SHAP Explanation",
        value=False,
    )

    customer_features = {
        "frequency_orders": frequency_orders,
        "monetary_total_spend": monetary_total_spend,
        "avg_order_value": avg_order_value,
        "total_items_bought": total_items_bought,
        "recency_days": recency_days,
        "customer_tenure_days": customer_tenure_days,
        "orders_last_30d": orders_last_30d,
        "spend_last_30d": spend_last_30d,
        "orders_last_90d": orders_last_90d,
        "spend_last_90d": spend_last_90d,
        "num_reviews": num_reviews,
        "avg_review_score": avg_review_score,
        "min_review_score": min_review_score,
        "negative_review_ratio": negative_review_ratio,
        "num_payment_types_used": num_payment_types_used,
        "avg_installments": avg_installments,
        "credit_card_ratio": credit_card_ratio,
        "num_unique_categories": num_unique_categories,
        "favorite_category_ratio": favorite_category_ratio,
    }

    return customer_features, explain