from src.pipeline.customer360_service import Customer360Service


class Customer360Predictor:

    def __init__(self):

        self.service = Customer360Service()

    def predict_customer(
        self,
        customer_features,
        review_text,
        product_id,
    ):

        result = {}

        # -----------------------------
        # Segmentation
        # -----------------------------

        result["segment"] = self.service.predict_segment(
            customer_features
        )

        # -----------------------------
        # Churn
        # -----------------------------

        result["churn"] = self.service.predict_churn(
            customer_features
        )

        # -----------------------------
        # CLV
        # -----------------------------

        result["clv"] = self.service.predict_clv(
            customer_features
        )

        # -----------------------------
        # Recommendation
        # -----------------------------

        result["recommendation"] = (
            self.service.recommend_products(
                product_id
            )
        )

        # -----------------------------
        # Anomaly
        # -----------------------------

        result["anomaly"] = (
            self.service.detect_anomaly(
                customer_features
            )
        )

        # -----------------------------
        # Topic Modeling
        # -----------------------------

        result["review_topic"] = (
            self.service.review_topics(
                review_text
            )
        )

        return result


if __name__ == "__main__":

    predictor = Customer360Predictor()

    customer_features = {}

    # Union of all required feature names
    feature_names = set()

    feature_names.update(
        predictor.service.models["segmentation"]["feature_cols"]
    )

    feature_names.update(
        predictor.service.models["churn"]["feature_cols"]
    )

    feature_names.update(
        predictor.service.models["clv"]["feature_names"]
    )

    feature_names.update(
        predictor.service.models["anomaly"]["feature_cols"]
    )

    for feature in feature_names:
        customer_features[feature] = 1.0

    product_id = (
        predictor
        .service
        .models["recommendation"]["product_index"][0]
    )

    review = (
        "Delivery was fast and the product quality is excellent."
    )

    report = predictor.predict_customer(
        customer_features,
        review,
        product_id,
    )

    print("\n")
    print("=" * 60)
    print("CUSTOMER 360 REPORT")
    print("=" * 60)

    for key, value in report.items():

        print()

        print(key.upper())

        print("-" * 40)

        print(value)