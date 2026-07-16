import numpy as np
import pandas as pd

from src.pipeline.load_models import load_all_models


class Customer360Service:

    def __init__(self):
        print("\nInitializing Customer360 Service...\n")
        self.models = load_all_models()

    # =====================================================
    # CUSTOMER SEGMENTATION
    # =====================================================

    def predict_segment(self, customer_features):

        bundle = self.models["segmentation"]

        scaler = bundle["scaler"]
        model = bundle["model"]

        feature_cols = bundle["feature_cols"]
        log_cols = bundle["log_cols"]

        df = pd.DataFrame([customer_features])

        # Ensure all expected columns exist
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0

        # Apply log transform
        for col in log_cols:
            if col in df.columns:
                df[col] = np.log1p(df[col])

        # Correct column order
        df = df[feature_cols]

        # Scale
        X = scaler.transform(df)

        # Predict
        cluster = model.predict(X)[0]

        return {
            "segment_id": int(cluster)
        }

    # =====================================================
    # CHURN
    # =====================================================

    def predict_churn(self, customer_features):

        bundle = self.models["churn"]

        scaler = bundle["scaler"]
        model = bundle["model"]
        feature_cols = bundle["feature_cols"]

        df = pd.DataFrame([customer_features])

        # Add missing columns
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0

        # Correct feature order
        df = df[feature_cols]

        # Scale
        X = scaler.transform(df)

        # Predict
        probability = model.predict_proba(X)[0][1]
        prediction = model.predict(X)[0]

        # Risk bucket
        if probability >= 0.70:
            risk = "High"
        elif probability >= 0.40:
            risk = "Medium"
        else:
            risk = "Low"

        return {
            "prediction": int(prediction),
            "churn_probability": round(float(probability), 4),
            "risk": risk,
        }

    # =====================================================
    # CLV PREDICTION
    # =====================================================

    def predict_clv(self, customer_features):

        bundle = self.models["clv"]

        model = bundle["model"]
        feature_cols = bundle["feature_names"]

        df = pd.DataFrame([customer_features])

        # ------------------------------------
        # Add missing columns
        # ------------------------------------

        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0

        # ------------------------------------
        # Correct feature order
        # ------------------------------------

        df = df[feature_cols]

        # ------------------------------------
        # Predict
        # ------------------------------------

        clv = float(model.predict(df)[0])

        # Safety
        clv = max(clv, 0)

        # ------------------------------------
        # Business Value Bucket
        # ------------------------------------

        if clv >= 500:
            value = "High"

        elif clv >= 150:
            value = "Medium"

        else:
            value = "Low"

        return {

            "predicted_clv": round(clv, 2),

            "customer_value": value

        }
    # =====================================================
    # RECOMMENDATION
    # =====================================================

    def recommend_products(self, customer_id, top_k=10):
        raise NotImplementedError("Coming next")

    # =====================================================
    # ANOMALY
    # =====================================================

    def detect_anomaly(self, customer_features):
        raise NotImplementedError("Coming next")

    # =====================================================
    # TOPIC MODELING
    # =====================================================

    def review_topics(self, review_text):
        raise NotImplementedError("Coming next")


# =====================================================
# TEST
# =====================================================

if __name__ == "__main__":

    service = Customer360Service()

    print("\n===================================")
    print(" CUSTOMER360 SERVICE TEST")
    print("===================================")

    # =====================================================
    # SEGMENTATION TEST
    # =====================================================

    segmentation_features = {}

    for col in service.models["segmentation"]["feature_cols"]:
        segmentation_features[col] = 1.0

    segment = service.predict_segment(segmentation_features)

    print("\nSegmentation Prediction")
    print("-----------------------")
    print(segment)

    # =====================================================
    # CHURN TEST
    # =====================================================

    churn_features = {}

    for col in service.models["churn"]["feature_cols"]:
        churn_features[col] = 1.0

    churn = service.predict_churn(churn_features)

    print("\nChurn Prediction")
    print("----------------")
    print(churn)

    print("\n===================================")
    print(" LOADED MODELS")
    print("===================================")

    for model_name in service.models.keys():
        print(model_name)
    # =====================================================
    # CLV TEST
    # =====================================================

    clv_features = {}

    for col in service.models["clv"]["feature_names"]:
        clv_features[col] = 1.0

    clv = service.predict_clv(clv_features)

    print("\nCLV Prediction")
    print("----------------")
    print(clv)