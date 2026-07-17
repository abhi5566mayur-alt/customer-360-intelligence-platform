import time
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.config.settings import (
    CHURN_HIGH_RISK_THRESHOLD,
    CHURN_MEDIUM_RISK_THRESHOLD,
    CLV_HIGH_VALUE_THRESHOLD,
    CLV_MEDIUM_VALUE_THRESHOLD,
    DEFAULT_TOP_K_RECOMMENDATIONS,
    ANOMALY_NORMAL_LABEL,
    ANOMALY_ANOMALY_LABEL,
    DEFAULT_TOPIC_PROBABILITY_DECIMALS,
)

from src.pipeline.business_rules import (
    SEGMENT_MAPPING,
    CHURN_ACTIONS,
    CLV_ACTIONS,
    ANOMALY_ACTIONS,
    TOPIC_MAPPING,
)
from src.pipeline.validators import (
    validate_customer_features,
    validate_product_id,
    validate_review_text,
)
from src.pipeline.response_formatter import (
    success_response,
    error_response,
)
from src.utils.logger import logger
from src.pipeline.load_models import load_all_models
from src.explainability.shap_explainer import ShapExplainer


class Customer360Service:

    def __init__(self):
        print("\nInitializing Customer360 Service...\n")
        self.models = load_all_models()
        self.shap_explainer = ShapExplainer(
            self.models
        )
        logger.info("Customer360 Service initialized successfully.")

    # =====================================================
    # CUSTOMER SEGMENTATION
    # =====================================================

    def predict_segment(self, customer_features):

        try:
            start_time = time.perf_counter()

            bundle = self.models["segmentation"]

            scaler = bundle["scaler"]
            model = bundle["model"]

            feature_cols = bundle["feature_cols"]
            log_cols = bundle["log_cols"]

            validate_customer_features(
                customer_features,
                feature_cols,
            )

            df = pd.DataFrame([customer_features])

            for col in feature_cols:
                if col not in df.columns:
                    df[col] = 0

            for col in log_cols:
                if col in df.columns:
                    df[col] = np.log1p(df[col])

            df = df[feature_cols]

            X = scaler.transform(df)

            cluster = model.predict(X)[0]

            segment = SEGMENT_MAPPING.get(
                int(cluster),
                {
                    "name": "Unknown",
                    "description": "Unknown segment.",
                    "business_action": "No recommendation available.",
                },
            )
            execution_time = time.perf_counter() - start_time
            logger.info(f"Model=Segmentation | Status=SUCCESS | ExecutionTime={execution_time:.4f}s")

            return success_response(
                "Segmentation",
                {
                    "segment_id": int(cluster),
                    "segment_name": segment["name"],
                    "description": segment["description"],
                    "business_action": segment["business_action"],
                },
            )

        except Exception as e:
            execution_time = time.perf_counter() - start_time
            logger.error(f"Model=Segmentation | Status=ERROR | Error={e} | ExecutionTime={execution_time:.4f}s")

            return error_response(
                "Segmentation",
                str(e),
            )
    # =====================================================
    # CHURN
    # =====================================================

    def predict_churn(
        self,
        customer_features,
        explain: bool = False,
    ):
        
        try:
            start_time = time.perf_counter()

            bundle = self.models["churn"]

            scaler = bundle["scaler"]
            model = bundle["model"]
            feature_cols = bundle["feature_cols"]

            validate_customer_features(
                customer_features,
                feature_cols,
            )

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
            if probability >= CHURN_HIGH_RISK_THRESHOLD:
                risk = "High"
            elif probability >= CHURN_MEDIUM_RISK_THRESHOLD:
                risk = "Medium"
            else:
                risk = "Low"
            
            execution_time = time.perf_counter() - start_time
            logger.info(f"Model=Churn | Status=SUCCESS | ExecutionTime={execution_time:.4f}s")

            
            prediction_result = {
                "will_churn": bool(prediction),
                "churn_probability": round(float(probability), 4),
                "risk": risk,
                "business_action": CHURN_ACTIONS[risk],
            }

            if explain:
                prediction_result["explanation"] = (
                    self.shap_explainer.explain_churn(
                        customer_features
                    )
                )

            return success_response(
                "Churn",
                prediction_result,
            )
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            logger.error(f"Model=Churn | Status=ERROR | Error={e} | ExecutionTime={execution_time:.4f}s")
            
            return error_response(
                 "Churn",
                str(e),
            )
    # =====================================================
    # CLV PREDICTION
    # =====================================================

    def predict_clv(
        self,
        customer_features,
        explain: bool = False,
    ):
        
        try:
            start_time = time.perf_counter()
       
            bundle = self.models["clv"]

            model = bundle["model"]
            
            feature_cols = bundle["feature_names"]

            validate_customer_features(
                customer_features,
                feature_cols,
            )

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

        
        # Business Value Bucket

            if clv >= CLV_HIGH_VALUE_THRESHOLD:
                value = "High Value Customer"

            elif clv >= CLV_MEDIUM_VALUE_THRESHOLD:
                value = "Medium Value Customer"

            else:
                value = "Low Value Customer"
                
            execution_time = time.perf_counter() - start_time    
            logger.info(f"Model=CLV | Status=SUCCESS | ExecutionTime={execution_time:.4f}s")

            prediction_result = {
                "predicted_clv": round(clv, 2),
                "customer_value": value,
                "business_action": CLV_ACTIONS[
                    value.replace(" Value Customer", "")
                ],
            }

            if explain:
                prediction_result["explanation"] = (
                    self.shap_explainer.explain_clv(
                        customer_features
                    )
                )

            return success_response(
                "CLV",
                prediction_result,
            )
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            logger.error(f"Model=CLV | Status=ERROR | Error={e} | ExecutionTime={execution_time:.4f}s")
            return error_response(
                "CLV",
                str(e),
            )
    # =====================================================
    # RECOMMENDATION
    # =====================================================

    def recommend_products(
        self,
        product_id,
        top_k=DEFAULT_TOP_K_RECOMMENDATIONS,
    ):
        try:
            start_time = time.perf_counter()

            bundle = self.models["recommendation"]

            product_embeddings = bundle["product_embeddings"]
            product_index = bundle["product_index"]

            validate_product_id(
                product_id,
                product_index,
            )

            idx = product_index.get_loc(product_id)

            query_embedding = product_embeddings[idx].reshape(1, -1)

            similarities = cosine_similarity(
                query_embedding,
                product_embeddings,
            )[0]

            sorted_indices = similarities.argsort()[::-1]

            recommendations = []

            for i in sorted_indices:

                if i == idx:
                    continue

                recommendations.append({

                    "product_id": product_index[i],

                    "similarity": round(float(similarities[i]), 4),

                })

                if len(recommendations) >= top_k:
                    break
            execution_time = time.perf_counter() - start_time    
            logger.info(f"Model=Recommendation | Status=SUCCESS | ExecutionTime={execution_time:.4f}s")

            return success_response(
                    "Recommendation",
                    {
                        "algorithm": "Truncated SVD",
                        "input_product": product_id,
                        "recommendations": recommendations,
                    },
                )
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            logger.error(f"Model=Recommendation | Status=ERROR | Error={e} | ExecutionTime={execution_time:.4f}s")
            return error_response(
                "Recommendation",
                str(e),
            )
    # =====================================================
    # ANOMALY DETECTION
    # =====================================================

    def detect_anomaly(
        self,
        customer_features,
        explain: bool = False,
    ):
        try:
            start_time = time.perf_counter()

            bundle = self.models["anomaly"]

            scaler = bundle["scaler"]
            model = bundle["model"]
            feature_cols = bundle["feature_cols"]
            log_cols = bundle["log_cols"]

            validate_customer_features(
                customer_features,
                feature_cols,
            )

            df = pd.DataFrame([customer_features])

            # ------------------------------------
            # Add missing columns
            # ------------------------------------

            for col in feature_cols:
                if col not in df.columns:
                    df[col] = 0

            # ------------------------------------
            # Log transform
            # ------------------------------------

            for col in log_cols:
                if col in df.columns:
                    df[col] = np.log1p(df[col])

            # ------------------------------------
            # Correct feature order
            # ------------------------------------

            df = df[feature_cols]

            # ------------------------------------
            # Scale
            # ------------------------------------

            X = scaler.transform(df)

            # ------------------------------------
            # Predict
            # ------------------------------------

            prediction = model.predict(X)[0]

            score = float(model.decision_function(X)[0])

            if prediction == -1:
                label = ANOMALY_ANOMALY_LABEL
                risk = "High"
            else:
                label = ANOMALY_NORMAL_LABEL
                risk = "Low"
                
            execution_time = time.perf_counter() - start_time
            logger.info(f"Model=Anomaly Detection | Status=SUCCESS | ExecutionTime={execution_time:.4f}s")
           
            prediction_result = {
                "status": label,
                "anomaly_score": round(score, 4),
                "risk": risk,
                "business_action": ANOMALY_ACTIONS[label],
            }

            if explain:
                prediction_result["explanation"] = (
                    self.shap_explainer.explain_anomaly(
                        customer_features
                    )
                )

            return success_response(
                "Anomaly Detection",
                prediction_result,
            )
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            logger.error(f"Model=Anomaly Detection | Status=ERROR | Error={e} | ExecutionTime={execution_time:.4f}s")
            return error_response(
                "Anomaly Detection",
                str(e),
            )

    
    # =====================================================
    # TOPIC MODELING
    # =====================================================
    

    def review_topics(self, review_text):
        
        
        try:
            start_time = time.perf_counter()
            
            validate_review_text(review_text)

            import re

            bundle = self.models["topic_model"]

            model = bundle["model"]
            vectorizer = bundle["vectorizer"]
            topic_keywords = bundle["topic_keywords"]

            # ------------------------------------
            # Clean review
            # ------------------------------------

            text = str(review_text).lower()

            text = re.sub(r"http\S+", " ", text)
            text = re.sub(r"\d+", " ", text)
            text = re.sub(r"[^\w\s]", " ", text)
            text = re.sub(r"\s+", " ", text).strip()

            # ------------------------------------
            # Vectorize
            # ------------------------------------

            X = vectorizer.transform([text])

            # ------------------------------------
            # Topic probabilities
            # ------------------------------------

            topic_probs = model.transform(X)[0]

            topic_id = int(np.argmax(topic_probs))

            probability = float(np.max(topic_probs))

            # ------------------------------------
            # Keywords
            # ------------------------------------

            keywords = ""

            for topic in topic_keywords:

                if topic["topic_id"] == topic_id:

                    keywords = topic["keywords"]

                    break
            topic = TOPIC_MAPPING.get(

            topic_id,

            {
                "name": "Unknown",

                "description": "Unknown topic.",

                "business_action": "No recommendation available.",
            },
        )  
            execution_time = time.perf_counter() - start_time
            logger.info(f"Model=Topic Modeling | Status=SUCCESS | ExecutionTime={execution_time:.4f}s")
            return success_response(
                    "Topic Modeling",
                    {
                        "dominant_topic": topic_id,
                        "topic_name": topic["name"],
                        "topic_probability": round(
                            probability,
                            DEFAULT_TOPIC_PROBABILITY_DECIMALS,
                        ),
                        "keywords": keywords,
                        "description": topic["description"],
                        "business_action": topic["business_action"],
                    },
                )
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            logger.error(f"Model=Topic Modeling | Status=ERROR | Error={e} | ExecutionTime={execution_time:.4f}s")
            return error_response(
                "Topic Modeling",
                str(e),
            )    
    


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

    churn = service.predict_churn(
        churn_features,
        explain=True,
    )

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

    clv = service.predict_clv(
        clv_features,
        explain=True,
    )

    print("\nCLV Prediction")
    print("----------------")
    print(clv)
    # =====================================================
    # RECOMMENDATION TEST
    # =====================================================

    sample_product = service.models["recommendation"]["product_index"][0]

    recommendation = service.recommend_products(sample_product)

    print("\nRecommendation")
    print("----------------")
    print(recommendation)
    # =====================================================
    # ANOMALY TEST
    # =====================================================

    anomaly_features = {}

    for col in service.models["anomaly"]["feature_cols"]:
        anomaly_features[col] = 1.0

    anomaly = service.detect_anomaly(
        anomaly_features,
        explain=True,
    )

    print("\nAnomaly Detection")
    print("----------------")
    print(anomaly)
    # =====================================================
    # TOPIC MODEL TEST
    # =====================================================

    sample_review = """
    Delivery was very fast.
    The product quality is excellent.
    I would definitely buy again.
    """

    topic = service.review_topics(sample_review)

    print("\nTopic Modeling")
    print("----------------")
    print(topic)