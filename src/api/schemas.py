from typing import Any

from pydantic import BaseModel, ConfigDict 


# =====================================================
# CUSTOMER FEATURES REQUEST
# =====================================================

class CustomerFeaturesRequest(BaseModel):
    customer_features: dict
    explain: bool = False

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "customer_features": {
                    "frequency_orders": 5,
                    "monetary_total_spend": 1850.75,
                    "avg_order_value": 370.15,
                    "total_items_bought": 12,
                    "recency_days": 18,
                    "customer_tenure_days": 420,
                    "orders_last_30d": 2,
                    "spend_last_30d": 520.0,
                    "orders_last_90d": 4,
                    "spend_last_90d": 980.0,
                    "num_reviews": 4,
                    "avg_review_score": 4.5,
                    "min_review_score": 3,
                    "negative_review_ratio": 0.10,
                    "num_payment_types_used": 1,
                    "avg_installments": 3,
                    "credit_card_ratio": 1.0,
                    "num_unique_categories": 4,
                    "favorite_category_ratio": 0.55
                },
                "explain": True
            }
        }
    )
# =====================================================
# PRODUCT RECOMMENDATION REQUEST
# =====================================================

class RecommendationRequest(BaseModel):
    product_id: str
    top_k: int = 5

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "product_id": "4244733e06e7ecb4970a6e2683c13e61",
                "top_k": 5
            }
        }
    )
# =====================================================
# REVIEW TOPIC REQUEST
# =====================================================

class ReviewRequest(BaseModel):
    review_text: str

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "review_text": "Excellent product. Fast delivery and great packaging. Highly recommended."
            }
        }
    )
# =====================================================
# COMMON RESPONSE
# =====================================================

class BaseResponse(BaseModel):
    status: str
    model: str

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "success",
                "model": "churn"
            }
        }
    )

# =====================================================
# CHURN
# =====================================================
class ChurnResponse(BaseResponse):
    prediction: dict[str, Any]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "success",
                "model": "churn",
                "prediction": {
                    "prediction": 0,
                    "probability": 0.18
                }
            }
        }
    )

# =====================================================
# CLV
# =====================================================
class CLVResponse(BaseResponse):
    prediction: dict[str, Any]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "success",
                "model": "clv",
                "prediction": {
                    "predicted_clv": 1250.75
                }
            }
        }
    )
# =====================================================
# SEGMENTATION
# =====================================================

class SegmentationResponse(BaseResponse):
    prediction: dict[str, Any]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "success",
                "model": "segmentation",
                "prediction": {
                    "segment": 2
                }
            }
        }
    )
# =====================================================
# ANOMALY
# =====================================================
class AnomalyResponse(BaseResponse):
    prediction: dict[str, Any]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "success",
                "model": "anomaly",
                "prediction": {
                    "is_anomaly": False
                }
            }
        }
    )

# =====================================================
# RECOMMENDATION
# =====================================================

class RecommendationResponse(BaseResponse):
    prediction: dict[str, Any]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "success",
                "model": "recommendation",
                "prediction": {
                    "recommended_products": [
                        "product_001",
                        "product_002",
                        "product_003"
                    ]
                }
            }
        }
    )
# =====================================================
# REVIEW TOPIC
# =====================================================
class ReviewTopicResponse(BaseResponse):
    prediction: dict[str, Any]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "success",
                "model": "review_topic",
                "prediction": {
                    "topic": 3,
                    "topic_name": "Delivery Experience"
                }
            }
        }
    )