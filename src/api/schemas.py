from typing import Any

from pydantic import BaseModel, Field


# =====================================================
# CUSTOMER FEATURES REQUEST
# =====================================================

class CustomerFeaturesRequest(BaseModel):
    """
    Request model for customer-based predictions.
    """

    customer_features: dict[str, Any] = Field(
        ...,
        description="Dictionary containing customer feature values.",
    )

    explain: bool = Field(
        default=False,
        description="Whether to include SHAP explanations.",
    )


# =====================================================
# PRODUCT RECOMMENDATION REQUEST
# =====================================================

class RecommendationRequest(BaseModel):
    """
    Request model for product recommendations.
    """

    product_id: str = Field(
        ...,
        description="Input product ID.",
    )

    top_k: int = Field(
        default=10,
        ge=1,
        description="Number of recommendations to return.",
    )


# =====================================================
# REVIEW TOPIC REQUEST
# =====================================================

class ReviewRequest(BaseModel):
    """
    Request model for topic modeling.
    """

    review_text: str = Field(
        ...,
        description="Customer review text.",
    )
from typing import Any

from pydantic import BaseModel


# =====================================================
# COMMON RESPONSE
# =====================================================

class BaseResponse(BaseModel):
    status: str
    model: str


# =====================================================
# CHURN
# =====================================================

class ChurnResponse(BaseResponse):
    prediction: dict[str, Any]


# =====================================================
# CLV
# =====================================================

class CLVResponse(BaseResponse):
    prediction: dict[str, Any]


# =====================================================
# SEGMENTATION
# =====================================================

class SegmentationResponse(BaseResponse):
    prediction: dict[str, Any]


# =====================================================
# ANOMALY
# =====================================================

class AnomalyResponse(BaseResponse):
    prediction: dict[str, Any]


# =====================================================
# RECOMMENDATION
# =====================================================

class RecommendationResponse(BaseResponse):
    prediction: dict[str, Any]


# =====================================================
# REVIEW TOPIC
# =====================================================

class ReviewTopicResponse(BaseResponse):
    prediction: dict[str, Any]