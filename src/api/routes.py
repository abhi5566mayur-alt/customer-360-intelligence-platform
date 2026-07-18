from fastapi import APIRouter
from src.api.exceptions import handle_service_response

from src.api.schemas import (
    CustomerFeaturesRequest,
    RecommendationRequest,
    ReviewRequest,
    ChurnResponse,
    CLVResponse,
    SegmentationResponse,
    AnomalyResponse,
    RecommendationResponse,
    ReviewTopicResponse,
)
from src.pipeline.customer360_service import Customer360Service

router = APIRouter()

# Load all models once when the API starts
service = Customer360Service()


# =====================================================
# SYSTEM
# =====================================================

@router.get(
    "/health",
    tags=["System"],
    summary="Health Check",
    description="Check whether the Customer360 Intelligence Platform API is running.",
)
def health():
    return {
        "status": "healthy",
        "service": "Customer360 Intelligence Platform API",
    }


# =====================================================
# CHURN PREDICTION
# =====================================================

@router.post(
    "/predict/churn",
    response_model=ChurnResponse,
    tags=["Predictions"],
    summary="Predict Customer Churn",
    description="Predict whether a customer is likely to churn and optionally return SHAP explanations.",
)
def predict_churn(request: CustomerFeaturesRequest):
    response = service.predict_churn(
        customer_features=request.customer_features,
        explain=request.explain,
    )

    return handle_service_response(response)


# =====================================================
# CUSTOMER LIFETIME VALUE
# =====================================================

@router.post(
    "/predict/clv",
    response_model=CLVResponse,
    tags=["Predictions"],
    summary="Predict Customer Lifetime Value",
    description="Predict the future Customer Lifetime Value (CLV).",
)

def predict_clv(request: CustomerFeaturesRequest):
    response = service.predict_clv(
        customer_features=request.customer_features,
        explain=request.explain,
    )

    return handle_service_response(response)

# =====================================================
# CUSTOMER SEGMENTATION
# =====================================================

@router.post(
    "/predict/segment",
    response_model=SegmentationResponse,
    tags=["Predictions"],
    summary="Customer Segmentation",
    description="Assign a customer to a behavioral segment.",
)
def predict_segment(request: CustomerFeaturesRequest):
    response = service.predict_segment(
        customer_features=request.customer_features,
    )

    return handle_service_response(response)


# =====================================================
# ANOMALY DETECTION
# =====================================================

@router.post(
    "/predict/anomaly",
    response_model=AnomalyResponse,
    tags=["Predictions"],
    summary="Detect Customer Anomalies",
    description="Detect unusual customer behavior using Isolation Forest.",
)
def predict_anomaly(request: CustomerFeaturesRequest):
    response = service.detect_anomaly(
        customer_features=request.customer_features,
        explain=request.explain,
    )

    return handle_service_response(response)

# =====================================================
# PRODUCT RECOMMENDATION
# =====================================================

@router.post(
    "/recommend",
    response_model=RecommendationResponse,
    tags=["Recommendations"],
    summary="Recommend Similar Products",
    description="Recommend products using Truncated SVD.",
)
def recommend(request: RecommendationRequest):
    response = service.recommend_products(
        product_id=request.product_id,
        top_k=request.top_k,
    )

    return handle_service_response(response)

# =====================================================
# REVIEW TOPIC MODELING
# =====================================================

@router.post(
    "/review-topic",
    response_model=ReviewTopicResponse,
    tags=["Analytics"],
    summary="Analyze Customer Reviews",
    description="Identify the dominant topic discussed in a customer review using LDA topic modeling.",
)
def review_topic(request: ReviewRequest):
    response = service.review_topics(
        review_text=request.review_text,
    )

    return handle_service_response(response)