from fastapi import APIRouter

from src.api.schemas import (
    CustomerFeaturesRequest,
    RecommendationRequest,
    ReviewRequest,
)
from src.pipeline.customer360_service import Customer360Service

router = APIRouter()

service = Customer360Service()


@router.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "Customer360 Intelligence Platform API",
    }


# ==========================
# CHURN
# ==========================

@router.post("/predict/churn")
def predict_churn(request: CustomerFeaturesRequest):
    return service.predict_churn(
        customer_features=request.customer_features,
        explain=request.explain,
    )


# ==========================
# CLV
# ==========================

@router.post("/predict/clv")
def predict_clv(request: CustomerFeaturesRequest):
    return service.predict_clv(
        customer_features=request.customer_features,
        explain=request.explain,
    )


# ==========================
# SEGMENTATION
# ==========================

@router.post("/predict/segment")
def predict_segment(request: CustomerFeaturesRequest):
    return service.predict_segment(
        customer_features=request.customer_features,
    )


# ==========================
# ANOMALY
# ==========================

@router.post("/predict/anomaly")
def predict_anomaly(request: CustomerFeaturesRequest):
    return service.detect_anomaly(
        customer_features=request.customer_features,
        explain=request.explain,
    )


# ==========================
# RECOMMENDATION
# ==========================

@router.post("/recommend")
def recommend(request: RecommendationRequest):
    return service.recommend_products(
        product_id=request.product_id,
        top_k=request.top_k,
    )


# ==========================
# REVIEW TOPIC
# ==========================

@router.post("/review-topic")
def review_topic(request: ReviewRequest):
    return service.classify_review_topic(
        review_text=request.review_text,
    )