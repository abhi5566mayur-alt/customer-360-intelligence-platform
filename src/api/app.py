from fastapi import FastAPI

from src.api.error_handlers import register_exception_handlers
from src.api.routes import router

app = FastAPI(
    title="Customer360 Intelligence Platform API",
    description="""
## Customer360 Intelligence Platform

A production-ready Machine Learning API for customer analytics.

### Features

- Customer Churn Prediction
- Customer Lifetime Value (CLV) Prediction
- Customer Segmentation
- Customer Anomaly Detection
- Product Recommendation
- Review Topic Modeling
- SHAP Explainability

Built using:

- FastAPI
- Scikit-learn
- SHAP
- Pandas
- NumPy
""",
    version="1.0.0",
    contact={
        "name": "Mayur Chakrawarty",
    },
    license_info={
        "name": "MIT License",
    },
)

# Register global exception handlers
register_exception_handlers(app)

# Register API routes with version prefix
app.include_router(
    router,
    prefix="/api/v1",
)


@app.get("/", tags=["System"], summary="API Root")
def root():
    return {
        "message": "Customer360 Intelligence Platform API is running.",
        "version": "1.0.0",
        "docs": "/docs",
        "api_base": "/api/v1",
    }