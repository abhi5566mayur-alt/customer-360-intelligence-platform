from fastapi import FastAPI

from src.api.routes import router

app = FastAPI(
    title="Customer360 Intelligence Platform API",
    description="Production-ready Customer360 Machine Learning API",
    version="1.0.0",
)

app.include_router(router)


@app.get("/")
def root():
    return {
        "message": "Customer360 Intelligence Platform API is running."
    }