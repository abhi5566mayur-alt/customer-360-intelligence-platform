"""
Utility functions for standardized API/service responses.
"""


def success_response(model_name: str, prediction: dict):
    """
    Return a standardized success response.
    """

    return {
        "status": "success",
        "model": model_name,
        "prediction": prediction
    }


def error_response(model_name: str, message: str):
    """
    Return a standardized error response.
    """

    return {
        "status": "error",
        "model": model_name,
        "message": message
    }