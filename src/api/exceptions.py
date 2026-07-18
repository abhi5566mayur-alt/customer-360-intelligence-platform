from fastapi import HTTPException


def handle_service_response(response: dict):
    """
    Convert service error responses into HTTP exceptions.
    """

    if response.get("status") == "error":
        raise HTTPException(
            status_code=400,
            detail={
                "model": response.get("model"),
                "message": response.get("message"),
            },
        )

    return response
