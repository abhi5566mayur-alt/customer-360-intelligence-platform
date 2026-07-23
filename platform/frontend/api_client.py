import requests

BASE_URL = "http://127.0.0.1:8000/api/v1"

def check_api():
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False
