"""
Application-wide configuration constants.

All configurable thresholds should live here.
Avoid hardcoding values inside service classes.
"""

# ==========================================================
# CHURN
# ==========================================================

CHURN_HIGH_RISK_THRESHOLD = 0.70
CHURN_MEDIUM_RISK_THRESHOLD = 0.40

# ==========================================================
# CUSTOMER LIFETIME VALUE
# ==========================================================

CLV_HIGH_VALUE_THRESHOLD = 500.0
CLV_MEDIUM_VALUE_THRESHOLD = 150.0

# ==========================================================
# RECOMMENDATION
# ==========================================================

DEFAULT_TOP_K_RECOMMENDATIONS = 10

# ==========================================================
# ANOMALY
# ==========================================================

ANOMALY_NORMAL_LABEL = "Normal"
ANOMALY_ANOMALY_LABEL = "Anomaly"

# ==========================================================
# TOPIC MODELING
# ==========================================================

DEFAULT_TOPIC_PROBABILITY_DECIMALS = 4

# ==========================================================
# GENERAL
# ==========================================================

RANDOM_STATE = 42