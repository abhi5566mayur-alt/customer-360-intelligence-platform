# =====================================================
# CUSTOMER SEGMENT DEFINITIONS
# =====================================================

SEGMENT_MAPPING = {

    0: {
        "name": "High Value Customers",
        "description": (
            "Customers with high spending and strong purchasing behavior."
        ),
        "business_action": (
            "Retain with VIP rewards and exclusive offers."
        ),
    },

    1: {
        "name": "At Risk Customers",
        "description": (
            "Customers showing declining engagement and purchase activity."
        ),
        "business_action": (
            "Launch retention campaigns immediately."
        ),
    },

    2: {
        "name": "Loyal Customers",
        "description": (
            "Customers who purchase consistently and interact frequently."
        ),
        "business_action": (
            "Upsell premium products and reward loyalty."
        ),
    },

    3: {
        "name": "New Customers",
        "description": (
            "Recently acquired customers with limited purchase history."
        ),
        "business_action": (
            "Improve onboarding and encourage second purchase."
        ),
    },

    4: {
        "name": "Occasional Customers",
        "description": (
            "Customers who purchase infrequently."
        ),
        "business_action": (
            "Increase engagement using personalized promotions."
        ),
    },
}


# =====================================================
# CHURN RISK
# =====================================================

CHURN_ACTIONS = {

    "High": "Contact customer immediately and offer retention incentives.",

    "Medium": "Recommend personalized discounts and targeted campaigns.",

    "Low": "Continue normal engagement strategy.",

}


# =====================================================
# CLV
# =====================================================

CLV_ACTIONS = {

    "High": "Prioritize premium services and loyalty benefits.",

    "Medium": "Upsell complementary products and bundles.",

    "Low": "Increase engagement through personalized offers.",

}


# =====================================================
# ANOMALY
# =====================================================

ANOMALY_ACTIONS = {

    "Normal": "No action required.",

    "Anomaly": (
        "Investigate unusual purchasing behavior or possible fraud."
    ),

}
# =====================================================
# TOPIC DEFINITIONS
# =====================================================

TOPIC_MAPPING = {

    0: {
        "name": "Delivery Experience",
        "description": (
            "Customers discuss shipping, delivery speed, and order arrival."
        ),
        "business_action": (
            "Improve logistics and delivery communication."
        ),
    },

    1: {
        "name": "Product Quality",
        "description": (
            "Customers discuss product quality and durability."
        ),
        "business_action": (
            "Improve quality control and supplier management."
        ),
    },

    2: {
        "name": "Customer Service",
        "description": (
            "Customers discuss customer support experiences."
        ),
        "business_action": (
            "Improve customer support response time."
        ),
    },

    3: {
        "name": "Price & Value",
        "description": (
            "Customers discuss pricing and perceived value."
        ),
        "business_action": (
            "Optimize pricing and promotional campaigns."
        ),
    },

    4: {
        "name": "Overall Satisfaction",
        "description": (
            "General positive or negative customer satisfaction."
        ),
        "business_action": (
            "Maintain customer satisfaction and loyalty."
        ),
    },

}