import streamlit as st
from frontend.api_client import check_api

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Customer360 Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------
# API Status
# -------------------------------------------------
api_online = check_api()

if api_online:
    api_status = "🟢 Online"
else:
    api_status = "🔴 Offline"

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
st.sidebar.title("📊 Customer360")

st.sidebar.markdown("---")

st.sidebar.info(
    """
Welcome to Customer360!

Use the pages below to access:

- 🏠 Home
- 🔄 Churn Prediction
- 💰 CLV Prediction
- 👥 Customer Segmentation
- 🛒 Product Recommendation
- 🚨 Anomaly Detection
- 💬 Topic Modeling
- 🔍 SHAP Explainability
"""
)

# -------------------------------------------------
# Header
# -------------------------------------------------
st.title("📊 Customer360 Intelligence Platform")

st.caption(
    "AI-Powered Customer Analytics using Classical Machine Learning"
)

st.success(
    "Welcome! Explore each Machine Learning module using the navigation menu on the left."
)

st.markdown("---")

# -------------------------------------------------
# Dashboard Metrics
# -------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("ML Models", "6")

with col2:
    st.metric("API Status", api_status)

with col3:
    st.metric("Backend", "FastAPI")

with col4:
    st.metric("Frontend", "Streamlit")

st.markdown("---")

# -------------------------------------------------
# Project Overview
# -------------------------------------------------
st.subheader("📖 Project Overview")

st.write(
    """
Customer360 Intelligence Platform is a complete end-to-end Machine Learning
application designed to help businesses understand customer behavior and make
data-driven decisions.

The platform integrates multiple Machine Learning models through a FastAPI
backend and provides an interactive Streamlit dashboard for predictions and
analytics.
"""
)

st.markdown("---")

# -------------------------------------------------
# Available AI Modules
# -------------------------------------------------
st.subheader("🤖 Available AI Modules")

col1, col2 = st.columns(2)

with col1:
    st.info("🔄 Customer Churn Prediction")
    st.info("👥 Customer Segmentation")
    st.info("🚨 Anomaly Detection")

with col2:
    st.info("💰 Customer Lifetime Value Prediction")
    st.info("🛒 Product Recommendation")
    st.info("💬 Review Topic Modeling")

st.info("🔍 SHAP Explainability")

st.markdown("---")

# -------------------------------------------------
# Technology Stack
# -------------------------------------------------
st.subheader("🛠 Technology Stack")

tech1, tech2, tech3 = st.columns(3)

with tech1:
    st.markdown(
        """
### Backend
- FastAPI
- Uvicorn
- Pydantic
"""
    )

with tech2:
    st.markdown(
        """
### Machine Learning
- Scikit-Learn
- XGBoost
- SHAP
- Pandas
"""
    )

with tech3:
    st.markdown(
        """
### Frontend
- Streamlit
- Plotly
- Requests
"""
    )

st.markdown("---")

# -------------------------------------------------
# System Architecture
# -------------------------------------------------
st.subheader("🏗 System Architecture")

st.code(
"""
          Streamlit Frontend
                  │
                  ▼
            FastAPI Backend
                  │
                  ▼
       Machine Learning Models
                  │
                  ▼
          Prediction Results
""",
language="text",
)

st.markdown("---")

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.caption(
    "Customer360 Intelligence Platform | Version 1.0 | Built with FastAPI + Streamlit"
)