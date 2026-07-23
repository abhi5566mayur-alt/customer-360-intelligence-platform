import streamlit as st
import requests

from frontend.customer_form import customer_feature_form

API_URL = "http://127.0.0.1:8000/api/v1/predict/churn"

st.set_page_config(
    page_title="Churn Prediction",
    page_icon="🔄",
    layout="wide",
)

st.title("🔄 Customer Churn Prediction")

st.write(
    """
Predict whether a customer is likely to churn using the trained
Machine Learning model.
"""
)

st.markdown("---")

# ============================
# Customer Input Form
# ============================

customer_features, explain = customer_feature_form()

st.markdown("---")

predict = st.button(
    "🔄 Predict Churn",
    use_container_width=True,
)

# ============================
# Prediction
# ============================

if predict:

    payload = {
        "customer_features": customer_features,
        "explain": explain,
    }

    try:

        with st.spinner("Predicting..."):

            response = requests.post(
                API_URL,
                json=payload,
                timeout=30,
            )
            

        if response.status_code == 200:

            result = response.json()

            prediction = result["prediction"]["will_churn"]
            probability = result["prediction"]["churn_probability"]
            risk = result["prediction"]["risk"]
            business_action = result["prediction"]["business_action"]

            st.success("Prediction Completed")

            col1, col2 = st.columns(2)

            with col1:

                if prediction:
                    st.error("⚠ Customer is likely to Churn")
                else:
                    st.success("✅ Customer is likely to Stay")
            with col2:
                st.metric(
                    "Churn Probability",
                    f"{probability:.2%}",
                )

            # ------------------------
            # Risk Level
            # ------------------------

            st.subheader("Risk Assessment")
            st.write(f"**Risk Level:** {risk}")

            st.info(f"**Recommended Action:** {business_action}")
            # ------------------------
            # Raw Response
            # ------------------------

            with st.expander("View API Response"):
                st.json(result)

        else:

            st.error(f"API Error : {response.status_code}")
            st.json(response.json())

    except Exception as e:
        st.error(str(e))