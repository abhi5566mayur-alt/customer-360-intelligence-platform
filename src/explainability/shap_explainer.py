import shap
import numpy as np
import pandas as pd

from src.utils.logger import logger

from typing import Any

class ShapExplainer:

    def __init__(
        self,
        models: dict[str, Any],
    ) -> None:
        """
        Initialize the SHAP explainer for all supported models.

        Parameters
        ----------
        models : dict
            Dictionary containing all trained models and their
            associated metadata.
        """

        self.models = models

        self.churn_explainer = None
        self.clv_explainer = None
        self.anomaly_explainer = None

        self._load_explainers()

    # =====================================================
    # LOAD SHAP EXPLAINERS
    # =====================================================

    def _load_explainers(self) -> None:
        """
        Initialize SHAP explainers for all supported models.
        """

        try:
            logger.info("Initializing SHAP explainers...")

            churn_bundle = self.models["churn"]

            self.churn_explainer = shap.LinearExplainer(
                churn_bundle["model"],
                np.zeros((1, len(churn_bundle["feature_cols"]))),
            )

            clv_bundle = self.models["clv"]

            self.clv_explainer = shap.TreeExplainer(
                clv_bundle["model"]
            )

            anomaly_bundle = self.models["anomaly"]

            self.anomaly_explainer = shap.TreeExplainer(
                anomaly_bundle["model"]
            )

            logger.info("SHAP explainers initialized successfully.")

        except Exception:
            logger.exception(
                "Failed to initialize SHAP explainers."
            )
            raise

    # =====================================================
    # TOP FEATURES
    # =====================================================

    def _top_features(
        self,
        shap_values: Any,
        feature_names: list[str],
        top_n: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Extract the top contributing features from SHAP values.

        Parameters
        ----------
        shap_values : array-like
            SHAP values returned by the explainer.

        feature_names : list
            Ordered list of feature names.

        top_n : int, default=5
            Number of top features to return.

        Returns
        -------
        list
            List of dictionaries containing feature names,
            SHAP impacts, and importance values.
        """

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        shap_values = np.asarray(shap_values)

        if shap_values.ndim == 3:
            shap_values = shap_values[0, :, 0]
        elif shap_values.ndim == 2:
            shap_values = shap_values[0]
        elif shap_values.ndim == 1:
            pass
        else:
            raise ValueError(
                f"Unsupported SHAP value shape: {shap_values.shape}"
            )

        importance = np.abs(shap_values)

        indices = np.argsort(importance)[::-1][:top_n]

        results: list[dict[str, Any]] = []

        for idx in indices:
            results.append(
                {
                    "feature": feature_names[idx],
                    "impact": round(float(shap_values[idx]), 4),
                    "importance": round(float(importance[idx]), 4),
                }
            )

        return results
    # =====================================================
    # CHURN EXPLANATION
    # =====================================================

    def explain_churn(
        self,
        customer_features: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Generate a SHAP explanation for a churn prediction.

        Parameters
        ----------
        customer_features : dict
            Customer feature dictionary.

        Returns
        -------
        dict
            SHAP explanation including the top contributing
            features for the churn prediction.
        """
        try:

            logger.info("Generating SHAP explanation for churn prediction.")

            bundle = self.models["churn"]

            feature_cols = bundle["feature_cols"]

            scaler = bundle["scaler"]

            df = pd.DataFrame([customer_features])

            for col in feature_cols:
                if col not in df.columns:
                    df[col] = 0

            df = df[feature_cols]

            X = scaler.transform(df)

            shap_values = self.churn_explainer.shap_values(X)

            logger.info("Successfully generated churn SHAP explanation.")

            return {
                "method": "SHAP",
                "model": "Churn",
                "top_features": self._top_features(
                    shap_values,
                    feature_cols,
                ),
            }

        except Exception:
                    
            logger.exception(
                "Failed to generate churn SHAP explanation."
            )
            raise
    # =====================================================
    # CLV EXPLANATION
    # =====================================================
    def explain_clv( 
        self,
        customer_features: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Generate a SHAP explanation for a customer lifetime value prediction.

        Parameters
        ----------
        customer_features : dict
            Customer feature dictionary.

        Returns
        -------
        dict
            SHAP explanation including the top contributing
            features for the CLV prediction.
        """
        try:

            logger.info("Generating SHAP explanation for CLV prediction.")

            bundle = self.models["clv"]

            feature_cols = bundle["feature_names"]

            df = pd.DataFrame([customer_features])

            for col in feature_cols:
                if col not in df.columns:
                    df[col] = 0

            df = df[feature_cols]

            shap_values = self.clv_explainer.shap_values(df)

            logger.info("Successfully generated CLV SHAP explanation.")

            return {
                "method": "SHAP",
                "model": "CLV",
                "top_features": self._top_features(
                    shap_values,
                    feature_cols,
                ),
            }
        except Exception:   
            logger.exception(
                "Failed to generate CLV SHAP explanation."
            )
            raise
            
       
    # =====================================================
    # ANOMALY EXPLANATION
    # =====================================================

    def explain_anomaly(
        self,
        customer_features: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Generate a SHAP explanation for anomaly detection.

        Parameters
        ----------
        customer_features : dict
            Customer feature dictionary.

        Returns
        -------
        dict
            SHAP explanation including the top contributing
            features responsible for the anomaly score.
        """
        try:
            logger.info("Generating SHAP explanation for anomaly detection.")

            bundle = self.models["anomaly"]

            feature_cols = bundle["feature_cols"]

            scaler = bundle["scaler"]

            df = pd.DataFrame([customer_features])

            for col in feature_cols:
                if col not in df.columns:
                    df[col] = 0

            df = df[feature_cols]

            X = scaler.transform(df)

            shap_values = self.anomaly_explainer.shap_values(X)
            logger.info("Successfully generated anomaly SHAP explanation.")
            return {
                "method": "SHAP",
                "model": "Anomaly",
                "top_features": self._top_features(
                    shap_values,
                    feature_cols,
                ),
            }
        except Exception:
            logger.exception(
                "Failed to generate anomaly SHAP explanation."
            )
            raise

if __name__ == "__main__":

    

    from pprint import pprint
    from src.pipeline.load_models import load_all_models

    models = load_all_models()

    explainer = ShapExplainer(models)

    features = {}

    for col in models["anomaly"]["feature_cols"]:
        features[col] = 1.0

    explanation = explainer.explain_anomaly(features)

    pprint(explanation)