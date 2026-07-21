from pathlib import Path

from src.utils.logger import logger

import joblib

EXPERIMENTS_DIR = Path("artifacts/experiments")


def latest_experiment(prefix: str):
    """
    Returns the newest experiment folder whose name starts with prefix.
    """

    folders = sorted(
        EXPERIMENTS_DIR.glob(f"{prefix}*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not folders:
        raise FileNotFoundError(
            f"No experiment found for prefix '{prefix}'"
        )

    return folders[0]


def load_bundle(experiment_dir: Path):

    bundle_path = experiment_dir / "model_bundle.joblib"

    if not bundle_path.exists():
        raise FileNotFoundError(bundle_path)

    return joblib.load(bundle_path)


def load_all_models():

    logger.info("Loading trained models...")

    models = {}

    # ----------------------------------------
    # Segmentation
    # ----------------------------------------

    seg_dir = latest_experiment(
        "segmentation_kmeans"
    )

    models["segmentation"] = load_bundle(seg_dir)

    logger.info(" Segmentation model loaded successfully.")

    # ----------------------------------------
    # Churn
    # ----------------------------------------

    churn_dir = latest_experiment(
        "repeat_logreg_adasyn"
    )

    models["churn"] = load_bundle(churn_dir)

    logger.info(" Churn model loaded successfully.")

    # ----------------------------------------
    # CLV
    # ----------------------------------------

    clv_dir = latest_experiment(
        "clv_xgb_regressor"
    )

    models["clv"] = load_bundle(clv_dir)

    logger.info(" CLV model loaded successfully.")

    # ----------------------------------------
    # Recommendation
    # ----------------------------------------

    rec_dir = latest_experiment(
        "recommendation_svd"
    )

    models["recommendation"] = load_bundle(rec_dir)

    logger.info(" Recommendation model loaded successfully.")

    # ----------------------------------------
    # Anomaly
    # ----------------------------------------

    anomaly_dir = latest_experiment(
        "anomaly_isolation"
    )

    models["anomaly"] = load_bundle(anomaly_dir)

    logger.info(" Anomaly Detection model loaded successfully.")

    # ----------------------------------------
    # Topic Modeling
    # ----------------------------------------

    topic_dir = latest_experiment(
        "topic_lda"
    )

    models["topic_model"] = load_bundle(topic_dir)

    logger.info(" Topic Modeling model loaded successfully.")

    logger.info("All trained models loaded successfully.")

    return models


def main():

    models = load_all_models()

    print()

    print(models.keys())


if __name__ == "__main__":
    main()