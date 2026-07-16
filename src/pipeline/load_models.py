from pathlib import Path
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

    print("\nLoading trained models...\n")

    models = {}

    # ----------------------------------------
    # Segmentation
    # ----------------------------------------

    seg_dir = latest_experiment(
        "segmentation_kmeans"
    )

    models["segmentation"] = load_bundle(seg_dir)

    print("✓ Segmentation")

    # ----------------------------------------
    # Churn
    # ----------------------------------------

    churn_dir = latest_experiment(
        "repeat_logreg_adasyn"
    )

    models["churn"] = load_bundle(churn_dir)

    print("✓ Churn")

    # ----------------------------------------
    # CLV
    # ----------------------------------------

    clv_dir = latest_experiment(
        "clv_xgb_regressor"
    )

    models["clv"] = load_bundle(clv_dir)

    print("✓ CLV")

    # ----------------------------------------
    # Recommendation
    # ----------------------------------------

    rec_dir = latest_experiment(
        "recommendation_svd"
    )

    models["recommendation"] = load_bundle(rec_dir)

    print("✓ Recommendation")

    # ----------------------------------------
    # Anomaly
    # ----------------------------------------

    anomaly_dir = latest_experiment(
        "anomaly_isolation"
    )

    models["anomaly"] = load_bundle(anomaly_dir)

    print("✓ Anomaly Detection")

    # ----------------------------------------
    # Topic Modeling
    # ----------------------------------------

    topic_dir = latest_experiment(
        "topic_lda"
    )

    models["topic_model"] = load_bundle(topic_dir)

    print("✓ Topic Modeling")

    print("\n==============================")
    print("All models loaded successfully.")
    print("==============================")

    return models


def main():

    models = load_all_models()

    print()

    print(models.keys())


if __name__ == "__main__":
    main()