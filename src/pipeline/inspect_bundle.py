from pathlib import Path
import joblib


# =====================================================
# CHANGE THIS PATH TO ANY MODEL BUNDLE
# =====================================================

MODEL_BUNDLE = Path(
    "artifacts/experiments/clv_rf_logtarget_20260716_144550/model_bundle.joblib"
)
# Examples:
#
# "artifacts/experiments/segmentation_kmeans_20260709_120046/segmentation_model_bundle.joblib"
#
# "artifacts/experiments/clv_rf_logtarget_20260716_144550/model_bundle.joblib"
#
# "artifacts/experiments/anomaly_isolation_20260711_122610/model_bundle.joblib"
#
# "artifacts/experiments/recommendation_svd_20260716_142544/recommendation_model_bundle.joblib"
#
# "artifacts/experiments/topic_lda_20260715_130121/model_bundle.joblib"


def inspect_object(obj, indent=0):

    space = " " * indent

    print(f"{space}Type : {type(obj)}")

    if hasattr(obj, "shape"):
        print(f"{space}Shape: {obj.shape}")

    if isinstance(obj, dict):

        print(f"{space}Keys:")

        for k in obj.keys():
            print(f"{space}  - {k}")

    elif isinstance(obj, list):

        print(f"{space}Length: {len(obj)}")

        if len(obj) > 0:
            print(f"{space}First element type: {type(obj[0])}")

    elif isinstance(obj, tuple):

        print(f"{space}Length: {len(obj)}")

    elif isinstance(obj, str):

        print(f"{space}{obj}")

    elif isinstance(obj, int):

        print(f"{space}{obj}")

    elif isinstance(obj, float):

        print(f"{space}{obj}")


def main():

    print("\nLoading bundle...\n")

    bundle = joblib.load(MODEL_BUNDLE)

    print("Bundle Path:")
    print(MODEL_BUNDLE)

    print("\n")

    if isinstance(bundle, dict):

        print("Bundle Keys:")
        print(bundle.keys())

        print("\nDetailed Information")

        for key, value in bundle.items():

            print("\n" + "=" * 60)

            print(f"KEY : {key}")

            inspect_object(value, indent=4)

    else:

        print("Bundle is not a dictionary.")

        inspect_object(bundle)

    print("\n" + "=" * 60)

    print("INSPECTION COMPLETED")

    print("=" * 60)


if __name__ == "__main__":
    main()