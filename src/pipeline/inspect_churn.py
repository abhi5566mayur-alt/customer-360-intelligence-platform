import joblib

bundle = joblib.load(
    Path(
        "artifacts/experiments/repeat_logreg_adasyn_20260709_114151/model_bundle.joblib"
    )
)
print("\nBundle Keys:")
print(bundle.keys())

print()

for key, value in bundle.items():
    print("=" * 40)
    print(key)
    print(type(value))

    if hasattr(value, "shape"):
        print(value.shape)