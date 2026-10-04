import mlflow
import json
from pathlib import Path
import joblib
import mlflow.xgboost

BASE_DIR = Path(__file__).resolve().parent

with open(BASE_DIR / "models_artifacts" / "order_lateness_results.json") as f:
    results = json.load(f)

mlflow.set_experiment("order_lateness_prediction")

model = joblib.load(
    BASE_DIR / "models_artifacts" / "order_lateness_model.joblib"
)

with mlflow.start_run(run_name="xgboost_weighted_v1"):
    mlflow.log_params(results["hyperparameters"])
    mlflow.log_param("strategy", results["strategy"])
    mlflow.log_param("threshold", results["threshold"])

    mlflow.log_metrics({
        f"val_{k}": v for k, v in results["validation_metrics"].items()
        if isinstance(v, (int, float))
    })

    mlflow.log_metrics({
        f"test_{k}": v for k, v in results["test_metrics"].items()
        if isinstance(v, (int, float))
    })

    mlflow.xgboost.log_model(
        model,
        artifact_path="model",
        registered_model_name="order_lateness_model"
    )

    mlflow.log_artifact(
        str(BASE_DIR / "models_artifacts" / "encoder.joblib")
    )

print("MLflow run logged successfully.")