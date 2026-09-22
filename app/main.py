from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import json
import sys
from pathlib import Path
import yaml
import time
import logging
from fastapi import HTTPException

logger = logging.getLogger("api")

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from preprocessing import preprocess_order
from validation import validate_order

app = FastAPI(title="Order Lateness Prediction API")

BASE_DIR = Path(__file__).resolve().parent.parent
with open(BASE_DIR / "config" / "config.yaml") as f:
    config = yaml.safe_load(f)

encoder = joblib.load(BASE_DIR / config['model']['encoder_path'])
model = joblib.load(BASE_DIR / config['model']['path'])
with open(BASE_DIR / config['model']['results_path']) as f:
    result = json.load(f)

metrics = {
    "total_requests": 0,
    "successful_predictions": 0,
    "validation_rejections": 0,
    "errors": 0,
    "total_latency": 0.0
}

feature_columns = result['feature_columns']
threshold = result['threshold']


class OrderInput(BaseModel):
    num_items: int
    total_freight: float
    total_payment_value: float
    order_purchase_timestamp: str
    order_approved_at: str
    order_estimated_delivery_date: str
    customer_state: str


@app.post("/predict")
def predict(order: OrderInput):
    metrics["total_requests"] += 1
    order_dict = order.dict()

    failed = validate_order(order_dict)
    if failed:
        metrics["validation_rejections"] += 1
        raise HTTPException(
            status_code=422,
            detail=f"Validation failed for columns: {failed}"
        )

    try:
        start_time = time.time()

        processed = preprocess_order(order_dict, encoder, feature_columns)
        probability = model.predict_proba(processed)[0][1]
        is_late = bool(probability >= threshold)

        model_version = f"{result['validation_winner']} (threshold = {threshold})"

        latency = time.time() - start_time

        metrics["successful_predictions"] += 1
        metrics["total_latency"] += latency

        logger.info(
            f"input={order_dict} | is_late={is_late} | probability={probability:.4f} "
            f"| latency={latency:.4f}s | model_version={model_version}"
        )

        return {
            "is_late": is_late,
            "probability": float(probability),
            "model_version": model_version
        }

    except Exception as e:
        metrics["errors"] += 1
        logger.error(f"Prediction failed: {e}")
        raise


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return {
        "model_type": result["validation_winner"],
        "threshold": threshold,
        "num_features": len(feature_columns),
        "validation_metrics": result["validation_metrics"],
        "test_metrics": result["test_metrics"]
    }


@app.get("/metrics")
def get_metrics():
    avg_latency = (
        metrics["total_latency"] / metrics["successful_predictions"]
        if metrics["successful_predictions"] > 0 else 0
    )
    return {
        "total_requests": metrics["total_requests"],
        "successful_predictions": metrics["successful_predictions"],
        "validation_rejections": metrics["validation_rejections"],
        "errors": metrics["errors"],
        "average_latency_seconds": round(avg_latency, 4)
    }