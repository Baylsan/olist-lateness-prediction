from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import json
import sys
from pathlib import Path
import yaml
import time
import logging

logger = logging.getLogger("api")

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from preprocessing import preprocess_order

app = FastAPI(title="Order Lateness Prediction API")

BASE_DIR = Path(__file__).resolve().parent.parent
with open(BASE_DIR/"config"/"config.yaml") as f:
    config = yaml.safe_load(f)

encoder = joblib.load(BASE_DIR / config['model']['encoder_path'])
model = joblib.load(BASE_DIR/config['model']['path'])
with open(BASE_DIR/config['model']['results_path']) as f :
    result = json.load(f)

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
def pridect(order: OrderInput):
    start_time = time.time()

    order_dict = order.dict()
    processed = preprocess_order(order_dict, encoder, feature_columns)
    probability = model.predict_proba(processed)[0][1]
    is_late = bool(probability>= threshold)

    model_version = f"{result['validation_winner']} (threshold = {threshold})"

    latency = time.time() - start_time

    logger.info(
        f"input={order_dict} | is_late={is_late} | probability={probability:.4f} "
        f"| latency={latency:.4f}s | model_version={model_version}"
    )

    return{
        "is_late" : is_late,
        "probability": float(probability),
        "model_version": model_version
    }

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return {
        "model_type": results["validation_winner"],
        "threshold": threshold,
        "num_features": len(feature_columns),
        "validation_metrics": results["validation_metrics"],
        "test_metrics": results["test_metrics"]
    }
