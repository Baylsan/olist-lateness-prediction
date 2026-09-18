import sys
import json
import yaml
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "src"))
from preprocessing import preprocess_order


with open(BASE_DIR / "config" / "config.yaml") as f:
    config = yaml.safe_load(f)

encoder = joblib.load(BASE_DIR / config['model']['encoder_path'])
with open(BASE_DIR / config['model']['results_path']) as f:
    results = json.load(f)
feature_columns = results['feature_columns']

sample_order = {
    "num_items": 2,
    "total_freight": 25.90,
    "total_payment_value": 150.75,
    "order_purchase_timestamp": "2018-05-03 10:22:00",
    "order_approved_at": "2018-05-03 11:05:00",
    "order_estimated_delivery_date": "2018-05-20",
    "customer_state": "SP"
}


def test_preprocess_order_returns_correct_shape():
    result = preprocess_order(sample_order, encoder, feature_columns)
    assert result.shape == (1, 35)


def test_preprocess_order_column_order_matches_feature_columns():
    result = preprocess_order(sample_order, encoder, feature_columns)
    assert list(result.columns) == feature_columns


def test_preprocess_order_customer_state_encoded_correctly():
    result = preprocess_order(sample_order, encoder, feature_columns)
    assert result["customer_state_SP"].iloc[0] == 1.0
    assert result["customer_state_RJ"].iloc[0] == 0.0


def test_preprocess_order_log_transform_applied():
    result = preprocess_order(sample_order, encoder, feature_columns)
    import numpy as np
    expected = np.log1p(2)
    assert abs(result["num_items"].iloc[0] - expected) < 0.0001