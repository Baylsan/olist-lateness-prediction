import yaml
import joblib
import json
import sys

sys.path.append('src')
from preprocessing import preprocess_order

with open('config/config.yaml') as f:
    config = yaml.safe_load(f)

encoder = joblib.load(config['model']['encoder_path'])

with open(config['model']['results_path']) as f:
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

result = preprocess_order(sample_order, encoder, feature_columns)
print(result)

model = joblib.load(config['model']['path'])
prediction = model.predict_proba(result)
print(prediction)