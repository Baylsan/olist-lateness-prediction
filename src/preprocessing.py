import joblib
import json
import numpy as np 
import pandas as pd 
import logging
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


FEATURES = ['num_items', 'total_freight', 'total_payment_value',
            'approval_delay_hours', 'promised_delivery_days',
            'purchase_month', 'purchase_weekday', 'purchase_hour', 'purchase_year']

def preprocess_order(order, encoder, feature_columns):
    
    try:
        order_df= pd.DataFrame([order])

        date_cols=["order_approved_at", "order_purchase_timestamp", "order_estimated_delivery_date"]
        for col in date_cols:
            order_df[col] = pd.to_datetime(order_df[col], errors='coerce')
        
        order_df["purchase_month"] = order_df["order_purchase_timestamp"].dt.month
        order_df["purchase_weekday"] = order_df["order_purchase_timestamp"].dt.dayofweek
        order_df["purchase_hour"] = order_df["order_purchase_timestamp"].dt.hour
        order_df["purchase_year"] = order_df["order_purchase_timestamp"].dt.year

        order_df['approval_delay_hours'] = (order_df['order_approved_at'] - order_df['order_purchase_timestamp']).dt.total_seconds()/3600
        order_df['promised_delivery_days'] = (order_df['order_estimated_delivery_date'] - order_df['order_purchase_timestamp']).dt.days

        for col in ['num_items', 'total_freight', 'total_payment_value']:
            order_df[col] = np.log1p(order_df[col])

        state_encoded = encoder.transform(order_df[['customer_state']])

        state_cols = encoder.get_feature_names_out(['customer_state'])
        state_df = pd.DataFrame(state_encoded, columns=state_cols, index=order_df.index)   

        final_order = pd.concat([order_df[FEATURES], state_df], axis=1)
        final_order = final_order[feature_columns]
        
        logger.info(f"Order preprocessed successfully: {order}")

        return final_order

    except Exception as e:
        logger.error(f"Failed to preprocess order: {order} | Error: {e}")
        raise