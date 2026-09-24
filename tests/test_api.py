from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


sample_order = {
    "num_items": 2,
    "total_freight": 25.90,
    "total_payment_value": 150.75,
    "order_purchase_timestamp": "2018-05-03 10:22:00",
    "order_approved_at": "2018-05-03 11:05:00",
    "order_estimated_delivery_date": "2018-05-20",
    "customer_state": "SP"
}

def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_model_info():
    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert "model_type" in data
    assert "threshold" in data
    assert "num_features" in data

def test_predict():
    response = client.post(
        "/predict",
        json=sample_order
    )

    assert response.status_code == 200

    data = response.json()

    assert "is_late" in data
    assert "probability" in data
    assert "model_version" in data

    assert isinstance(data["is_late"], bool)
    assert 0 <= data["probability"] <= 1

def test_predict_rejects_invalid_order():
    invalid_order = sample_order.copy()
    invalid_order["num_items"] = 0

    response = client.post(
        "/predict",
        json=invalid_order
    )

    assert response.status_code == 422

def test_batch_predict():
    second_order = sample_order.copy()
    second_order["customer_state"] = "RJ"

    response = client.post(
        "/predict/batch",
        json=[
            sample_order,
            second_order
        ]
    )

    assert response.status_code == 200

    data = response.json()

    assert isinstance(data, list)
    assert len(data) == 2

    for prediction in data:
        assert "is_late" in prediction
        assert "probability" in prediction
        assert "model_version" in prediction

        assert isinstance(prediction["is_late"], bool)
        assert 0 <= prediction["probability"] <= 1

def test_batch_predict_rejects_invalid_order():
    invalid_order = sample_order.copy()
    invalid_order["num_items"] = 0

    response = client.post(
        "/predict/batch",
        json=[
            sample_order,
            invalid_order
        ]
    )

    assert response.status_code == 422

