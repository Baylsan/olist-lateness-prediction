# Order Lateness Prediction — Inference Service

Production-oriented inference service for predicting e-commerce order delivery lateness using **FastAPI, XGBoost, MLflow, DVC, Great Expectations, Docker, and GitHub Actions**.

This project is part of an MLOps training program (**Task 3: From Notebooks to Production**) and focuses on converting a notebook-based ML workflow into a reproducible inference service.

**Repository:** https://github.com/Baylsan/olist-lateness-prediction

---

## Overview

The service receives raw order information and predicts whether the order is likely to be delivered late.

**Input:** raw order fields such as item count, freight, payment value, timestamps, and customer state.

**Output:**

* `is_late` — predicted lateness (`true` / `false`)
* `probability` — probability of lateness
* `model_version` — loaded model version

The XGBoost model and preprocessing artifacts were trained separately in Task 2 and are loaded by the inference service. **No model training or fitting occurs at inference time.**

The project focuses on the inference and MLOps layer rather than model tuning.

---

## Architecture

```text
Raw Order
    │
    ▼
Great Expectations
    │
    │ valid
    ▼
Preprocessing
    │
    ▼
Trained XGBoost Model
    │
    ▼
Prediction
    │
    ├── is_late
    ├── probability
    └── model_version
```

The service loads the trained model, encoder, feature configuration, and prediction threshold once during application startup.

---

## Tech Stack

| Component           | Technology                     |
| ------------------- | ------------------------------ |
| API                 | FastAPI                        |
| Model               | XGBoost                        |
| Preprocessing       | Python / Pandas / Scikit-learn |
| Data Validation     | Great Expectations             |
| Experiment Tracking | MLflow                         |
| Data Versioning     | DVC                            |
| Testing             | Pytest                         |
| Containerization    | Docker                         |
| CI/CD               | GitHub Actions                 |
| Configuration       | YAML                           |

---

## Project Structure

```text
task3/
├── app/
│   └── main.py                  # FastAPI application and endpoints
│
├── src/
│   ├── preprocessing.py         # Raw order → model-ready features
│   └── validation.py            # Input data validation
│
├── config/
│   └── config.yaml              # Artifact and application configuration
│
├── models_artifacts/
│   ├── order_lateness_model.joblib
│   ├── encoder.joblib
│   └── order_lateness_results.json
│
├── tests/
│   ├── test_preprocessing.py
│   └── test_preprocessing_manual.py
│
├── .github/
│   └── workflows/
│       └── ci.yml               # Automated tests on push
│
├── logs/                        # Runtime logs (gitignored)
├── mlflow_log.py                # MLflow experiment logging
├── Dockerfile
├── requirements.txt
└── README.md
```

The original training notebooks (NB1–NB6) belong to Task 2 and are not duplicated in this repository. This repository contains the inference service and its MLOps infrastructure.

---

# Running Locally

## 1. Create the environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2. Start the API

```bash
uvicorn app.main:app --reload
```

The API runs at:

```text
http://127.0.0.1:8000
```

Interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

---

# API Endpoints

| Method | Endpoint         | Purpose                                          |
| ------ | ---------------- | ------------------------------------------------ |
| `POST` | `/predict`       | Validate, preprocess, and predict a single order |
| `POST` | `/predict/batch` | Predict multiple orders                          |
| `GET`  | `/health`        | Service health check                             |
| `GET`  | `/model-info`    | Model and training information                   |
| `GET`  | `/metrics`       | Runtime service metrics                          |

The request and response schemas are documented automatically through FastAPI at `/docs`.

---

## Example Request

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "num_items": 2,
    "total_freight": 25.90,
    "total_payment_value": 150.75,
    "order_purchase_timestamp": "2018-05-03 10:22:00",
    "order_approved_at": "2018-05-03 11:05:00",
    "order_estimated_delivery_date": "2018-05-20",
    "customer_state": "SP"
  }'
```

---

# Testing

The project uses **pytest** for automated testing.

Tests cover the preprocessing and inference pipeline, including:

* output structure
* feature column ordering
* categorical encoding
* numerical transformations
* API prediction behavior

Tests use the actual trained artifacts rather than mocked models.

Run the test suite with:

```bash
pytest -v
```

The same test suite is executed automatically by GitHub Actions on repository pushes.

---

# Data Validation — Great Expectations

Incoming requests are validated before they reach preprocessing and model inference.

Current validation rules include:

* numeric fields must satisfy defined sanity constraints
* item counts cannot be negative
* financial values must satisfy defined ranges
* `customer_state` must belong to the Brazilian state codes represented in the training data

Invalid requests are rejected before prediction, preventing invalid values or unknown categories from silently reaching the model.

---

# Data Versioning — DVC

DVC was initialized as part of the MLOps workflow and used to explore versioning of model artifacts and datasets independently from Git.

For the final version of this project, the relatively small inference artifacts are stored directly in Git:

* `order_lateness_model.joblib`
* `encoder.joblib`

This was a deliberate design choice because these artifacts are small enough to be practical in Git and the project does not currently use a remote DVC storage backend.

For larger datasets or models, a remote DVC backend such as S3 or Google Drive would be more appropriate.

---

# Experiment Tracking — MLflow

Training results from Task 2 can be logged to MLflow using:

```bash
python3 mlflow_log.py
```

The script reads the stored training results and records relevant parameters, metrics, and model artifacts.

The local MLflow UI can be started with:

```bash
mlflow ui
```

and accessed at:

```text
http://127.0.0.1:5000
```

Local MLflow databases and run directories are excluded from Git because they are generated runtime data.

---

# Monitoring

## Service Metrics

The `/metrics` endpoint exposes runtime metrics including:

* `total_requests`
* `successful_predictions`
* `validation_rejections`
* `errors`
* `average_latency_seconds`

Validation rejections are tracked separately from unexpected application errors because invalid user input is an expected API behavior rather than necessarily a system failure.

## Prediction Logging

Prediction requests are logged to:

```text
logs/app.log
```

The logs contain information such as the prediction result, probability, latency, and model version.

These logs provide a foundation for future monitoring of model performance and data drift when actual delivery outcomes become available.

---

# CI/CD

GitHub Actions automatically runs the test suite when changes are pushed to the repository.

The workflow verifies that the inference service and its automated tests remain functional after changes.

---

# Docker

The application includes a Dockerfile intended to package the inference service and its runtime dependencies.

A local Docker build was attempted successfully through the Dockerfile stages until dependency installation.

The build is currently blocked by a **network timeout while downloading the XGBoost wheel from PyPI**:

```text
xgboost-3.2.0-py3-none-manylinux_2_28_x86_64.whl
131.7 MB
```

During the latest build attempt, Docker downloaded approximately 5.7 MB at around 5.9 kB/s before the connection timed out after approximately 12 minutes:

```text
pip._vendor.urllib3.exceptions.ReadTimeoutError:
HTTPSConnectionPool(host='files.pythonhosted.org', port=443):
Read timed out.
```

The same `xgboost==3.2.0` package is already installed and works correctly in the local Python environment.

Therefore, the current Docker limitation is related to the **network transfer of the large XGBoost package during image construction**, rather than an application-level Dockerfile error.

Dockerization can be completed later when a sufficiently stable connection is available.

---

# Known Limitations

### XGBoost Version

The model was originally trained using `xgboost==3.4.1` in Google Colab. The inference environment currently uses `xgboost==3.2.0`.

The model loads successfully and produces consistent predictions in the local inference environment, but exact bitwise reproducibility across different XGBoost versions is not guaranteed.

### Docker Build

The Dockerfile is present and configured, but the image has not yet been successfully built because downloading the 131.7 MB XGBoost wheel from PyPI repeatedly times out under the current network conditions.

This is an environment/network limitation rather than a demonstrated application failure.

### MLflow UI

MLflow tracking works locally and the run data is stored successfully. Accessing the MLflow dashboard through a Windows browser from WSL2 may require additional network configuration.

---

# Key Design Decisions

### Stateless Inference

The API accepts the fields required by the model directly rather than querying PostgreSQL during inference.

This keeps the inference service stateless and independently deployable.

### Pre-loaded Artifacts

The model, encoder, threshold, and feature configuration are loaded once during application startup rather than for every request.

This avoids unnecessary disk I/O during inference.

### Single Source of Truth

The prediction threshold and feature columns are loaded from the stored model results rather than duplicated across configuration files.

### Shared Preprocessing Logic

Single-order preprocessing is implemented as the atomic preprocessing operation, while batch prediction reuses the same logic rather than maintaining a separate preprocessing implementation.

### Separation of Training and Inference

Model training remains in the Task 2 notebooks.

This repository contains the inference layer and MLOps infrastructure required to serve the already-trained model.

---

# Future Improvements

Potential production extensions include:

* automated Docker image builds and deployment
* remote DVC storage
* MLflow Model Registry
* automated model monitoring
* automated data-drift detection
* alerting based on service metrics
* periodic model evaluation using newly observed delivery outcomes
* production-grade observability and centralized logging
