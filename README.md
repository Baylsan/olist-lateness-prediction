# Order Lateness Prediction — Inference Service

Production-oriented inference service for predicting e-commerce order delivery lateness.

This project is part of an MLOps training program (**Task 3: From Notebooks to Production**) and focuses on converting a notebook-based ML workflow into a reproducible inference service.

---

# Overview

The service receives raw order information and predicts whether the order is likely to be delivered late.

**Input:**

Raw order fields such as item count, freight, payment value, timestamps, and customer state.

**Output:**

- `is_late` — predicted lateness (`true` / `false`)
- `probability` — probability of lateness
- `model_version` — loaded model version

The XGBoost model and preprocessing artifacts were trained separately in Task 2 and are loaded by the inference service.

**No model training or fitting occurs at inference time.**

The project focuses on the inference and MLOps layer rather than model tuning.

---

# Architecture

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
MLflow Model Registry
    │
    │ production alias
    ▼
XGBoost Model
    │
    ▼
Prediction
    │
    ├── is_late
    ├── probability
    └── model_version
```

The service loads the registered model, encoder, feature configuration, and prediction threshold once during application startup.

---

# Tech Stack

| Component | Technology |
|---|---|
| API | FastAPI |
| Model | XGBoost |
| Preprocessing | Python / Pandas / Scikit-learn |
| Data Validation | Great Expectations |
| Experiment Tracking | MLflow |
| Model Registry | MLflow Model Registry |
| Testing | Pytest |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Configuration | YAML |
| Data Versioning | DVC evaluated during development |

---

# Project Structure

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
│       └── ci.yml               # Automated testing and MLflow registration
│
├── logs/                        # Runtime logs (gitignored)
├── mlflow_log.py                # MLflow experiment/model logging
├── Dockerfile
├── requirements.txt
└── README.md
```

The original training notebooks (NB1–NB6) belong to Task 2 and are not duplicated in this repository.

This repository contains the inference service and its MLOps infrastructure.

The local model artifact is retained as the source used by the MLflow logging and registration workflow, while the running inference service loads the model from the MLflow Model Registry.

---

# Running Locally

## 1. Create the environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2. Ensure the MLflow Model Registry is available

The inference service loads the model using the MLflow `production` alias.

The registered model must exist under:

```text
order_lateness_model
```

and the `production` alias must point to the model version used by the service.

To inspect the local MLflow tracking environment:

```bash
mlflow ui
```

The MLflow UI is available at:

```text
http://127.0.0.1:5000
```

## 3. Start the API

```bash
uvicorn app.main:app --reload
```

The API runs at:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

---

# API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/predict` | Validate, preprocess, and predict a single order |
| `POST` | `/predict/batch` | Predict multiple orders |
| `GET` | `/health` | Service health check |
| `GET` | `/model-info` | Model and training information |
| `GET` | `/metrics` | Runtime service metrics |

The request and response schemas are documented automatically through FastAPI at `/docs`.

---

# Example Request

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

- output structure
- feature column ordering
- categorical encoding
- numerical transformations
- API prediction behavior

Tests use the actual trained artifacts rather than mocked models.

Run the test suite with:

```bash
pytest -v
```

The same test suite is executed automatically by GitHub Actions on repository pushes and pull requests.

---

# Data Validation — Great Expectations

Incoming requests are validated before they reach preprocessing and model inference.

Current validation rules include:

- numeric fields must satisfy defined sanity constraints
- item counts cannot be negative
- financial values must satisfy defined ranges
- `customer_state` must belong to the Brazilian state codes represented in the training data

Invalid requests are rejected before prediction, preventing invalid values or unknown categories from silently reaching the model.

---

# Experiment Tracking and Model Registry — MLflow

MLflow is used to track the training run information produced by Task 2 and to register the trained XGBoost model.

The logging script records relevant:

- parameters
- validation metrics
- test metrics
- training results
- model artifacts

Run locally with:

```bash
python3 mlflow_log.py
```

The registered model is:

```text
order_lateness_model
```

The model is registered in the **MLflow Model Registry**, creating versioned model entries.

The CI workflow also automatically:

1. registers the model
2. discovers the latest registered model version
3. assigns that version to the `production` alias

This makes the CI pipeline responsible for keeping the production alias aligned with the latest registered model version.

The inference service loads the XGBoost model from the MLflow Model Registry using the `production` alias:

```text
models:/order_lateness_model@production
```

The `production` alias provides a stable deployment reference while allowing the underlying model version to change without modifying the application code.

The local MLflow UI can be started with:

```bash
mlflow ui
```

and accessed at:

```text
http://127.0.0.1:5000
```

Local MLflow databases and generated run directories are excluded from Git.

---

# Data Versioning — DVC

DVC was evaluated during the project as a mechanism for versioning model artifacts independently from Git.

The initial approach used a Google Drive DVC remote. The remote was configured successfully and the DVC Google Drive plugin was installed.

However, the first authentication attempt failed because Google blocked the default DVC OAuth application with:

```text
This app is blocked
This app tried to access sensitive info in your Google Account.
```

This is a known issue documented by DVC for its Google Drive integration.

The project therefore reverted to storing the relatively small inference artifacts directly in Git:

```text
models_artifacts/order_lateness_model.joblib
models_artifacts/encoder.joblib
```

This was a deliberate engineering decision to keep the project simple and ensure that the CI environment can obtain all artifacts through a normal Git checkout without requiring external DVC authentication.

The local model artifact is also used by `mlflow_log.py` as the source for model registration.

For substantially larger datasets or model artifacts, a remote DVC backend would be more appropriate.

---

# Monitoring

## Service Metrics

The `/metrics` endpoint exposes runtime metrics including:

- `total_requests`
- `successful_predictions`
- `validation_rejections`
- `errors`
- `average_latency_seconds`

Validation rejections are tracked separately from unexpected application errors because invalid user input is an expected API behavior rather than necessarily a system failure.

## Prediction Logging

Prediction requests are logged to:

```text
logs/app.log
```

The logs contain information such as:

- prediction result
- probability
- latency
- model version

These logs provide a foundation for future monitoring of model performance and data drift when actual delivery outcomes become available.

---

# CI/CD

GitHub Actions automatically runs the project checks when changes are pushed to the repository or when a pull request is opened.

The current workflow:

1. checks out the repository
2. sets up Python 3.10
3. installs project dependencies
4. registers the model in MLflow
5. assigns the latest registered model version to the `production` alias
6. runs the automated test suite

The workflow is designed to ensure that changes do not break the inference service or its MLOps workflow.

---

# Docker

The project includes a Dockerfile intended to package the inference service and its runtime dependencies.

The Docker build was tested locally and successfully reached the dependency installation stage.

The current build is limited by a network problem while downloading the large XGBoost wheel from PyPI:

```text
xgboost-3.2.0-py3-none-manylinux_2_28_x86_64.whl
131.7 MB
```

The download from `files.pythonhosted.org` was extremely slow under the current network conditions.

An initial build attempt ended with a `ReadTimeoutError`.

A later build attempt downloaded only part of the wheel before pip reported a SHA256 mismatch, indicating that the package received by Docker was incomplete or corrupted during transfer.

The same XGBoost version was downloaded successfully outside Docker and its SHA256 checksum matched the expected package checksum.

Therefore, the current Docker limitation is related to the dependency download environment rather than a demonstrated application-level failure.

The Docker build will be revisited once a stable dependency download path is available.

---

# Known Limitations

## XGBoost Version

The model was originally trained using `xgboost==3.4.1` in Google Colab.

The current inference environment uses:

```text
xgboost==3.2.0
```

The model loads successfully and produces consistent predictions in the current local inference environment, but exact bitwise reproducibility across different XGBoost versions is not guaranteed.

## Docker Build

The Dockerfile is present and configured, but a complete image build has not yet been achieved because downloading the 131.7 MB XGBoost wheel from PyPI repeatedly fails under the current network conditions.

## MLflow UI

MLflow tracking and model registration work locally.

Accessing the MLflow dashboard through a Windows browser from WSL2 may require additional network configuration.

## DVC Remote

DVC was evaluated but a cloud remote was not retained in the final project because the Google Drive OAuth flow was blocked and the current model artifacts are small enough to remain in Git.

---

# Key Design Decisions

## Stateless Inference

The API accepts the fields required by the model directly rather than querying PostgreSQL during inference.

This keeps the inference service stateless and independently deployable.

## Pre-loaded Artifacts

The MLflow-registered model, encoder, threshold, and feature configuration are loaded once during application startup rather than for every request.

This avoids unnecessary disk I/O during inference.

## Single Source of Truth

The prediction threshold and feature columns are loaded from the stored model results rather than duplicated across configuration files.

## Shared Preprocessing Logic

Single-order preprocessing is implemented as the atomic preprocessing operation, while batch prediction reuses the same logic rather than maintaining a separate preprocessing implementation.

## Separation of Training and Inference

Model training remains in the Task 2 notebooks.

This repository contains the inference layer and the MLOps infrastructure required to serve the already-trained model.

## CI-driven Model Registration

Model registration and production alias assignment are integrated into the CI workflow so that model versions can be tracked consistently alongside code changes.

---

# Future Improvements

Potential production extensions include:

- completing and validating the Docker image build
- deploying the Docker image to a production environment
- remote DVC storage for larger datasets and artifacts
- automated model monitoring
- automated data-drift detection
- alerting based on service metrics
- periodic model evaluation using newly observed delivery outcomes
- production-grade observability and centralized logging