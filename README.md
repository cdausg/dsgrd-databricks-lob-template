# {{cookiecutter.lob_display_name}} - {{cookiecutter.project_slug}}

This is the official blueprint template for creating new Databricks projects in the hub and spoke platform.

## Project Overview
<!-- Describe your use case here -->

## Architecture
This project follows the standard hub and spoke MLops architecture:
```
Feature Engineering (DLT)
        ↓
Model Training (MLflow + Unity Catalog)
        ↓
Model Evaluation (Champion/Challenger)
        ↓
Batch Inference (DLT) / Model Serving
        ↓
Result Layer (Gold tables → BI tools)
```

## Model Archetypes
This project supports the following model archetypes — delete the ones you don't need:
- **Classification** — binary or multi-class prediction
- **Time Series** — forecasting and demand planning
- **Clustering** — segmentation and anomaly detection

## Getting Started

### Prerequisites
- Databricks CLI installed
- Access to `{{cookiecutter.workspace_url}}`
- Unity Catalog access to `{{cookiecutter.catalog_name}}`

### Deploy to dev
```bash
databricks bundle deploy --target dev
```

### Run the pipeline
```bash
databricks bundle run feature_pipeline
databricks bundle run training_job
databricks bundle run inference_pipeline
```

## Project Structure
- `resources/` — Databricks Asset Bundle resource definitions (jobs, pipelines)
- `src/` — Python source code
  - `features/` — DLT feature engineering pipeline
  - `training/` — Model training scripts per archetype
  - `evaluation/` — Model evaluation and champion/challenger logic
  - `inference/` — Batch inference pipeline
  - `serving/` — Model serving endpoint deployment
  - `monitoring/` — Lakehouse Monitoring setup
- `tests/` — Unit tests
- `notebooks/exploratory/` — Ad-hoc exploration notebooks
- `docs/` — EU AI Act documentation templates

## Monitoring
This project uses Databricks Lakehouse Monitoring for standard platform monitoring
(data drift, model drift). Ground truth monitoring is the responsibility of the
project team and is out of scope for the central platform.

## EU AI Act
See `docs/risk_classification.md` to determine the risk tier of your use case.
High-risk systems require additional documentation — see `docs/model_card.md`
and `docs/bias_log.md`.

## Support
Contact the hub platform team for:
- Access issues
- Package approval requests
- Platform questions

For use-case specific questions, contact {{cookiecutter.engineer_group}}.