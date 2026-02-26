# {{cookiecutter.lob_display_name}} - {{cookiecutter.project_slug}}

This is the official blueprint template for creating new Databricks projects in the hub and spoke platform.

## Project Overview
<!-- Describe your use case here -->

## Architecture
This project follows the standard hub and spoke MLops architecture:
```
Data Ingestion (DLT)
        ↓
Feature Engineering (Databricks Feature Store)
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
- Databricks CLI installed (`curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh`)
- Access to `{{cookiecutter.workspace_url}}`
- Unity Catalog access to `{{cookiecutter.catalog_name}}`

### Option A — GitHub repo (GitHub Actions CI/CD)
1. Click "Use this template" on GitHub to create your repo
2. Delete the `.azure/` folder — you won't need it
3. Set up GitHub Actions variables in your repo settings:
   - `AZURE_TENANT_ID`
   - `AZURE_CLIENT_ID`
   - `AZURE_SUBSCRIPTION_ID`
4. Push to `main` to trigger deployment to dev

### Option B — Azure DevOps repo (Azure DevOps Pipelines)
1. Run Cookiecutter locally:
```bash
   cookiecutter gh:your-org/dsgrd-databricks-lob-template
```
2. Push the generated project to your Azure DevOps repo
3. Delete the `.github/` folder — you won't need it
4. Create a pipeline in Azure DevOps pointing to `.azure/pipelines/bundle-deploy.yml`
5. Create a variable group named `databricks-lob-variables` with:
   - `AZURE_SERVICE_CONNECTION` — name of your Azure service connection in Azure DevOps

### Option C — GitHub repo (Azure DevOps Pipelines)
1. Click "Use this template" on GitHub or run Cookiecutter
2. Keep both CI/CD folders or delete the one you don't use
3. Connect Azure DevOps Pipelines to your GitHub repo
4. Follow step 4-5 from Option B

### Deploy manually
```bash
# Deploy to dev
databricks bundle deploy --target dev

# Deploy to staging
databricks bundle deploy --target staging

# Deploy to prod
databricks bundle deploy --target prod
```

### Run pipelines manually
```bash
# Ingest data
databricks bundle run ingestion_pipeline

# Compute features
databricks bundle run feature_table_job

# Train and evaluate models
databricks bundle run training_job

# Run batch inference
databricks bundle run inference_pipeline

# Deploy serving endpoint (optional)
databricks bundle run serving_job

# Run monitoring
databricks bundle run monitoring_job
```

## Project Structure
```
├── resources/                          # Databricks Asset Bundle resource definitions
│   ├── feature_pipeline.yml            # DLT ingestion pipeline (bronze/silver)
│   ├── feature_table_job.yml           # Feature Store job
│   ├── training_job.yml                # Model training and evaluation
│   ├── inference_pipeline.yml          # Batch inference pipeline
│   ├── serving_job.yml                 # Model serving deployment
│   ├── monitoring_job.yml              # Lakehouse monitoring
│   └── orchestration_job.yml          # End-to-end orchestration
├── src/
│   └── {{cookiecutter.project_slug}}/
│       ├── features/
│       │   ├── ingestion_pipeline.py   # DLT bronze/silver ingestion
│       │   └── feature_pipeline.py     # Feature Store feature engineering
│       ├── training/
│       │   ├── classification/         # Classification model training
│       │   ├── time_series/            # Time series model training
│       │   └── clustering/             # Clustering model training
│       ├── evaluation/
│       │   ├── classification/         # Champion/challenger evaluation
│       │   ├── time_series/            # Champion/challenger evaluation
│       │   └── clustering/             # Champion/challenger evaluation
│       ├── inference/                  # Batch inference pipeline
│       ├── serving/                    # Model serving endpoint deployment
│       └── monitoring/                 # Lakehouse monitoring setup
├── tests/                              # Unit tests
├── notebooks/exploratory/              # Ad-hoc exploration notebooks
├── docs/                               # EU AI Act documentation templates
├── .github/workflows/                  # GitHub Actions CI/CD
└── .azure/pipelines/                   # Azure DevOps CI/CD
```

## Feature Store
Features are managed via the Databricks Feature Engineering API and stored in Unity Catalog.
The feature table `{{cookiecutter.catalog_name}}.{{cookiecutter.schema_name}}.{{cookiecutter.project_slug}}_features`
is registered in the Feature Store and can be shared with other projects.

Contact the hub platform team to share features with other LOBs or to consume
features produced by other teams.

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
- Feature Store sharing across LOBs
- Platform questions

For use-case specific questions, contact {{cookiecutter.engineer_group}}.