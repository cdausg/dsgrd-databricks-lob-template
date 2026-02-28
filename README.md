# dsgrd-databricks-lob-template

GitHub template repository and Cookiecutter template for scaffolding new MLOps projects on the Databricks hub-and-spoke platform. Supports both **Azure** and **AWS**.

## What this template generates

A fully structured Databricks MLOps project including:

- **DLT ingestion pipeline** — bronze/silver layers with Auto Loader
- **Feature engineering** — Databricks Feature Engineering API, Unity Catalog feature tables
- **Model training** — classification, time series (Prophet), and clustering archetypes
- **Model evaluation** — champion/challenger pattern using MLflow model aliases
- **Batch inference** — DLT pipeline writing Gold tables
- **Model serving** — optional real-time endpoint deployment via Databricks SDK
- **Monitoring** — Databricks Lakehouse Monitoring for data and model drift
- **End-to-end orchestration** — single orchestration job wiring all steps together
- **CI/CD** — GitHub Actions (Azure and AWS) and Azure DevOps pipelines
- **EU AI Act docs** — model card, bias log, and risk classification templates

## What you need from the hub platform team

Before setting up a project, collect the following from the hub platform team:

| Item | Azure | AWS |
|---|---|---|
| Workspace URLs (dev/staging/prod) | `.azuredatabricks.net` URLs | `.cloud.databricks.com` URLs |
| Unity Catalog catalog name | e.g. `lob_a` | e.g. `lob_a` |
| Cluster policy ID | from Terraform output | from Terraform output |
| Instance pool ID (dev) | from Terraform output | from Terraform output |
| CI/CD identity | Azure AD app client ID | IAM role ARN (`github_actions_role_arn` output) |
| Secret store | Azure Key Vault `kv-dsgrd-hub` | AWS Secrets Manager region |
| Node type | e.g. `Standard_DS3_v2` | e.g. `i3.xlarge` |

## How to use

### Option A — Cookiecutter (recommended)

```bash
pip install cookiecutter
cookiecutter gh:your-org/dsgrd-databricks-lob-template
```

The post-generation hook automatically selects and places the correct CI/CD workflow files based on your `cloud_provider` and `cicd_platform` choices. No manual cleanup needed.

### Option B — GitHub template (manual setup)

1. Click **Use this template** on GitHub to create your repo
2. Replace all `{{cookiecutter.*}}` placeholders with your values
3. Follow the CI/CD setup below for your cloud

## CI/CD Setup

### Azure + GitHub Actions

1. Copy `.github/workflows/azure/bundle-deploy.yml` to `.github/workflows/bundle-deploy.yml`
2. Delete `.github/workflows/azure/`, `.github/workflows/aws/`
3. Delete `.azure/` if not using Azure DevOps
4. Set the following **GitHub repository variables**:
   - `AZURE_CLIENT_ID`
   - `AZURE_TENANT_ID`
   - `AZURE_SUBSCRIPTION_ID`

Databricks tokens are fetched at runtime from Azure Key Vault (`kv-dsgrd-hub`) using the OIDC identity — no Databricks secrets needed in GitHub.

### Azure + Azure DevOps Pipelines

1. Delete `.github/` folder
2. In Azure DevOps, create a pipeline pointing to `.azure/pipelines/bundle-deploy.yml`
3. Create a variable group named `databricks-lob-variables` containing:
   - `AZURE_SERVICE_CONNECTION` — name of your Azure service connection in Azure DevOps

### AWS + GitHub Actions

1. Copy `.github/workflows/aws/bundle-deploy.yml` to `.github/workflows/bundle-deploy.yml`
2. Delete `.github/workflows/azure/`, `.github/workflows/aws/`
3. Delete `.azure/` folder (Azure DevOps is not applicable for AWS LOBs)
4. Set the following **GitHub repository variables**:
   - `AWS_ROLE_ARN` — IAM role ARN from hub Terraform output `github_actions_role_arn`
   - `AWS_REGION` — AWS region where Secrets Manager secrets are stored

Databricks tokens are fetched at runtime from AWS Secrets Manager using OIDC — no Databricks secrets needed in GitHub.

## Cookiecutter Variables

| Variable | Description | Azure example | AWS example |
|---|---|---|---|
| `project_slug` | Project name (lowercase, underscores) | `lob_a_churn_prediction` | `lob_b_churn_prediction` |
| `lob_name` | LOB identifier | `lob-a` | `lob-b` |
| `lob_display_name` | LOB display name | `LOB A` | `LOB B` |
| `catalog_name` | Unity Catalog catalog — get from hub team | `lob_a` | `lob_b` |
| `schema_name` | Schema name | `default` | `default` |
| `workspace_url_dev` | Dev workspace URL | `https://adb-xxx.azuredatabricks.net` | `https://xxx.cloud.databricks.com` |
| `workspace_url_staging` | Staging workspace URL | | |
| `workspace_url_prod` | Prod workspace URL | | |
| `cloud_provider` | Cloud provider | `azure` | `aws` |
| `aws_region` | AWS Secrets Manager region (AWS only) | — | `eu-west-1` |
| `spark_version` | Databricks runtime | `15.4.x-scala2.12` | `15.4.x-scala2.12` |
| `node_type_id` | VM node type — get from hub team | `Standard_DS3_v2` | `i3.xlarge` |
| `cluster_policy_id` | Cluster policy ID — get from hub team | | |
| `instance_pool_id` | Dev instance pool ID — get from hub team | | |
| `engineer_group` | Identity group for your team | `lob-a-engineers@client.com` (Entra ID) | `lob-b-engineers` (IAM group) |
| `model_name` | MLflow registered model name | `lob_a_churn_model` | `lob_b_churn_model` |
| `mlflow_experiment_path` | MLflow experiment path | `/Shared/lob_a.../experiments` | |
| `cicd_platform` | CI/CD platform (Azure LOBs only) | `github_actions` or `azure_devops` | ignored |

## CI/CD Flow

| Event | Action |
|---|---|
| PR to `main` | Validate bundle |
| Push to `main` | Validate + deploy to **dev** |
| Push to `release/*` | Validate → deploy **dev** → **staging** → **prod** (prod requires environment approval gate) |

## Project Structure

```
{{project_slug}}/
├── src/{{project_slug}}/
│   ├── features/
│   │   └── ingestion_pipeline.py   # DLT bronze/silver ingestion
│   ├── training/
│   │   ├── classification/train.py
│   │   ├── time_series/train.py
│   │   └── clustering/train.py
│   ├── evaluation/
│   │   ├── classification/evaluate.py
│   │   ├── time_series/evaluate.py
│   │   └── clustering/evaluate.py
│   ├── inference/
│   │   └── batch_inference.py      # DLT Gold table
│   ├── serving/
│   │   └── deploy_serving.py
│   └── monitoring/
│       └── setup_monitoring.py
├── resources/                      # Databricks Asset Bundle resource YAMLs
├── databricks.yml                  # Bundle config (targets: dev, staging, prod)
├── .github/workflows/
│   ├── azure/bundle-deploy.yml     # Azure OIDC + Key Vault
│   └── aws/bundle-deploy.yml       # AWS OIDC + Secrets Manager
└── .azure/pipelines/
    └── bundle-deploy.yml           # Azure DevOps (Azure LOBs only)
```

## Related

- [dsgrd-databricks-hub](https://github.com/your-org/dsgrd-databricks-hub) — Hub infrastructure repo (Terraform + shared bundles, supports Azure and AWS)
