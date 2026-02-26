# dsgrd-databricks-lob-template

GitHub template repository and Cookiecutter template for scaffolding new MLOps projects on the Databricks hub-and-spoke platform.

## What this template generates

A fully structured Databricks MLOps project including:

- **DLT ingestion pipeline** — bronze/silver layers with Auto Loader
- **Feature engineering** — Databricks Feature Engineering API, Unity Catalog feature tables
- **Model training** — classification, time series (Prophet), and clustering archetypes
- **Model evaluation** — champion/challenger pattern using MLflow model aliases
- **Batch inference** — DLT pipeline writing Gold tables (Live Lake model)
- **Model serving** — optional real-time endpoint deployment via Databricks SDK
- **Monitoring** — Databricks Lakehouse Monitoring for data and model drift
- **End-to-end orchestration** — single orchestration job wiring all steps together
- **CI/CD** — both GitHub Actions and Azure DevOps pipelines included
- **EU AI Act docs** — model card, bias log, and risk classification templates

## How to use

The template supports any combination of Git host and CI/CD platform. Both `.github/workflows/` and `.azure/pipelines/` are generated — delete the folder you don't need.

### Option A — GitHub repo + GitHub Actions

1. Click **Use this template** on GitHub to create your repo, or run Cookiecutter (see below)
2. Replace all `{{cookiecutter.*}}` placeholders with your actual values (skip if using Cookiecutter)
3. Delete the `.azure/` folder
4. Set up GitHub Actions variables in your repo settings:
   - `AZURE_TENANT_ID`
   - `AZURE_CLIENT_ID`
   - `AZURE_SUBSCRIPTION_ID`
5. Push to `main` to trigger your first deployment to dev

### Option B — GitHub repo + Azure DevOps Pipelines

1. Click **Use this template** on GitHub to create your repo, or run Cookiecutter
2. Replace all `{{cookiecutter.*}}` placeholders with your actual values (skip if using Cookiecutter)
3. Delete the `.github/` folder
4. In Azure DevOps, create a pipeline pointing to `.azure/pipelines/bundle-deploy.yml`
5. Connect the pipeline to your GitHub repo
6. Create a variable group named `databricks-lob-variables` with:
   - `AZURE_SERVICE_CONNECTION` — name of your Azure service connection in Azure DevOps

### Option C — Azure DevOps repo + Azure DevOps Pipelines

1. Run Cookiecutter locally and push the generated project to your Azure DevOps repo:
```bash
pip install cookiecutter
cookiecutter gh:your-org/dsgrd-databricks-lob-template
```
2. Delete the `.github/` folder
3. In Azure DevOps, create a pipeline pointing to `.azure/pipelines/bundle-deploy.yml`
4. Create a variable group named `databricks-lob-variables` with:
   - `AZURE_SERVICE_CONNECTION` — name of your Azure service connection in Azure DevOps

### Option D — Azure DevOps repo + GitHub Actions

This combination is less common but supported. Connect GitHub Actions to your Azure DevOps repo using a GitHub mirror or Azure DevOps service hook, then follow Option A steps for the pipeline setup.

## Cookiecutter variables

| Variable | Description | Example |
|---|---|---|
| `project_slug` | Project name (lowercase, underscores) | `lob_a_churn_prediction` |
| `lob_name` | LOB identifier | `lob-a` |
| `lob_display_name` | LOB display name | `LOB A` |
| `catalog_name` | Unity Catalog catalog name — get from hub platform team | `lob_a` |
| `schema_name` | Schema name (leave as `default` unless told otherwise) | `default` |
| `workspace_url_dev` | Databricks workspace URL for dev — get from hub platform team | `https://adb-xxx.azuredatabricks.net` |
| `workspace_url_staging` | Databricks workspace URL for staging — get from hub platform team | `https://adb-yyy.azuredatabricks.net` |
| `workspace_url_prod` | Databricks workspace URL for prod — get from hub platform team | `https://adb-zzz.azuredatabricks.net` |
| `cloud_provider` | Cloud provider | `azure` or `aws` |
| `spark_version` | Databricks runtime version — confirm with hub platform team | `15.4.x-scala2.12` |
| `node_type_id` | VM node type — confirm with hub platform team | `Standard_DS3_v2` |
| `cluster_policy_id` | Cluster policy ID — get from hub platform team | |
| `engineer_group` | Entra ID / IAM group for your team | `lob-a-engineers@client.com` |
| `model_name` | MLflow registered model name | `lob_a_churn_prediction_model` |
| `mlflow_experiment_path` | MLflow experiment path | `/Shared/lob_a_churn_prediction/experiments` |
| `cicd_platform` | CI/CD platform — delete the unused folder after generation | `github_actions` or `azure_devops` |

## What you need from the hub platform team

Before generating a project, collect the following from the hub platform team:

- Workspace URL for each environment (dev / staging / prod)
- Unity Catalog catalog name for your LOB
- Cluster policy ID
- Approved Spark runtime version and node type

## CI/CD flow

| Event | Action |
|---|---|
| PR to `main` | Validate bundle |
| Push to `main` | Validate + deploy to **dev** |
| Push to `release/*` | Validate + deploy to **dev** → **staging** → **prod** (prod requires approval via environment gate) |

## Related

- [dsgrd-databricks-hub](https://github.com/your-org/dsgrd-databricks-hub) — Hub infrastructure repo (Terraform + shared bundles)
