# Databricks notebook source
# Classification Model Evaluation
# Implements champion/challenger pattern using MLflow and Unity Catalog model registry

# COMMAND ----------

# MAGIC %md
# MAGIC # {{cookiecutter.lob_display_name}} - Classification Model Evaluation
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Loads the latest challenger model from Unity Catalog
# MAGIC 2. Evaluates it against the current champion
# MAGIC 3. Promotes the challenger to champion if it outperforms
# MAGIC
# MAGIC This implements the champion/challenger pattern recommended by Databricks.
# MAGIC Adjust the promotion threshold and metric to match your business requirements.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

import mlflow
from mlflow import MlflowClient
from databricks.feature_engineering import FeatureEngineeringClient
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

# COMMAND ----------

dbutils.widgets.text("catalog", "{{cookiecutter.catalog_name}}_dev")
dbutils.widgets.text("schema", "{{cookiecutter.schema_name}}")
dbutils.widgets.text("model_name", "{{cookiecutter.model_name}}_classification")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name")

model_uri = f"{catalog}.{schema}.{model_name}"

print(f"Catalog: {catalog}")
print(f"Schema: {schema}")
print(f"Model URI: {model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load Test Data

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()
fe = FeatureEngineeringClient()

# Load labels for evaluation - keep id for fe.score_batch lookup
label_df = spark.table(f"{catalog}.{schema}.labels").select("id", "target")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Evaluate Challenger Model

# COMMAND ----------

def evaluate_model(model_uri_versioned, label_df, y_col="target"):
    """Evaluate classification model using fe.score_batch (handles feature lookups).
    Returns metrics dict and predictions joined with labels.
    """
    preds_df = fe.score_batch(
        model_uri=model_uri_versioned,
        df=label_df.select("id")
    )
    result = preds_df.join(label_df.select("id", y_col), on="id").toPandas()
    preds = result["prediction"].values
    y_true = result[y_col].values
    return {
        "accuracy": accuracy_score(y_true, preds),
        "f1_score": f1_score(y_true, preds, average="weighted"),
    }

# COMMAND ----------

# Get latest challenger version
versions = client.search_model_versions(f"name='{model_uri}'")
if not versions:
    raise ValueError(f"No model versions found for {model_uri}")

challenger_version = sorted(versions, key=lambda x: int(x.version), reverse=True)[0]
challenger_model_uri = f"models:/{model_uri}/{challenger_version.version}"

challenger_metrics = evaluate_model(challenger_model_uri, label_df)
print(f"Challenger version: {challenger_version.version}")
print(f"Challenger metrics: {challenger_metrics}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Compare with Champion and Promote if Better

# COMMAND ----------

try:
    client.get_model_version_by_alias(model_uri, "champion")
    champion_metrics = evaluate_model(f"models:/{model_uri}@champion", label_df)
    print(f"Champion metrics: {champion_metrics}")

    # Promote challenger if F1 score improves by more than 1%
    # Adjust the threshold and metric to your own requirements
    if challenger_metrics["f1_score"] > champion_metrics["f1_score"] * 1.01:
        print("Challenger outperforms champion - promoting to champion")
        client.set_registered_model_alias(model_uri, "champion", challenger_version.version)
    else:
        print("Champion retained - challenger did not meet promotion threshold")

except Exception:
    # No champion exists yet - promote challenger automatically
    print("No champion found - promoting challenger to champion")
    client.set_registered_model_alias(model_uri, "champion", challenger_version.version)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Log Evaluation Results

# COMMAND ----------

with mlflow.start_run():
    mlflow.log_metrics(challenger_metrics)
    mlflow.log_params({
        "catalog": catalog,
        "schema": schema,
        "model_name": model_name,
        "challenger_version": challenger_version.version
    })
    print("Evaluation results logged to MLflow")