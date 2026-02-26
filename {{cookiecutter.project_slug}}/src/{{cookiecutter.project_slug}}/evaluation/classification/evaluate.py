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

# Load labels and join numeric features (same preprocessing as training)
label_df = spark.table(f"{catalog}.{schema}.labels").select("id", "target")
feature_table_name = f"{catalog}.{schema}.{{cookiecutter.project_slug}}_features"

eval_set = fe.create_training_set(
    df=label_df,
    feature_lookups=[FeatureLookup(table_name=feature_table_name, lookup_key="id")],
    label="target",
    exclude_columns=["id"]
)
df = eval_set.load_df().toPandas()
# Use same numeric-only features as training (drops feature_updated_timestamp)
X = df.drop(columns=["target"]).select_dtypes(include=["number"])
y = df["target"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Evaluate Challenger Model

# COMMAND ----------

def evaluate_model(model, X, y):
    """Evaluate classification model. Model is a plain sklearn model."""
    preds = model.predict(X)
    return {
        "accuracy": accuracy_score(y, preds),
        "f1_score": f1_score(y, preds, average="weighted"),
    }

# COMMAND ----------

# Get latest challenger version
versions = client.search_model_versions(f"name='{model_uri}'")
if not versions:
    raise ValueError(f"No model versions found for {model_uri}")

challenger_version = sorted(versions, key=lambda x: int(x.version), reverse=True)[0]
challenger_model = mlflow.sklearn.load_model(
    f"models:/{model_uri}/{challenger_version.version}"
)

challenger_metrics = evaluate_model(challenger_model, X, y)
print(f"Challenger version: {challenger_version.version}")
print(f"Challenger metrics: {challenger_metrics}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Compare with Champion and Promote if Better

# COMMAND ----------

try:
    client.get_model_version_by_alias(model_uri, "champion")
    champion_model = mlflow.sklearn.load_model(f"models:/{model_uri}@champion")
    champion_metrics = evaluate_model(champion_model, X, y)
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