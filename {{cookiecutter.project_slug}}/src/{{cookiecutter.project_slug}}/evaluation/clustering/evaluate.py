# Databricks notebook source
# Clustering Model Evaluation
# Implements champion/challenger pattern using MLflow and Unity Catalog model registry

# COMMAND ----------

# MAGIC %md
# MAGIC # {{cookiecutter.lob_display_name}} - Clustering Model Evaluation
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Loads the latest challenger model from Unity Catalog
# MAGIC 2. Evaluates it against the current champion
# MAGIC 3. Promotes the challenger to champion if it outperforms
# MAGIC
# MAGIC Note: Clustering evaluation is inherently more subjective than supervised learning.
# MAGIC Silhouette score is used as the primary metric but business validation
# MAGIC of cluster interpretability is strongly recommended before promoting to champion.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

import mlflow
from mlflow import MlflowClient
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# COMMAND ----------

dbutils.widgets.text("catalog", "{{cookiecutter.catalog_name}}_dev")
dbutils.widgets.text("schema", "{{cookiecutter.schema_name}}")
dbutils.widgets.text("model_name", "{{cookiecutter.model_name}}_clustering")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name")

model_uri = f"{catalog}.{schema}.{model_name}"

print(f"Catalog: {catalog}")
print(f"Schema: {schema}")
print(f"Model URI: {model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load Features from Feature Store

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()
fe = FeatureEngineeringClient()

feature_table_name = f"{catalog}.{schema}.{{cookiecutter.project_slug}}_features"

# Read entity IDs from feature table (no separate entities table)
entity_df = spark.table(feature_table_name).select("id")

eval_set = fe.create_training_set(
    df=entity_df,
    feature_lookups=[
        FeatureLookup(
            table_name=feature_table_name,
            lookup_key="id"
        )
    ],
    label=None,
    exclude_columns=["id"]
)

df = eval_set.load_df().toPandas()
feature_cols = [c for c in df.select_dtypes(include=["number"]).columns
               if c not in ["id"]]
X = df[feature_cols].fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Evaluate Challenger Model

# COMMAND ----------

def evaluate_model(model, X_scaled):
    """Evaluate clustering model and return metrics."""
    labels = model.predict(X_scaled)
    return {
        "silhouette_score": silhouette_score(X_scaled, labels),
        "davies_bouldin_score": davies_bouldin_score(X_scaled, labels),
        "inertia": model.inertia_,
        "n_clusters": model.n_clusters
    }

# COMMAND ----------

versions = client.search_model_versions(f"name='{model_uri}'")
if not versions:
    raise ValueError(f"No model versions found for {model_uri}")

def is_pipeline_version(v):
    """Return True if this version was registered by our training pipeline."""
    try:
        if not v.run_id:
            return False
        run = client.get_run(v.run_id)
        return run.data.params.get("model_type") == "KMeans"
    except Exception:
        return False

pipeline_versions = [v for v in versions if is_pipeline_version(v)]
if not pipeline_versions:
    raise ValueError(f"No pipeline-registered versions found for {model_uri}. "
                     "Check that training has run successfully.")

challenger_version = sorted(pipeline_versions, key=lambda x: int(x.version), reverse=True)[0]
challenger_model = mlflow.sklearn.load_model(
    f"models:/{model_uri}/{challenger_version.version}"
)

challenger_metrics = evaluate_model(challenger_model, X_scaled)
print(f"Challenger version: {challenger_version.version}")
print(f"Challenger metrics: {challenger_metrics}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Compare with Champion and Promote if Better

# COMMAND ----------

try:
    client.get_model_version_by_alias(model_uri, "champion")
    try:
        champion_model = mlflow.sklearn.load_model(f"models:/{model_uri}@champion")
        champion_metrics = evaluate_model(champion_model, X_scaled)
        print(f"Champion metrics: {champion_metrics}")

        # Promote challenger if silhouette score is at least as good as champion.
        # Adjust the threshold and metric to your own requirements.
        # Note: always validate cluster interpretability with business stakeholders
        if challenger_metrics["silhouette_score"] >= champion_metrics["silhouette_score"]:
            print("Challenger meets promotion threshold - promoting to champion")
            client.set_registered_model_alias(model_uri, "champion", challenger_version.version)
        else:
            print("Champion retained - challenger did not meet promotion threshold")
    except Exception as e:
        # Champion model cannot be loaded (e.g. environment/dependency mismatch) - promote challenger
        print(f"Champion model failed to load ({e}) - promoting challenger to champion")
        client.set_registered_model_alias(model_uri, "champion", challenger_version.version)

except Exception:
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