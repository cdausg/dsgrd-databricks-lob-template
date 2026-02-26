# Databricks notebook source
# Clustering Model Training
# Uses KMeans for segmentation and MLflow for experiment tracking

# COMMAND ----------

# MAGIC %md
# MAGIC # {{cookiecutter.lob_display_name}} - Clustering Model Training
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Loads features from the Feature Store
# MAGIC 2. Trains a KMeans clustering model
# MAGIC 3. Logs the model and metrics to MLflow
# MAGIC 4. Registers the model in Unity Catalog

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import pandas as pd
import numpy as np

# COMMAND ----------

dbutils.widgets.text("catalog", "{{cookiecutter.catalog_name}}_dev")
dbutils.widgets.text("schema", "{{cookiecutter.schema_name}}")
dbutils.widgets.text("model_name", "{{cookiecutter.model_name}}_clustering")
dbutils.widgets.text("experiment_path", "/Shared/{{cookiecutter.project_slug}}/experiments/clustering")
dbutils.widgets.text("n_clusters", "5")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name")
experiment_path = dbutils.widgets.get("experiment_path")
n_clusters = int(dbutils.widgets.get("n_clusters"))

print(f"Catalog: {catalog}")
print(f"Schema: {schema}")
print(f"Model name: {model_name}")
print(f"N clusters: {n_clusters}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load Features from Feature Store

# COMMAND ----------

fe = FeatureEngineeringClient()
mlflow.set_registry_uri("databricks-uc")

# Create parent workspace directory if it doesn't exist
import os
from databricks.sdk import WorkspaceClient
_ws = WorkspaceClient()
_ws.workspace.mkdirs(os.path.dirname(experiment_path))

mlflow.set_experiment(experiment_path)

feature_table_name = f"{catalog}.{schema}.{{cookiecutter.project_slug}}_features"

# For clustering there is no label - use the feature table itself as the entity source
entity_df = spark.table(feature_table_name).select("id")

training_set = fe.create_training_set(
    df=entity_df,
    feature_lookups=[
        FeatureLookup(
            table_name=feature_table_name,
            lookup_key="id"
        )
    ],
    label=None,  # No label for unsupervised learning
    exclude_columns=["id"]
)

df = training_set.load_df().toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Find Optimal Number of Clusters
# MAGIC Elbow method - review the plot before choosing n_clusters

# COMMAND ----------

feature_cols = [c for c in df.select_dtypes(include=["number"]).columns
               if c not in ["id"]]
X = df[feature_cols].fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow curve
inertias = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append((k, km.inertia_))

print("Elbow curve (k, inertia):")
for k, inertia in inertias:
    print(f"  k={k}: {inertia:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Train Model
# MAGIC Replace n_clusters widget value with your chosen number

# COMMAND ----------

with mlflow.start_run() as run:
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)

    metrics = {
        "silhouette_score": silhouette_score(X_scaled, labels),
        "davies_bouldin_score": davies_bouldin_score(X_scaled, labels),
        "inertia": model.inertia_,
        "n_clusters": n_clusters
    }

    # Log elbow curve
    for k, inertia in inertias:
        mlflow.log_metric("elbow_inertia", inertia, step=k)

    mlflow.log_params({
        "catalog": catalog,
        "schema": schema,
        "model_type": "KMeans",
        "n_clusters": n_clusters
    })
    mlflow.log_metrics(metrics)

    # Unity Catalog requires both input and output signatures.
    # fe.log_model ignores the signature parameter (it bakes in the Feature Store schema).
    # For clustering (no feature lookup needed at inference time), log directly with sklearn.
    X_sample = pd.DataFrame(X_scaled, columns=feature_cols)
    output_sample = pd.Series(labels.astype(float), name="cluster")
    signature = infer_signature(X_sample, output_sample)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        registered_model_name=f"{catalog}.{schema}.{model_name}"
    )

    print(f"Run ID: {run.info.run_id}")
    print(f"Model registered: {catalog}.{schema}.{model_name}")
    print(f"Metrics: {metrics}")