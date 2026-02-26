# Databricks notebook source
# Classification Model Training
# Uses MLflow for experiment tracking and Unity Catalog for model registry

# COMMAND ----------

# MAGIC %md
# MAGIC # {{cookiecutter.lob_display_name}} - Classification Model Training
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Loads features from the Feature Store
# MAGIC 2. Trains a classification model
# MAGIC 3. Logs the model and metrics to MLflow
# MAGIC 4. Registers the model in Unity Catalog

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

import mlflow
import mlflow.sklearn
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# COMMAND ----------

dbutils.widgets.text("catalog", "{{cookiecutter.catalog_name}}_dev")
dbutils.widgets.text("schema", "{{cookiecutter.schema_name}}")
dbutils.widgets.text("model_name", "{{cookiecutter.model_name}}_classification")
dbutils.widgets.text("experiment_path", "/Shared/{{cookiecutter.project_slug}}/experiments/classification")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name")
experiment_path = dbutils.widgets.get("experiment_path")

print(f"Catalog: {catalog}")
print(f"Schema: {schema}")
print(f"Model name: {model_name}")
print(f"Experiment path: {experiment_path}")

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

# Load feature table - Feature Store handles point-in-time lookups automatically
# Replace 'target' with your actual label table
label_df = spark.table(f"{catalog}.{schema}.labels").select("id", "target")

training_set = fe.create_training_set(
    df=label_df,
    feature_lookups=[
        FeatureLookup(
            table_name=feature_table_name,
            lookup_key="id"
        )
    ],
    label="target",
    exclude_columns=["id"]
)

df = training_set.load_df().toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Train Model
# MAGIC Replace with your own model and hyperparameters

# COMMAND ----------

X = df.drop(columns=["target"]).select_dtypes(include=["number"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with mlflow.start_run() as run:
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    metrics = {
        "accuracy": accuracy_score(y_test, model.predict(X_test)),
        "f1_score": f1_score(y_test, model.predict(X_test), average="weighted"),
        "roc_auc": roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    }

    mlflow.log_params({
        "catalog": catalog,
        "schema": schema,
        "model_type": "RandomForestClassifier",
        "n_estimators": 100
    })
    mlflow.log_metrics(metrics)

    # Log model with Feature Store - preserves feature lineage
    fe.log_model(
        model=model,
        artifact_path="model",
        flavor=mlflow.sklearn,
        training_set=training_set,
        registered_model_name=f"{catalog}.{schema}.{model_name}"
    )

    print(f"Run ID: {run.info.run_id}")
    print(f"Model registered: {catalog}.{schema}.{model_name}")
    print(f"Metrics: {metrics}")