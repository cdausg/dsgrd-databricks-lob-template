# Databricks notebook source
# Time Series Model Evaluation
# Implements champion/challenger pattern using MLflow and Unity Catalog model registry

# COMMAND ----------

# MAGIC %md
# MAGIC # {{cookiecutter.lob_display_name}} - Time Series Model Evaluation
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Loads the latest challenger model from Unity Catalog
# MAGIC 2. Evaluates it against the current champion on a holdout period
# MAGIC 3. Promotes the challenger to champion if it outperforms
# MAGIC
# MAGIC Adjust the promotion threshold and metric to match your business requirements.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

import mlflow
from mlflow import MlflowClient
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np
import pyspark.sql.functions as F

# COMMAND ----------

dbutils.widgets.text("catalog", "{{cookiecutter.catalog_name}}_dev")
dbutils.widgets.text("schema", "{{cookiecutter.schema_name}}")
dbutils.widgets.text("model_name", "{{cookiecutter.model_name}}_time_series")
dbutils.widgets.text("horizon", "30")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name")
horizon = int(dbutils.widgets.get("horizon"))

model_uri = f"{catalog}.{schema}.{model_name}"

print(f"Catalog: {catalog}")
print(f"Schema: {schema}")
print(f"Model URI: {model_uri}")
print(f"Horizon: {horizon} days")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load Test Data

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

# Load labels - rename target to y, generate synthetic ds matching training
label_df = spark.table(f"{catalog}.{schema}.labels").select(
    F.col("target").alias("y")
).toPandas()

label_df = label_df[["y"]].dropna().reset_index(drop=True)
label_df["ds"] = pd.date_range(end=pd.Timestamp.today(), periods=len(label_df), freq="D")
df = label_df[["ds", "y"]]
test_df = df[-horizon:]
future = pd.DataFrame({"ds": test_df["ds"]})

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Evaluate Challenger Model

# COMMAND ----------

def evaluate_model(model, test_df, future, horizon):
    """Evaluate time series model on holdout period."""
    forecast = model.predict(future)
    y_true = test_df["y"].values
    y_pred = forecast["yhat"].values[-horizon:]

    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mape": np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }

# COMMAND ----------

versions = client.search_model_versions(f"name='{model_uri}'")
if not versions:
    raise ValueError(f"No model versions found for {model_uri}")

challenger_version = sorted(versions, key=lambda x: int(x.version), reverse=True)[0]
challenger_model = mlflow.pyfunc.load_model(
    f"models:/{model_uri}/{challenger_version.version}"
)

challenger_metrics = evaluate_model(challenger_model, test_df, future, horizon)
print(f"Challenger version: {challenger_version.version}")
print(f"Challenger metrics: {challenger_metrics}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Compare with Champion and Promote if Better

# COMMAND ----------

try:
    client.get_model_version_by_alias(model_uri, "champion")
    champion_model = mlflow.pyfunc.load_model(f"models:/{model_uri}@champion")
    champion_metrics = evaluate_model(champion_model, test_df, future, horizon)
    print(f"Champion metrics: {champion_metrics}")

    # Promote challenger if MAE improves by more than 1%
    # Adjust the threshold and metric to your own requirements
    if challenger_metrics["mae"] < champion_metrics["mae"] * 0.99:
        print("Challenger outperforms champion - promoting to champion")
        client.set_registered_model_alias(model_uri, "champion", challenger_version.version)
    else:
        print("Champion retained - challenger did not meet promotion threshold")

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
        "challenger_version": challenger_version.version,
        "horizon": horizon
    })
    print("Evaluation results logged to MLflow")