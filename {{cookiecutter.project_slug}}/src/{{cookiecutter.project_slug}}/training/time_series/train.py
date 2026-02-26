# Databricks notebook source
# Time Series Model Training
# Uses Prophet for forecasting and MLflow for experiment tracking

# COMMAND ----------

# MAGIC %md
# MAGIC # {{cookiecutter.lob_display_name}} - Time Series Model Training
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Loads features from the Feature Store
# MAGIC 2. Trains a Prophet forecasting model
# MAGIC 3. Logs the model and metrics to MLflow
# MAGIC 4. Registers the model in Unity Catalog

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

import mlflow
import mlflow.pyfunc
from databricks.feature_engineering import FeatureEngineeringClient
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np

# COMMAND ----------

dbutils.widgets.text("catalog", "{{cookiecutter.catalog_name}}_dev")
dbutils.widgets.text("schema", "{{cookiecutter.schema_name}}")
dbutils.widgets.text("model_name", "{{cookiecutter.model_name}}_time_series")
dbutils.widgets.text("experiment_path", "{{cookiecutter.mlflow_experiment_path}}/time_series")
dbutils.widgets.text("horizon", "30")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name")
experiment_path = dbutils.widgets.get("experiment_path")
horizon = int(dbutils.widgets.get("horizon"))

print(f"Catalog: {catalog}")
print(f"Schema: {schema}")
print(f"Model name: {model_name}")
print(f"Horizon: {horizon} days")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load Features from Feature Store

# COMMAND ----------

fe = FeatureEngineeringClient()
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(experiment_path)

feature_table_name = f"{catalog}.{schema}.{{cookiecutter.project_slug}}_features"

# Load feature table
# For time series, the lookup key is typically a time-based entity (e.g. product_id, store_id)
label_df = spark.table(f"{catalog}.{schema}.labels").select("id", "ds", "y")

training_set = fe.create_training_set(
    df=label_df,
    feature_lookups=[
        {
            "table_name": feature_table_name,
            "lookup_key": "id",
            "timestamp_lookup_key": None
        }
    ],
    label="y",
    exclude_columns=["id"]
)

df = training_set.load_df().toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Train Model
# MAGIC Replace with your own model and hyperparameters

# COMMAND ----------

class ProphetWrapper(mlflow.pyfunc.PythonModel):
    """MLflow wrapper for Prophet to enable Unity Catalog registration."""
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        future = self.model.make_future_dataframe(periods=len(model_input))
        forecast = self.model.predict(future)
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]

# COMMAND ----------

train_df = df[:-horizon]
test_df = df[-horizon:]

with mlflow.start_run() as run:
    model = Prophet(
        seasonality_mode="multiplicative",
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )
    model.fit(train_df[["ds", "y"]])

    future = model.make_future_dataframe(periods=horizon)
    forecast = model.predict(future)

    y_true = test_df["y"].values
    y_pred = forecast[-horizon:]["yhat"].values

    metrics = {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mape": np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }

    mlflow.log_params({
        "catalog": catalog,
        "schema": schema,
        "model_type": "Prophet",
        "horizon": horizon,
        "seasonality_mode": "multiplicative"
    })
    mlflow.log_metrics(metrics)

    fe.log_model(
        model=ProphetWrapper(model),
        artifact_path="model",
        flavor=mlflow.pyfunc,
        training_set=training_set,
        registered_model_name=f"{catalog}.{schema}.{model_name}"
    )

    print(f"Run ID: {run.info.run_id}")
    print(f"Model registered: {catalog}.{schema}.{model_name}")
    print(f"Metrics: {metrics}")