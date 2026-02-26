# Time Series Model Training
# Uses Prophet for forecasting and MLflow for experiment tracking
# Replace the model and features with your own implementation

import argparse
import mlflow
import mlflow.pyfunc
from pyspark.sql import SparkSession
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--experiment-path", required=True)
    parser.add_argument("--horizon", type=int, default=30, help="Forecast horizon in days")
    return parser.parse_args()

def load_features(spark, catalog, schema):
    """Load time series feature table from Unity Catalog.
    Expected columns: ds (datetime), y (target value), and optional regressors.
    """
    return spark.table(f"{catalog}.{schema}.feature_table").toPandas()

def train(df, horizon):
    """Train a Prophet forecasting model - replace with your own implementation."""
    # Prophet expects columns named 'ds' (datetime) and 'y' (target)
    # Rename your columns accordingly before passing to this function
    train_df = df[:-horizon]
    test_df = df[-horizon:]

    model = Prophet(
        seasonality_mode="multiplicative",
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )
    model.fit(train_df)

    future = model.make_future_dataframe(periods=horizon)
    forecast = model.predict(future)

    y_true = test_df["y"].values
    y_pred = forecast[-horizon:]["yhat"].values

    metrics = {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mape": np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }

    return model, metrics

class ProphetWrapper(mlflow.pyfunc.PythonModel):
    """MLflow wrapper for Prophet model to enable Unity Catalog registration."""
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        future = self.model.make_future_dataframe(periods=len(model_input))
        forecast = self.model.predict(future)
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]

def main():
    args = parse_args()
    spark = SparkSession.builder.getOrCreate()

    mlflow.set_experiment(args.experiment_path)
    mlflow.set_registry_uri("databricks-uc")

    with mlflow.start_run():
        df = load_features(spark, args.catalog, args.schema)
        model, metrics = train(df, args.horizon)

        mlflow.log_params({
            "catalog": args.catalog,
            "schema": args.schema,
            "model_type": "Prophet",
            "horizon": args.horizon,
            "seasonality_mode": "multiplicative"
        })
        mlflow.log_metrics(metrics)

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=ProphetWrapper(model),
            registered_model_name=f"{args.catalog}.{args.schema}.{args.model_name}"
        )

        print(f"Model registered: {args.catalog}.{args.schema}.{args.model_name}")
        print(f"Metrics: {metrics}")

if __name__ == "__main__":
    main()