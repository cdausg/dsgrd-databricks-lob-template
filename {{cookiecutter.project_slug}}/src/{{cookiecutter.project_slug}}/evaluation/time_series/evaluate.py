# Time Series Model Evaluation
# Implements champion/challenger pattern using MLflow and Unity Catalog model registry
# Replace thresholds and metrics with your own requirements

import argparse
import mlflow
from mlflow import MlflowClient
from pyspark.sql import SparkSession
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--horizon", type=int, default=30)
    return parser.parse_args()

def load_test_data(spark, catalog, schema):
    """Load time series test data from Unity Catalog."""
    return spark.table(f"{catalog}.{schema}.feature_table").toPandas()

def get_champion_model(client, model_uri):
    """Get the current champion model if one exists."""
    try:
        client.get_model_version_by_alias(model_uri, "champion")
        return mlflow.pyfunc.load_model(f"models:/{model_uri}@champion")
    except Exception:
        return None

def get_challenger_model(client, model_uri):
    """Get the latest challenger model."""
    versions = client.search_model_versions(f"name='{model_uri}'")
    if not versions:
        raise ValueError(f"No model versions found for {model_uri}")
    latest = sorted(versions, key=lambda x: int(x.version), reverse=True)[0]
    return mlflow.pyfunc.load_model(f"models:/{model_uri}/{latest.version}"), latest.version

def evaluate_model(model, df, horizon):
    """Evaluate time series model on holdout period."""
    test_df = df[-horizon:]
    future = pd.DataFrame({"ds": test_df["ds"]})
    forecast = model.predict(future)

    y_true = test_df["y"].values
    y_pred = forecast["yhat"].values[-horizon:]

    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mape": np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }

def main():
    args = parse_args()
    spark = SparkSession.builder.getOrCreate()

    mlflow.set_registry_uri("databricks-uc")
    client = MlflowClient()
    model_uri = f"{args.catalog}.{args.schema}.{args.model_name}"

    df = load_test_data(spark, args.catalog, args.schema)

    # Evaluate challenger
    challenger_model, challenger_version = get_challenger_model(client, model_uri)
    challenger_metrics = evaluate_model(challenger_model, df, args.horizon)
    print(f"Challenger metrics: {challenger_metrics}")

    # Compare with champion if one exists
    champion_model = get_champion_model(client, model_uri)

    if champion_model is None:
        print("No champion found - promoting challenger to champion")
        client.set_registered_model_alias(model_uri, "champion", challenger_version)
    else:
        champion_metrics = evaluate_model(champion_model, df, args.horizon)
        print(f"Champion metrics: {champion_metrics}")

        # Promote challenger if MAE improves by more than 1%
        # Adjust the threshold and metric to your own requirements
        if challenger_metrics["mae"] < champion_metrics["mae"] * 0.99:
            print("Challenger outperforms champion - promoting to champion")
            client.set_registered_model_alias(model_uri, "champion", challenger_version)
        else:
            print("Champion retained - challenger did not meet promotion threshold")

if __name__ == "__main__":
    main()