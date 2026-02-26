# Clustering Model Evaluation
# Implements champion/challenger pattern using MLflow and Unity Catalog model registry
# Replace thresholds and metrics with your own requirements

import argparse
import mlflow
from mlflow import MlflowClient
from pyspark.sql import SparkSession
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--model-name", required=True)
    return parser.parse_args()

def load_test_data(spark, catalog, schema):
    """Load feature table from Unity Catalog."""
    return spark.table(f"{catalog}.{schema}.feature_table").toPandas()

def get_champion_model(client, model_uri):
    """Get the current champion model if one exists."""
    try:
        client.get_model_version_by_alias(model_uri, "champion")
        return mlflow.sklearn.load_model(f"models:/{model_uri}@champion")
    except Exception:
        return None

def get_challenger_model(client, model_uri):
    """Get the latest challenger model."""
    versions = client.search_model_versions(f"name='{model_uri}'")
    if not versions:
        raise ValueError(f"No model versions found for {model_uri}")
    latest = sorted(versions, key=lambda x: int(x.version), reverse=True)[0]
    return mlflow.sklearn.load_model(f"models:/{model_uri}/{latest.version}"), latest.version

def evaluate_model(model, df):
    """Evaluate clustering model on feature data."""
    feature_cols = [c for c in df.columns if c not in
                   ["id", "event_timestamp", "processed_timestamp"]]
    X = df[feature_cols].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    labels = model.predict(X_scaled)

    return {
        "silhouette_score": silhouette_score(X_scaled, labels),
        "davies_bouldin_score": davies_bouldin_score(X_scaled, labels),
        "inertia": model.inertia_,
        "n_clusters": model.n_clusters
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
    challenger_metrics = evaluate_model(challenger_model, df)
    print(f"Challenger metrics: {challenger_metrics}")

    # Compare with champion if one exists
    champion_model = get_champion_model(client, model_uri)

    if champion_model is None:
        print("No champion found - promoting challenger to champion")
        client.set_registered_model_alias(model_uri, "champion", challenger_version)
    else:
        champion_metrics = evaluate_model(champion_model, df)
        print(f"Champion metrics: {champion_metrics}")

        # Promote challenger if silhouette score improves by more than 1%
        # Adjust the threshold and metric to your own requirements
        if challenger_metrics["silhouette_score"] > champion_metrics["silhouette_score"] * 1.01:
            print("Challenger outperforms champion - promoting to champion")
            client.set_registered_model_alias(model_uri, "champion", challenger_version)
        else:
            print("Champion retained - challenger did not meet promotion threshold")

if __name__ == "__main__":
    main()