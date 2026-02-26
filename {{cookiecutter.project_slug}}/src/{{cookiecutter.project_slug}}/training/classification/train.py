# Classification Model Training
# Uses MLflow for experiment tracking and Unity Catalog for model registry
# Replace the model and features with your own implementation

import argparse
import mlflow
import mlflow.sklearn
from pyspark.sql import SparkSession
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--experiment-path", required=True)
    return parser.parse_args()

def load_features(spark, catalog, schema):
    """Load feature table from Unity Catalog."""
    return spark.table(f"{catalog}.{schema}.feature_table").toPandas()

def train(df):
    """Train a classification model - replace with your own implementation."""
    # Replace 'target' with your actual target column
    X = df.drop(columns=["target", "id", "event_timestamp", "processed_timestamp"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    metrics = {
        "accuracy": accuracy_score(y_test, model.predict(X_test)),
        "f1_score": f1_score(y_test, model.predict(X_test), average="weighted"),
        "roc_auc": roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    }

    return model, metrics, X_train

def main():
    args = parse_args()
    spark = SparkSession.builder.getOrCreate()

    mlflow.set_experiment(args.experiment_path)
    mlflow.set_registry_uri("databricks-uc")

    with mlflow.start_run():
        df = load_features(spark, args.catalog, args.schema)
        model, metrics, X_train = train(df)

        mlflow.log_params({
            "catalog": args.catalog,
            "schema": args.schema,
            "model_type": "RandomForestClassifier",
            "n_estimators": 100
        })
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=f"{args.catalog}.{args.schema}.{args.model_name}",
            input_example=X_train.head(5)
        )

        print(f"Model registered: {args.catalog}.{args.schema}.{args.model_name}")
        print(f"Metrics: {metrics}")

if __name__ == "__main__":
    main()