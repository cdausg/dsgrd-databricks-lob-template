# Lakehouse Monitoring Setup
# Configures Databricks Lakehouse Monitoring on feature and inference tables
# This implements the Standard-Only monitoring model - built-in drift detection only
# Ground truth / business accuracy monitoring is out of scope for the central platform

import argparse
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    MonitorInferenceLog,
    MonitorInferenceLogProblemType,
    MonitorSnapshot
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--table", required=True)
    parser.add_argument("--model-name", required=True)
    return parser.parse_args()

def setup_feature_monitoring(w, catalog, schema, table):
    """
    Set up snapshot monitoring on the feature table.
    Detects data drift in input features.
    """
    table_name = f"{catalog}.{schema}.{table}"
    assets_dir = f"/Shared/{catalog}/{schema}/{table}/monitoring"

    print(f"Setting up feature monitoring on {table_name}")

    try:
        w.quality_monitors.create(
            table_name=table_name,
            assets_dir=assets_dir,
            snapshot=MonitorSnapshot()
        )
        print(f"Feature monitoring created for {table_name}")
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"Monitor already exists for {table_name} - skipping")
        else:
            raise e

def setup_inference_monitoring(w, catalog, schema, table, model_name):
    """
    Set up inference log monitoring on the inference results table.
    Detects model drift and prediction distribution shifts.

    Note: This monitors prediction distributions only.
    Ground truth tracking is the responsibility of the project team
    and is out of scope for the central platform (Standard-Only model).
    """
    table_name = f"{catalog}.{schema}.{table}"
    assets_dir = f"/Shared/{catalog}/{schema}/{table}/monitoring"

    print(f"Setting up inference monitoring on {table_name}")

    try:
        w.quality_monitors.create(
            table_name=table_name,
            assets_dir=assets_dir,
            inference_log=MonitorInferenceLog(
                # Replace with your actual column names
                timestamp_col="inference_timestamp",
                prediction_col="prediction",
                model_id_col="model_version",
                # Set problem type based on your use case:
                # PROBLEM_TYPE_CLASSIFICATION or PROBLEM_TYPE_REGRESSION
                problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_CLASSIFICATION
            )
        )
        print(f"Inference monitoring created for {table_name}")
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"Monitor already exists for {table_name} - skipping")
        else:
            raise e

def main():
    args = parse_args()
    w = WorkspaceClient()

    if args.table == "feature_table":
        setup_feature_monitoring(w, args.catalog, args.schema, args.table)
    elif args.table == "inference_results":
        setup_inference_monitoring(w, args.catalog, args.schema, args.table, args.model_name)
    else:
        # Default to snapshot monitoring for any other table
        setup_feature_monitoring(w, args.catalog, args.schema, args.table)

if __name__ == "__main__":
    main()