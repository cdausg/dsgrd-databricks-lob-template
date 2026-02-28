# Databricks notebook source
# Batch Inference Pipeline
# Uses the champion model from Unity Catalog to generate predictions
# Results are written to a Gold table for BI tool consumption (Live Lake model)

import dlt
import mlflow
import mlflow.sklearn
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import pandas_udf

# Load champion model inside the UDF so each executor loads it independently.
# Broadcast variables are unreliable in DLT pipelines due to lifecycle management.
def make_predict_udf(catalog, schema, model_name, feature_cols):
    model_uri = f"models:/{catalog}.{schema}.{model_name}@champion"
    col_names = list(feature_cols)

    @pandas_udf(DoubleType())
    def predict(*cols):
        import mlflow
        import mlflow.sklearn
        from sklearn.tree import DecisionTreeClassifier
        mlflow.set_tracking_uri("databricks")
        mlflow.set_registry_uri("databricks-uc")
        model = mlflow.sklearn.load_model(model_uri)
        # Patch internal DecisionTreeClassifier estimators that were pickled with
        # an older sklearn version (< 1.2) which did not have the monotonic_cst
        # attribute. This allows models trained on older runtimes to run on newer ones.
        if hasattr(model, "estimators_"):
            for est in model.estimators_:
                if not hasattr(est, "monotonic_cst"):
                    est.monotonic_cst = None
        X = pd.concat(cols, axis=1)
        X.columns = col_names
        return pd.Series(model.predict(X).astype(float))

    return predict

# ---------------------------------------------------------------
# Silver layer - prepared inference input
# Reads from the feature table produced by the feature pipeline
# ---------------------------------------------------------------

@dlt.table(
    name="inference_input",
    comment="Prepared input data for model inference",
    table_properties={"quality": "silver"}
)
def inference_input():
    catalog = spark.conf.get("pipeline.catalog")
    schema = spark.conf.get("pipeline.schema", "default")
    feature_table = f"{catalog}.{schema}.{{cookiecutter.project_slug}}_features"
    return (
        spark.read.table(feature_table)
        .filter(F.col("id").isNotNull())
    )

# ---------------------------------------------------------------
# Gold layer - inference results
# This is the Result Layer that BI tools connect to directly
# following the Live Lake model - no data movement required
# ---------------------------------------------------------------

@dlt.table(
    name="inference_results",
    comment="Model predictions - Gold table for BI consumption",
    table_properties={
        "quality": "gold",
        "delta.enableChangeDataFeed": "true"
    }
)
def inference_results():
    # Use only numeric columns, excluding non-feature columns.
    # Must match the feature set used during training (numeric-only, no timestamps).
    _exclude = {"id", "feature_updated_timestamp", "event_timestamp", "processed_timestamp"}
    input_df = dlt.read("inference_input")
    feature_cols = [c for c in input_df.columns
                   if c not in _exclude and dict(input_df.dtypes)[c] in ("int", "bigint", "double", "float")]

    predict_udf = make_predict_udf(
        catalog=spark.conf.get("pipeline.catalog"),
        schema=spark.conf.get("pipeline.schema", "default"),
        model_name=spark.conf.get("pipeline.model_name"),
        feature_cols=feature_cols
    )

    return (
        input_df
        .withColumn("prediction", predict_udf(*[F.col(c) for c in feature_cols]))
        .withColumn("inference_timestamp", F.current_timestamp())
        .withColumn("model_version", F.lit("champion"))
        .select(
            "id",
            "prediction",
            "inference_timestamp",
            "model_version",
        )
    )