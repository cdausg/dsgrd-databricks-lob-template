# Databricks notebook source
# Batch Inference Pipeline
# Uses the champion model from Unity Catalog to generate predictions
# Results are written to a Gold table for BI tool consumption (Live Lake model)

import dlt
import mlflow
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType

# Load champion model from Unity Catalog as a Spark UDF
# This ensures the latest approved model is always used
def load_champion_model(catalog, schema, model_name):
    model_uri = f"models:/{catalog}.{schema}.{model_name}@champion"
    return mlflow.pyfunc.spark_udf(spark, model_uri=model_uri)

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
    # Load champion model
    # Replace catalog/schema/model_name with your actual values
    # or pass them in via pipeline parameters
    predict_udf = load_champion_model(
        catalog=spark.conf.get("pipeline.catalog"),
        schema=spark.conf.get("pipeline.schema", "default"),
        model_name=spark.conf.get("pipeline.model_name")
    )

    # Use only numeric columns, excluding non-feature columns.
    # Must match the feature set used during training (numeric-only, no timestamps).
    _exclude = {"id", "feature_updated_timestamp", "event_timestamp", "processed_timestamp"}
    input_df = dlt.read("inference_input")
    feature_cols = [c for c in input_df.columns
                   if c not in _exclude and dict(input_df.dtypes)[c] in ("int", "bigint", "double", "float")]

    return (
        dlt.read("inference_input")
        .withColumn("prediction", predict_udf(F.struct(*feature_cols)))
        .withColumn("inference_timestamp", F.current_timestamp())
        .withColumn("model_version", F.lit("champion"))
        # Result layer - ready for direct BI tool consumption
        .select(
            "id",
            "prediction",
            "inference_timestamp",
            "model_version",
            # Add any additional columns needed by BI tools here
        )
    )