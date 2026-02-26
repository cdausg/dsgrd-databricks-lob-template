# Databricks notebook source
# Delta Live Tables - Feature Engineering Pipeline
# Replace this example with your own feature definitions

import dlt
from pyspark.sql import functions as F

# ---------------------------------------------------------------
# Bronze layer - raw data ingestion
# Replace the source path and schema with your own data source
# ---------------------------------------------------------------

@dlt.table(
    name="raw_data",
    comment="Raw data ingested from source system",
    table_properties={"quality": "bronze"}
)
def raw_data():
    return (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "csv")
        .option("cloudFiles.inferColumnTypes", "true")
        .load("/Volumes/{{cookiecutter.catalog_name}}/{{cookiecutter.schema_name}}/raw/")
    )

# ---------------------------------------------------------------
# Silver layer - cleaned and validated data
# Add your own data quality expectations and transformations
# ---------------------------------------------------------------

@dlt.table(
    name="cleaned_data",
    comment="Cleaned and validated data",
    table_properties={"quality": "silver"}
)
@dlt.expect_or_drop("valid_id", "id IS NOT NULL")
@dlt.expect_or_drop("valid_timestamp", "event_timestamp IS NOT NULL")
def cleaned_data():
    return (
        dlt.read_stream("raw_data")
        .withColumn("processed_timestamp", F.current_timestamp())
        .dropDuplicates(["id"])
    )

# ---------------------------------------------------------------
# Gold layer - feature table for ML training
# Add your own feature engineering logic here
# ---------------------------------------------------------------

@dlt.table(
    name="feature_table",
    comment="Feature table for ML model training",
    table_properties={"quality": "gold"}
)
def feature_table():
    return (
        dlt.read("cleaned_data")
        .withColumn("feature_1", F.col("value_1") / F.col("value_2"))
        .withColumn("feature_2", F.log1p(F.col("value_3")))
        .withColumn("feature_3", F.when(F.col("category") == "A", 1).otherwise(0))
    )