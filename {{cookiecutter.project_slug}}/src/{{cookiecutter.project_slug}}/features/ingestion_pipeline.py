# Databricks notebook source
# DLT Ingestion Pipeline - Bronze and Silver layers
# Handles raw data ingestion and initial cleaning
# Feature engineering is handled separately by feature_pipeline.py

# COMMAND ----------

# MAGIC %md
# MAGIC # {{cookiecutter.lob_display_name}} - Data Ingestion Pipeline
# MAGIC
# MAGIC This DLT pipeline handles:
# MAGIC - **Bronze layer**: Raw data ingestion from source systems
# MAGIC - **Silver layer**: Data cleaning, validation and deduplication
# MAGIC
# MAGIC Engineered features are written to the Feature Store
# MAGIC by the feature_table_job which runs after this pipeline.

# COMMAND ----------

import dlt
from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bronze Layer - Raw Ingestion
# MAGIC Replace the source path and format with your actual data source

# COMMAND ----------

@dlt.table(
    name="raw_data",
    comment="Raw data ingested from source system",
    table_properties={"quality": "bronze"}
)
def raw_data():
    catalog = spark.conf.get("pipeline.catalog")
    schema = spark.conf.get("pipeline.schema", "default")
    return (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "csv")
        .option("cloudFiles.inferColumnTypes", "true")
        # Source path: abfss://container@storageaccount.dfs.core.windows.net/path
        .load(f"/Volumes/{catalog}/{schema}/raw/")
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Silver Layer - Cleaned Data
# MAGIC Add your own data quality expectations and transformations

# COMMAND ----------

@dlt.table(
    name="cleaned_data",
    comment="Cleaned and validated data ready for feature engineering",
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