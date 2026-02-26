# Databricks notebook source
# Exploratory Data Analysis - Starting Point
# Use this notebook to explore your data before building pipelines
# This notebook is not part of the automated pipeline - it is for ad-hoc exploration only

# COMMAND ----------

# MAGIC %md
# MAGIC # {{cookiecutter.lob_display_name}} - {{cookiecutter.project_slug}}
# MAGIC ## Exploratory Data Analysis
# MAGIC
# MAGIC Use this notebook to:
# MAGIC - Explore raw data sources
# MAGIC - Understand data distributions and quality
# MAGIC - Prototype feature engineering logic
# MAGIC - Validate assumptions before building production pipelines

# COMMAND ----------

# Setup - replace with your actual catalog and schema
catalog = "{{cookiecutter.catalog_name}}_dev"
schema = "{{cookiecutter.schema_name}}"

print(f"Using catalog: {catalog}")
print(f"Using schema: {schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Data Discovery

# COMMAND ----------

# List available tables in your catalog
display(spark.sql(f"SHOW TABLES IN {catalog}.{schema}"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Raw Data Exploration
# MAGIC Replace the table name with your actual source table

# COMMAND ----------

# Load and inspect raw data
# df_raw = spark.table(f"{catalog}.{schema}.your_table_name")
# display(df_raw.limit(100))
# print(f"Row count: {df_raw.count()}")
# print(f"Schema: {df_raw.printSchema()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Data Quality Checks

# COMMAND ----------

# Check for nulls
# null_counts = df_raw.select(
#     [F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df_raw.columns]
# )
# display(null_counts)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Feature Distribution Analysis

# COMMAND ----------

# Summarise numeric features
# display(df_raw.describe())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Target Variable Analysis
# MAGIC Replace 'target' with your actual target column

# COMMAND ----------

# Analyse target distribution
# display(df_raw.groupBy("target").count().orderBy("target"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Prototype Feature Engineering
# MAGIC Use this section to prototype transformations before adding them to the DLT pipeline

# COMMAND ----------

# from pyspark.sql import functions as F
#
# df_features = (
#     df_raw
#     .withColumn("feature_1", F.col("value_1") / F.col("value_2"))
#     .withColumn("feature_2", F.log1p(F.col("value_3")))
# )
# display(df_features.limit(100))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Notes
# MAGIC <!-- Add your findings and observations here -->