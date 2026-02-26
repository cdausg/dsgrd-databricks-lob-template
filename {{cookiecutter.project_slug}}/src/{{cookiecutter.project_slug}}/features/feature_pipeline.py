# Databricks notebook source
# Feature Engineering Pipeline
# Uses Databricks Feature Engineering API to create and register feature tables
# This enables feature sharing and reuse across projects and LOBs

# COMMAND ----------

# MAGIC %md
# MAGIC # {{cookiecutter.lob_display_name}} - Feature Engineering
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Reads raw data from the bronze/silver layer
# MAGIC 2. Engineers features
# MAGIC 3. Registers features in the Databricks Feature Store
# MAGIC
# MAGIC Feature Store tables are discoverable and reusable across projects.
# MAGIC Contact the hub platform team to share features with other LOBs.

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

# Get parameters - set defaults for interactive development
dbutils.widgets.text("catalog", "{{cookiecutter.catalog_name}}_dev")
dbutils.widgets.text("schema", "{{cookiecutter.schema_name}}")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

print(f"Catalog: {catalog}")
print(f"Schema: {schema}")

fe = FeatureEngineeringClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Read Source Data
# MAGIC Replace this with your actual data source

# COMMAND ----------

# Read from silver layer - replace with your actual table
# For streaming sources use readStream instead
df_source = spark.table(f"{catalog}.{schema}.cleaned_data")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Engineer Features
# MAGIC Replace these transformations with your own feature engineering logic

# COMMAND ----------

df_features = (
    df_source
    .withColumn("feature_1", F.col("value_1") / F.col("value_2"))
    .withColumn("feature_2", F.log1p(F.col("value_3")))
    .withColumn("feature_3", F.when(F.col("category") == "A", 1).otherwise(0))
    .withColumn("feature_updated_timestamp", F.current_timestamp())
    # Select only the columns needed for the feature table
    # The primary key column (id) must be included
    .select(
        "id",
        "feature_1",
        "feature_2",
        "feature_3",
        "feature_updated_timestamp"
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create or Update Feature Table

# COMMAND ----------

feature_table_name = f"{catalog}.{schema}.{{cookiecutter.project_slug}}_features"

# Create feature table if it doesn't exist
try:
    fe.get_table(name=feature_table_name)
    print(f"Feature table already exists: {feature_table_name}")
except Exception:
    print(f"Creating feature table: {feature_table_name}")
    fe.create_table(
        name=feature_table_name,
        primary_keys=["id"],
        timestamp_keys=["feature_updated_timestamp"],
        schema=df_features.schema,
        description=f"Feature table for {{cookiecutter.project_slug}} - managed by {{cookiecutter.lob_display_name}}"
    )

# COMMAND ----------

# Write features to the feature table
fe.write_table(
    name=feature_table_name,
    df=df_features,
    mode="merge"  # Use 'overwrite' for full refresh
)

print(f"Features written to: {feature_table_name}")
print(f"Row count: {df_features.count()}")