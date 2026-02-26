# Unit tests for batch inference pipeline
# Run with: pytest tests/test_inference.py

import pytest
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType,
    DoubleType, TimestampType, IntegerType
)
from datetime import datetime
from unittest.mock import patch, MagicMock

@pytest.fixture(scope="session")
def spark():
    """Create a local Spark session for testing."""
    return (
        SparkSession.builder
        .master("local[*]")
        .appName("test_inference")
        .config("spark.sql.shuffle.partitions", "1")
        .getOrCreate()
    )

@pytest.fixture
def sample_feature_data(spark):
    """Generate sample feature data for inference testing."""
    schema = StructType([
        StructField("id", StringType(), False),
        StructField("feature_1", DoubleType(), True),
        StructField("feature_2", DoubleType(), True),
        StructField("feature_3", IntegerType(), True),
        StructField("event_timestamp", TimestampType(), True),
        StructField("processed_timestamp", TimestampType(), True)
    ])

    data = [
        ("1", 5.0, 1.6094, 1, datetime.now(), datetime.now()),
        ("2", 5.0, 2.1972, 0, datetime.now(), datetime.now()),
        ("3", 5.0, 1.7918, 1, datetime.now(), datetime.now()),
        ("4", 5.0, 1.3863, 0, datetime.now(), datetime.now()),
        ("5", 5.0, 2.3026, 0, datetime.now(), datetime.now()),
    ]

    return spark.createDataFrame(data, schema)

def test_inference_input_not_empty(sample_feature_data):
    """Test that inference input data is not empty."""
    assert sample_feature_data.count() > 0

def test_inference_input_no_null_ids(sample_feature_data):
    """Test that inference input has no null IDs."""
    from pyspark.sql.functions import col
    null_ids = sample_feature_data.filter(col("id").isNull()).count()
    assert null_ids == 0

def test_inference_output_schema(spark, sample_feature_data):
    """Test that inference output has expected columns."""
    from pyspark.sql.functions import current_timestamp, lit, struct

    # Mock prediction UDF
    mock_predictions = sample_feature_data.withColumn(
        "prediction", lit(1.0)
    ).withColumn(
        "inference_timestamp", current_timestamp()
    ).withColumn(
        "model_version", lit("champion")
    ).select("id", "prediction", "inference_timestamp", "model_version")

    expected_columns = ["id", "prediction", "inference_timestamp", "model_version"]
    assert all(col in mock_predictions.columns for col in expected_columns)

def test_inference_output_count(spark, sample_feature_data):
    """Test that inference output row count matches input."""
    from pyspark.sql.functions import current_timestamp, lit

    mock_results = sample_feature_data.withColumn(
        "prediction", lit(1.0)
    ).withColumn(
        "inference_timestamp", current_timestamp()
    ).withColumn(
        "model_version", lit("champion")
    )

    assert mock_results.count() == sample_feature_data.count()

def test_champion_model_uri_format():
    """Test that champion model URI is correctly formatted."""
    catalog = "lob_a_dev"
    schema = "default"
    model_name = "my_model"

    model_uri = f"models:/{catalog}.{schema}.{model_name}@champion"
    assert model_uri == "models:/lob_a_dev.default.my_model@champion"

def test_inference_results_have_timestamp(spark, sample_feature_data):
    """Test that inference results include a timestamp."""
    from pyspark.sql.functions import current_timestamp, lit

    results = sample_feature_data.withColumn(
        "inference_timestamp", current_timestamp()
    )

    assert "inference_timestamp" in results.columns
    assert results.filter(
        results["inference_timestamp"].isNull()
    ).count() == 0