# Unit tests for feature engineering pipeline
# Run with: pytest tests/test_features.py

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType
from datetime import datetime

@pytest.fixture(scope="session")
def spark():
    """Create a local Spark session for testing."""
    return (
        SparkSession.builder
        .master("local[*]")
        .appName("test_features")
        .config("spark.sql.shuffle.partitions", "1")
        .getOrCreate()
    )

@pytest.fixture
def sample_raw_data(spark):
    """Generate sample raw data for testing."""
    schema = StructType([
        StructField("id", StringType(), False),
        StructField("value_1", DoubleType(), True),
        StructField("value_2", DoubleType(), True),
        StructField("value_3", DoubleType(), True),
        StructField("category", StringType(), True),
        StructField("event_timestamp", TimestampType(), True)
    ])

    data = [
        ("1", 10.0, 2.0, 5.0, "A", datetime.now()),
        ("2", 20.0, 4.0, 8.0, "B", datetime.now()),
        ("3", 15.0, 3.0, 6.0, "A", datetime.now()),
        ("4", None, 2.0, 4.0, "C", datetime.now()),  # Test null handling
        ("5", 25.0, 5.0, 10.0, "B", datetime.now()),
    ]

    return spark.createDataFrame(data, schema)

def test_raw_data_schema(sample_raw_data):
    """Test that raw data has expected columns."""
    expected_columns = ["id", "value_1", "value_2", "value_3", "category", "event_timestamp"]
    assert all(col in sample_raw_data.columns for col in expected_columns)

def test_no_duplicate_ids(sample_raw_data):
    """Test that deduplication removes duplicate IDs."""
    from pyspark.sql.functions import col
    deduped = sample_raw_data.dropDuplicates(["id"])
    assert deduped.count() == sample_raw_data.count()

def test_feature_engineering(spark, sample_raw_data):
    """Test feature engineering transformations."""
    from pyspark.sql.functions import col, log1p, when

    result = (
        sample_raw_data
        .filter(col("id").isNotNull())
        .withColumn("feature_1", col("value_1") / col("value_2"))
        .withColumn("feature_2", log1p(col("value_3")))
        .withColumn("feature_3", when(col("category") == "A", 1).otherwise(0))
    )

    # Check feature columns exist
    assert "feature_1" in result.columns
    assert "feature_2" in result.columns
    assert "feature_3" in result.columns

def test_category_encoding(spark, sample_raw_data):
    """Test that category A is encoded as 1 and others as 0."""
    from pyspark.sql.functions import col, when

    result = sample_raw_data.withColumn(
        "feature_3", when(col("category") == "A", 1).otherwise(0)
    )

    category_a_rows = result.filter(col("category") == "A").select("feature_3").collect()
    assert all(row["feature_3"] == 1 for row in category_a_rows)

    category_b_rows = result.filter(col("category") == "B").select("feature_3").collect()
    assert all(row["feature_3"] == 0 for row in category_b_rows)