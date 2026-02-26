# Unit tests for model training
# Run with: pytest tests/test_training.py

import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------
# Classification tests
# ---------------------------------------------------------------

@pytest.fixture
def sample_classification_data():
    """Generate sample classification data for testing."""
    np.random.seed(42)
    n_samples = 100
    return pd.DataFrame({
        "feature_1": np.random.randn(n_samples),
        "feature_2": np.random.randn(n_samples),
        "feature_3": np.random.randint(0, 2, n_samples),
        "target": np.random.randint(0, 2, n_samples),
        "id": [str(i) for i in range(n_samples)],
        "event_timestamp": pd.Timestamp.now(),
        "processed_timestamp": pd.Timestamp.now()
    })

def test_classification_model_trains(sample_classification_data):
    """Test that classification model trains without errors."""
    df = sample_classification_data
    X = df.drop(columns=["target", "id", "event_timestamp", "processed_timestamp"])
    y = df["target"]

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    assert model is not None
    assert hasattr(model, "predict")

def test_classification_model_predicts(sample_classification_data):
    """Test that classification model produces valid predictions."""
    df = sample_classification_data
    X = df.drop(columns=["target", "id", "event_timestamp", "processed_timestamp"])
    y = df["target"]

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    predictions = model.predict(X)

    assert len(predictions) == len(df)
    assert all(p in [0, 1] for p in predictions)

def test_classification_metrics_range(sample_classification_data):
    """Test that classification metrics are within valid ranges."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score

    df = sample_classification_data
    X = df.drop(columns=["target", "id", "event_timestamp", "processed_timestamp"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    f1 = f1_score(y_test, model.predict(X_test), average="weighted")

    assert 0.0 <= accuracy <= 1.0
    assert 0.0 <= f1 <= 1.0

# ---------------------------------------------------------------
# Clustering tests
# ---------------------------------------------------------------

@pytest.fixture
def sample_clustering_data():
    """Generate sample clustering data for testing."""
    np.random.seed(42)
    n_samples = 100
    return pd.DataFrame({
        "feature_1": np.random.randn(n_samples),
        "feature_2": np.random.randn(n_samples),
        "feature_3": np.random.randn(n_samples),
        "id": [str(i) for i in range(n_samples)],
        "event_timestamp": pd.Timestamp.now(),
        "processed_timestamp": pd.Timestamp.now()
    })

def test_clustering_model_trains(sample_clustering_data):
    """Test that clustering model trains without errors."""
    df = sample_clustering_data
    feature_cols = [c for c in df.columns if c not in
                   ["id", "event_timestamp", "processed_timestamp"]]
    X = df[feature_cols].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=3, random_state=42, n_init=10)
    model.fit(X_scaled)

    assert model is not None
    assert model.n_clusters == 3

def test_clustering_produces_valid_labels(sample_clustering_data):
    """Test that clustering produces valid cluster labels."""
    df = sample_clustering_data
    feature_cols = [c for c in df.columns if c not in
                   ["id", "event_timestamp", "processed_timestamp"]]
    X = df[feature_cols].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_clusters = 3
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)

    assert len(labels) == len(df)
    assert all(0 <= label < n_clusters for label in labels)

# ---------------------------------------------------------------
# Time series tests
# ---------------------------------------------------------------

@pytest.fixture
def sample_time_series_data():
    """Generate sample time series data for testing."""
    dates = pd.date_range(start="2023-01-01", periods=365, freq="D")
    np.random.seed(42)
    return pd.DataFrame({
        "ds": dates,
        "y": np.random.randn(365).cumsum() + 100
    })

def test_time_series_data_format(sample_time_series_data):
    """Test that time series data has required Prophet columns."""
    df = sample_time_series_data
    assert "ds" in df.columns
    assert "y" in df.columns
    assert len(df) > 0

def test_time_series_no_nulls(sample_time_series_data):
    """Test that time series data has no null values in key columns."""
    df = sample_time_series_data
    assert df["ds"].isnull().sum() == 0
    assert df["y"].isnull().sum() == 0