"""
MongoDB storage module: persists pipeline outputs to MongoDB collections.
"""

import logging
from typing import Any

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from pyspark.sql import DataFrame

from src.config import (
    MONGO_URI,
    MONGO_DB,
    MONGO_COLLECTION_PREPROCESSED,
    MONGO_COLLECTION_METRICS,
    MONGO_COLLECTION_PREDICTIONS,
    FEATURE_COLS,
    LABEL_COL,
)

logger = logging.getLogger(__name__)


def get_mongo_client(uri: str = MONGO_URI) -> MongoClient:
    """Create and return a MongoDB client.

    Args:
        uri: MongoDB connection URI.

    Returns:
        A connected MongoClient instance.

    Raises:
        ConnectionFailure: If the server is not reachable.
    """
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    return client


def save_metrics(metrics: dict, uri: str = MONGO_URI) -> None:
    """Save model evaluation metrics to MongoDB.

    Args:
        metrics: Dictionary of metric names to values.
        uri: MongoDB connection URI.
    """
    client = get_mongo_client(uri)
    try:
        db = client[MONGO_DB]
        collection = db[MONGO_COLLECTION_METRICS]
        collection.insert_one(metrics)
        logger.info("Saved metrics for model '%s'.", metrics.get("model", "unknown"))
    finally:
        client.close()


def save_predictions(predictions_df: DataFrame, uri: str = MONGO_URI) -> None:
    """Save prediction results (sample rows) to MongoDB.

    Converts the Spark DataFrame to a list of Python dicts and inserts them
    into MongoDB in batches.  Only scalar columns are stored; ML vector
    columns are excluded.

    Args:
        predictions_df: Spark DataFrame containing ``prediction`` and
            ``probability`` columns alongside the original features.
        uri: MongoDB connection URI.
    """
    # Select only serialisable columns
    cols_to_save = FEATURE_COLS + [LABEL_COL, "prediction"]
    available = [c for c in cols_to_save if c in predictions_df.columns]
    rows = [row.asDict() for row in predictions_df.select(available).collect()]

    client = get_mongo_client(uri)
    try:
        db = client[MONGO_DB]
        collection = db[MONGO_COLLECTION_PREDICTIONS]
        if rows:
            collection.insert_many(rows)
        logger.info("Saved %d prediction records to MongoDB.", len(rows))
    finally:
        client.close()


def save_preprocessed_sample(
    preprocessed_df: DataFrame,
    sample_fraction: float = 0.1,
    uri: str = MONGO_URI,
) -> None:
    """Save a sample of preprocessed records to MongoDB for inspection.

    Args:
        preprocessed_df: Preprocessed Spark DataFrame.
        sample_fraction: Fraction of rows to store (default 10 %).
        uri: MongoDB connection URI.
    """
    cols_to_save = FEATURE_COLS + [LABEL_COL]
    available = [c for c in cols_to_save if c in preprocessed_df.columns]
    sample = preprocessed_df.sample(fraction=sample_fraction, seed=42)
    rows = [row.asDict() for row in sample.select(available).collect()]

    client = get_mongo_client(uri)
    try:
        db = client[MONGO_DB]
        collection = db[MONGO_COLLECTION_PREPROCESSED]
        if rows:
            collection.insert_many(rows)
        logger.info(
            "Saved %d preprocessed sample records to MongoDB.", len(rows)
        )
    finally:
        client.close()
