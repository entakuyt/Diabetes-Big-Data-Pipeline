"""
Main pipeline orchestrator: ties together ingestion, preprocessing,
model training, evaluation and storage.
"""

import logging
import sys

from pyspark.sql import SparkSession

from src import config
from src.ingestion import load_data
from src.preprocessing import preprocess
from src.model import split_data, train_logistic_regression, train_random_forest, evaluate_model
from src.storage import save_metrics, save_predictions, save_preprocessed_sample

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)


def create_spark_session() -> SparkSession:
    """Create and return a SparkSession."""
    spark = (
        SparkSession.builder.appName(config.SPARK_APP_NAME)
        .master(config.SPARK_MASTER)
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def run(data_path: str = config.DATA_PATH, persist_to_mongo: bool = True) -> dict:
    """Execute the full pipeline end-to-end.

    Args:
        data_path: Path to the diabetes CSV file.
        persist_to_mongo: Whether to write results to MongoDB.

    Returns:
        Dictionary mapping model names to their evaluation metrics dicts.
    """
    spark = create_spark_session()
    logger.info("SparkSession created.")

    # 1. Ingestion
    logger.info("Loading data from '%s'.", data_path)
    raw_df = load_data(spark, data_path)
    logger.info("Raw dataset: %d rows, %d columns.", raw_df.count(), len(raw_df.columns))

    # 2. Preprocessing
    logger.info("Preprocessing data …")
    preprocessed_df, _ = preprocess(raw_df)

    if persist_to_mongo:
        try:
            save_preprocessed_sample(preprocessed_df)
        except Exception as exc:
            logger.warning("Could not save preprocessed sample to MongoDB: %s", exc)

    # 3. Train / test split
    train_df, test_df = split_data(preprocessed_df)
    logger.info(
        "Train/test split: %d / %d rows.", train_df.count(), test_df.count()
    )

    results = {}

    # 4a. Logistic Regression
    logger.info("Training Logistic Regression …")
    lr_model = train_logistic_regression(train_df)
    lr_metrics = evaluate_model(lr_model, test_df, "LogisticRegression")
    logger.info("LR metrics: %s", lr_metrics)
    results["LogisticRegression"] = lr_metrics

    if persist_to_mongo:
        try:
            save_metrics(lr_metrics)
            save_predictions(lr_model.transform(test_df))
        except Exception as exc:
            logger.warning("Could not persist LR results to MongoDB: %s", exc)

    # 4b. Random Forest
    logger.info("Training Random Forest …")
    rf_model = train_random_forest(train_df)
    rf_metrics = evaluate_model(rf_model, test_df, "RandomForest")
    logger.info("RF metrics: %s", rf_metrics)
    results["RandomForest"] = rf_metrics

    if persist_to_mongo:
        try:
            save_metrics(rf_metrics)
        except Exception as exc:
            logger.warning("Could not persist RF results to MongoDB: %s", exc)

    spark.stop()
    return results


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else config.DATA_PATH
    metrics = run(data_path=data_path, persist_to_mongo=True)
    print("\n=== Pipeline Results ===")
    for model_name, m in metrics.items():
        print(f"\n{model_name}:")
        for k, v in m.items():
            print(f"  {k}: {v}")
