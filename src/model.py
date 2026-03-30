"""
Model training and evaluation module using Spark MLlib.

Trains both a Logistic Regression and a Random Forest classifier and
returns evaluation metrics for both models.
"""

from pyspark.sql import DataFrame
from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier,
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)

from src.config import LABEL_COL, RANDOM_SEED, TEST_SIZE


def split_data(df: DataFrame):
    """Split data into training and test sets.

    Args:
        df: Preprocessed DataFrame with a ``features`` column.

    Returns:
        Tuple (train_df, test_df).
    """
    train_df, test_df = df.randomSplit(
        [1.0 - TEST_SIZE, TEST_SIZE], seed=RANDOM_SEED
    )
    return train_df, test_df


def train_logistic_regression(train_df: DataFrame):
    """Train a logistic regression classifier.

    Args:
        train_df: Training DataFrame with ``features`` and label columns.

    Returns:
        Fitted LogisticRegressionModel.
    """
    lr = LogisticRegression(
        featuresCol="features",
        labelCol=LABEL_COL,
        maxIter=100,
        regParam=0.01,
        elasticNetParam=0.0,
    )
    return lr.fit(train_df)


def train_random_forest(train_df: DataFrame):
    """Train a random forest classifier.

    Args:
        train_df: Training DataFrame with ``features`` and label columns.

    Returns:
        Fitted RandomForestClassificationModel.
    """
    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol=LABEL_COL,
        numTrees=100,
        maxDepth=5,
        seed=RANDOM_SEED,
    )
    return rf.fit(train_df)


def evaluate_model(model, test_df: DataFrame, model_name: str) -> dict:
    """Evaluate a fitted classification model on the test set.

    Args:
        model: A fitted Spark ML classification model.
        test_df: Test DataFrame with ``features`` and label columns.
        model_name: A string label for the model (e.g. "LogisticRegression").

    Returns:
        Dictionary with evaluation metrics.
    """
    predictions = model.transform(test_df)

    binary_eval = BinaryClassificationEvaluator(
        labelCol=LABEL_COL, rawPredictionCol="rawPrediction"
    )
    auc = binary_eval.evaluate(predictions, {binary_eval.metricName: "areaUnderROC"})
    auc_pr = binary_eval.evaluate(predictions, {binary_eval.metricName: "areaUnderPR"})

    mc_eval = MulticlassClassificationEvaluator(
        labelCol=LABEL_COL, predictionCol="prediction"
    )
    accuracy = mc_eval.evaluate(predictions, {mc_eval.metricName: "accuracy"})
    f1 = mc_eval.evaluate(predictions, {mc_eval.metricName: "f1"})
    precision = mc_eval.evaluate(predictions, {mc_eval.metricName: "weightedPrecision"})
    recall = mc_eval.evaluate(predictions, {mc_eval.metricName: "weightedRecall"})

    metrics = {
        "model": model_name,
        "auc_roc": round(auc, 4),
        "auc_pr": round(auc_pr, 4),
        "accuracy": round(accuracy, 4),
        "f1_score": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }
    return metrics
