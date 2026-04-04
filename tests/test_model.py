"""
Unit tests for the model training and evaluation module.
"""

import os
import unittest

from pyspark.sql import SparkSession
from pyspark.ml.classification import (
    LogisticRegressionModel,
    RandomForestClassificationModel,
)

from src.ingestion import load_data
from src.preprocessing import preprocess
from src.model import (
    split_data,
    train_logistic_regression,
    train_random_forest,
    evaluate_model,
)

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "diabetes.csv"
)


class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = (
            SparkSession.builder.appName("TestModel")
            .master("local[2]")
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )
        cls.spark.sparkContext.setLogLevel("ERROR")
        raw_df = load_data(cls.spark, DATA_PATH)
        cls.preprocessed_df, _ = preprocess(raw_df)
        cls.train_df, cls.test_df = split_data(cls.preprocessed_df)

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_split_data_sizes(self):
        total = self.preprocessed_df.count()
        train_n = self.train_df.count()
        test_n = self.test_df.count()
        # Allow rounding: train + test must equal total
        self.assertEqual(train_n + test_n, total)

    def test_train_logistic_regression_returns_model(self):
        model = train_logistic_regression(self.train_df)
        self.assertIsInstance(model, LogisticRegressionModel)

    def test_train_random_forest_returns_model(self):
        model = train_random_forest(self.train_df)
        self.assertIsInstance(model, RandomForestClassificationModel)

    def test_evaluate_logistic_regression_metrics(self):
        model = train_logistic_regression(self.train_df)
        metrics = evaluate_model(model, self.test_df, "LogisticRegression")
        self.assertEqual(metrics["model"], "LogisticRegression")
        for key in ("auc_roc", "auc_pr", "accuracy", "f1_score", "precision", "recall"):
            self.assertIn(key, metrics)
            self.assertGreaterEqual(metrics[key], 0.0)
            self.assertLessEqual(metrics[key], 1.0)

    def test_evaluate_random_forest_metrics(self):
        model = train_random_forest(self.train_df)
        metrics = evaluate_model(model, self.test_df, "RandomForest")
        self.assertEqual(metrics["model"], "RandomForest")
        self.assertGreater(metrics["auc_roc"], 0.5)


if __name__ == "__main__":
    unittest.main()
