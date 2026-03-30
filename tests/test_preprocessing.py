"""
Unit tests for the preprocessing module.
"""

import os
import unittest

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

from src.ingestion import load_data
from src.preprocessing import (
    replace_zeros_with_null,
    impute_nulls_with_median,
    build_preprocessing_pipeline,
    preprocess,
)
from src.config import FEATURE_COLS

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "diabetes.csv"
)


class TestPreprocessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = (
            SparkSession.builder.appName("TestPreprocessing")
            .master("local[2]")
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )
        cls.spark.sparkContext.setLogLevel("ERROR")
        cls.raw_df = load_data(cls.spark, DATA_PATH)

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_replace_zeros_with_null_glucose(self):
        df = replace_zeros_with_null(self.raw_df)
        # After replacement there should be no zero Glucose values
        zero_glucose = df.filter(df["Glucose"] == 0).count()
        self.assertEqual(zero_glucose, 0)

    def test_impute_nulls_with_median_no_nulls_after(self):
        df = replace_zeros_with_null(self.raw_df)
        df = impute_nulls_with_median(df)
        for col in ["Glucose", "BloodPressure", "BMI"]:
            null_count = df.filter(df[col].isNull()).count()
            self.assertEqual(null_count, 0, f"Nulls remain in column '{col}'")

    def test_preprocess_adds_features_column(self):
        preprocessed_df, model = preprocess(self.raw_df)
        self.assertIn("features", preprocessed_df.columns)

    def test_preprocess_row_count_preserved(self):
        preprocessed_df, _ = preprocess(self.raw_df)
        # Rows should not increase after preprocessing
        self.assertLessEqual(preprocessed_df.count(), self.raw_df.count())
        self.assertGreater(preprocessed_df.count(), 0)

    def test_build_preprocessing_pipeline_stages(self):
        pipeline = build_preprocessing_pipeline()
        self.assertEqual(len(pipeline.getStages()), 2)

    def test_preprocess_returns_pipeline_model(self):
        _, model = preprocess(self.raw_df)
        self.assertIsInstance(model, PipelineModel)


if __name__ == "__main__":
    unittest.main()
