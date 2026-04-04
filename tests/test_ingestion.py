"""
Unit tests for the data ingestion module.
"""

import os
import unittest

from pyspark.sql import SparkSession

from src.ingestion import load_data, get_expected_columns
from src.config import FEATURE_COLS, LABEL_COL

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "diabetes.csv"
)


class TestIngestion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = (
            SparkSession.builder.appName("TestIngestion")
            .master("local[2]")
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_load_data_returns_dataframe(self):
        df = load_data(self.spark, DATA_PATH)
        self.assertIsNotNone(df)
        self.assertGreater(df.count(), 0)

    def test_load_data_columns(self):
        df = load_data(self.spark, DATA_PATH)
        expected = get_expected_columns()
        for col in expected:
            self.assertIn(col, df.columns, f"Missing column: {col}")

    def test_load_data_no_extra_rows(self):
        df = load_data(self.spark, DATA_PATH)
        # All rows must have the label column present (non-null after correct schema)
        null_labels = df.filter(df[LABEL_COL].isNull()).count()
        self.assertEqual(null_labels, 0)

    def test_get_expected_columns(self):
        cols = get_expected_columns()
        self.assertIn(LABEL_COL, cols)
        for fc in FEATURE_COLS:
            self.assertIn(fc, cols)


if __name__ == "__main__":
    unittest.main()
