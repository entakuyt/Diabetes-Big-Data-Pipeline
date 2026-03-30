"""
Data ingestion module: loads the diabetes CSV dataset into a Spark DataFrame.
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    DoubleType,
)

from src.config import FEATURE_COLS, LABEL_COL

# Explicit schema to ensure correct types are inferred
_SCHEMA = StructType(
    [
        StructField("Pregnancies", IntegerType(), True),
        StructField("Glucose", IntegerType(), True),
        StructField("BloodPressure", IntegerType(), True),
        StructField("SkinThickness", IntegerType(), True),
        StructField("Insulin", IntegerType(), True),
        StructField("BMI", DoubleType(), True),
        StructField("DiabetesPedigreeFunction", DoubleType(), True),
        StructField("Age", IntegerType(), True),
        StructField("Outcome", IntegerType(), True),
    ]
)


def load_data(spark: SparkSession, path: str) -> DataFrame:
    """Load the diabetes CSV dataset into a Spark DataFrame.

    Args:
        spark: Active SparkSession.
        path: Path to the CSV file (local or HDFS).

    Returns:
        Spark DataFrame with the raw diabetes data.
    """
    df = (
        spark.read.format("csv")
        .option("header", "true")
        .schema(_SCHEMA)
        .load(path)
    )
    return df


def get_expected_columns() -> list:
    """Return the list of expected columns (features + label)."""
    return FEATURE_COLS + [LABEL_COL]
