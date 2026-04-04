"""
Preprocessing module: cleans and transforms raw diabetes data for ML.

Zero values in physiological columns (Glucose, BloodPressure, etc.) are
treated as missing and replaced with the column median.  All features are
then assembled into a single vector and standardised with StandardScaler.
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler

from src.config import FEATURE_COLS, LABEL_COL

# Columns where 0 is physiologically impossible and should be treated as null
_ZERO_AS_NULL_COLS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]


def replace_zeros_with_null(df: DataFrame) -> DataFrame:
    """Replace physiologically invalid zero values with null."""
    for col in _ZERO_AS_NULL_COLS:
        if col in df.columns:
            df = df.withColumn(
                col,
                F.when(F.col(col) == 0, None).otherwise(F.col(col)),
            )
    return df


def impute_nulls_with_median(df: DataFrame) -> DataFrame:
    """Replace null values with per-column median (approximated via percentile)."""
    for col in _ZERO_AS_NULL_COLS:
        if col not in df.columns:
            continue
        median_val = df.approxQuantile(col, [0.5], 0.001)[0]
        df = df.withColumn(
            col,
            F.when(F.col(col).isNull(), median_val).otherwise(F.col(col)),
        )
    return df


def build_preprocessing_pipeline() -> Pipeline:
    """Build a Spark ML pipeline for feature assembly and scaling.

    Returns:
        An unfitted Spark ML Pipeline with VectorAssembler + StandardScaler.
    """
    assembler = VectorAssembler(
        inputCols=FEATURE_COLS,
        outputCol="raw_features",
        handleInvalid="skip",
    )
    scaler = StandardScaler(
        inputCol="raw_features",
        outputCol="features",
        withStd=True,
        withMean=True,
    )
    return Pipeline(stages=[assembler, scaler])


def preprocess(df: DataFrame):
    """Run the full preprocessing workflow.

    Args:
        df: Raw Spark DataFrame loaded via the ingestion module.

    Returns:
        Tuple of (preprocessed_df, fitted_pipeline_model).
        preprocessed_df contains the original columns plus
        ``raw_features`` and ``features`` vector columns.
    """
    df = replace_zeros_with_null(df)
    df = impute_nulls_with_median(df)

    pipeline = build_preprocessing_pipeline()
    model = pipeline.fit(df)
    preprocessed_df = model.transform(df)
    return preprocessed_df, model
