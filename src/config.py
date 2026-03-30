"""
Configuration settings for the Diabetes Big Data Pipeline.
"""

import os

# MongoDB settings
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB = os.getenv("MONGO_DB", "diabetes_db")
MONGO_COLLECTION_PREPROCESSED = os.getenv("MONGO_COLLECTION_PREPROCESSED", "preprocessed_data")
MONGO_COLLECTION_METRICS = os.getenv("MONGO_COLLECTION_METRICS", "model_metrics")
MONGO_COLLECTION_PREDICTIONS = os.getenv("MONGO_COLLECTION_PREDICTIONS", "predictions")

# Spark settings
SPARK_APP_NAME = os.getenv("SPARK_APP_NAME", "DiabetesPipeline")
SPARK_MASTER = os.getenv("SPARK_MASTER", "local[*]")

# Data settings
DATA_PATH = os.getenv("DATA_PATH", "data/diabetes.csv")
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

# Feature columns (input features)
FEATURE_COLS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

# Label column
LABEL_COL = "Outcome"
