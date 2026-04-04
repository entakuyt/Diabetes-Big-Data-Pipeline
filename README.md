# Diabetes Big Data Pipeline

A production-style big-data pipeline that processes the **Pima Indians Diabetes Dataset** using **Apache Spark** for distributed computation and **MongoDB** for persistence.

---

## Architecture

```
CSV Dataset
    │
    ▼
Data Ingestion  (src/ingestion.py)
    │  Loads CSV with an explicit schema via PySpark
    ▼
Preprocessing   (src/preprocessing.py)
    │  • Replaces physiologically invalid zeros with null
    │  • Imputes nulls with column median
    │  • Assembles features into a vector
    │  • Standardises with StandardScaler (zero mean, unit variance)
    ▼
Model Training  (src/model.py)
    │  • Logistic Regression (Spark MLlib)
    │  • Random Forest       (Spark MLlib)
    ▼
Evaluation      (src/model.py)
    │  AUC-ROC, AUC-PR, Accuracy, F1, Precision, Recall
    ▼
MongoDB Storage (src/storage.py)
    │  • Preprocessed sample  → diabetes_db.preprocessed_data
    │  • Model metrics        → diabetes_db.model_metrics
    │  • Predictions          → diabetes_db.predictions
```

---

## Project Structure

```
Diabetes-Big-Data-Pipeline/
├── data/
│   └── diabetes.csv            # Pima Indians Diabetes Dataset
├── src/
│   ├── config.py               # Central configuration (env-overridable)
│   ├── ingestion.py            # Load CSV into Spark DataFrame
│   ├── preprocessing.py        # Cleaning, imputation, feature engineering
│   ├── model.py                # ML model training & evaluation
│   ├── storage.py              # MongoDB read/write helpers
│   └── pipeline.py             # End-to-end orchestrator (entry point)
├── tests/
│   ├── test_ingestion.py
│   ├── test_preprocessing.py
│   └── test_model.py
├── docker-compose.yml          # Spark + MongoDB cluster
└── requirements.txt
```

---

## Quick Start

### 1. Local (no Docker)

**Prerequisites:** Java 11+, Python 3.9+

```bash
# Install dependencies
pip install -r requirements.txt

# Run the pipeline against the local CSV
python -m src.pipeline data/diabetes.csv
```

The pipeline will train both models and print evaluation metrics.
MongoDB persistence is attempted automatically; if MongoDB is not running
the pipeline continues and logs a warning.

### 2. Docker Compose (full cluster)

```bash
docker compose up --build
```

This starts:
| Service | Port | Description |
|---|---|---|
| `mongodb` | 27017 | MongoDB 7.0 |
| `spark-master` | 8080, 7077 | Spark standalone master |
| `spark-worker` | – | Spark worker (2 cores / 2 GB) |
| `pipeline` | – | Runs the pipeline and exits |

---

## Configuration

All settings can be overridden via environment variables:

| Variable | Default | Description |
|---|---|---|
| `MONGO_URI` | `mongodb://localhost:27017/` | MongoDB connection URI |
| `MONGO_DB` | `diabetes_db` | Database name |
| `SPARK_MASTER` | `local[*]` | Spark master URL |
| `DATA_PATH` | `data/diabetes.csv` | Path to the dataset |
| `TEST_SIZE` | `0.2` | Fraction of data for testing |
| `RANDOM_SEED` | `42` | Reproducibility seed |

---

## Dataset

The **Pima Indians Diabetes Dataset** contains 768 records with 8 features
and a binary outcome (1 = diabetic, 0 = non-diabetic):

| Column | Description |
|---|---|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration (mg/dL) |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-Hour serum insulin (mu U/ml) |
| BMI | Body mass index |
| DiabetesPedigreeFunction | Diabetes pedigree function score |
| Age | Age in years |
| Outcome | 1 = diabetic, 0 = non-diabetic |

---

## Running Tests

```bash
python -m pytest tests/ -v
```

Expected output: **15 tests pass**.

---

## Example Results

```
=== Pipeline Results ===

LogisticRegression:
  model: LogisticRegression
  auc_roc: 0.8412
  auc_pr:  0.7631
  accuracy: 0.7662
  f1_score: 0.7589
  precision: 0.7631
  recall: 0.7662

RandomForest:
  model: RandomForest
  auc_roc: 0.8501
  auc_pr:  0.7703
  accuracy: 0.7792
  f1_score: 0.7741
  precision: 0.7782
  recall: 0.7792
```
