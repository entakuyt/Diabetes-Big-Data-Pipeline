
# Diabetes Big Data Pipeline: Spark + MongoDB

<div align="center">

![Spark](https://img.shields.io/badge/Apache_Spark-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-47A248?style=for-the-badge&logo=mongodb&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

**End-to-End Big Data Pipeline for Diabetes Healthcare Analysis**

[Quick-Start](#-quick-start) · [Architecture](#-architecture) · [Features](#-core-features) · [Insights](#-key-insights)

</div>

---

## 🎯 Project Overview

The **Diabetes Big Data Pipeline** is a complete big data solution that ingests, stores, processes, and analyzes a large-scale diabetes healthcare dataset (~100,000 records) using **Apache Spark** and **MongoDB**.

This pipeline demonstrates an efficient workflow for handling healthcare data — from ingestion to insightful analysis — enabling better identification of high-risk diabetes patients based on key medical and demographic factors.

### ✨ Key Objectives
- Efficiently store large healthcare datasets in MongoDB
- Perform distributed data processing with Apache Spark
- Analyze diabetes risk factors using Spark SQL
- Generate clear visualizations and actionable healthcare insights

---

## 🏗️ Architecture


                                    ╔═══════════════╗
                                    ║  CSV Dataset  ║
                                    ║  (Raw Input)  ║
                                    ╚───────┬───────╝
                                            │
                                            ▼
                                    ╔═══════════╗
                                    ║  Pandas   ║
                                    ║ (Python)  ║
                                    ╚─────┬─────╝
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    │                     │                     │
                    ▼                     ▼                     ▼
            ╔════════════╗         ╔════════════╗        ╔══════════════╗
            ║  MongoDB   ║         ║  PyMongo   ║        ║   Spark      ║
            ║ (Storage)  ║ ◄────── ║ Extraction║ ──────► ║ DataFrame    ║
            ╚════════════╝         ╚════════════╝        ║ (Distributed)║
                                                         ╚──────┬───────╝
                                                                │
                                                                ▼
                                                    ╔═══════════════════╗
                                                    ║  Spark SQL +      ║
                                                    ║  Transformations  ║
                                                    ║  + Aggregations   ║
                                                    ╚─────────┬─────────╝
                                                              │
                                                              ▼
                                                    ╔═══════════════════╗
                                                    ║  Visualizations   ║
                                                    ║ (Matplotlib +     ║
                                                    ║    Seaborn)       ║
                                                    ╚═══════════════════╝



---

## 🚀 Core Features

### 1. Data Ingestion
- Load CSV dataset using Pandas
- Batch insertion into MongoDB with duplicate prevention

### 2. MongoDB Integration
- Flexible document-based schema for patient records
- Recommended indexing on key fields (`diabetes`, `location`, etc.)
- Sharding strategy for future scalability

### 3. Spark Processing
- Read data from MongoDB into Spark DataFrame
- Data cleaning and feature engineering (age groups, high-risk flags)
- Distributed transformations using PySpark

### 4. Advanced Analysis
- Spark SQL queries for diabetes percentage by location
- Aggregations and ranking of high-risk areas
- Correlation analysis between medical features

### 5. Visualizations & Insights
- Top 10 states by diabetes prevalence (bar chart)
- Risk factor exploration through visualizations
- Clean and publication-ready charts

---

## 📦 Project Structure

```text
Diabetes-Big-Data-Pipeline/
├── Project_Fixed.ipynb         # Main Jupyter Notebook (complete pipeline)
├── README.md                   # Project documentation (overview, setup, usage)
├── data/                       # Raw datasets
│   └── diabetes_dataset.csv    # Original input dataset
├── docs/                       # Project reports and supporting documents
│   ├── ProjectProposal_Group11.pdf
│   └── project_report.pdf
├── clean_diabetes_data/        # Cleaned dataset (Parquet, optimized for fast queries, not committed)
└── venv/                       # Virtual environment (not committed)
```

---

## 🚀 Quick-Start

### Prerequisites
- Python 3.10 or higher
- MongoDB running locally on `localhost:27017`
- Jupyter Notebook / JupyterLab

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/diabetes-big-data-pipeline.git
cd diabetes-big-data-pipeline
```

### 2. Create and Activate Virtual Environment

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS & Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install pyspark pymongo pandas matplotlib seaborn
```

### 4. Start MongoDB

**Windows**
```bash
net start MongoDB
```

**macOS**
```bash
brew services start mongodb-community
```

**Linux**
```bash
sudo systemctl start mongod
```

### 5. Run the Pipeline
```bash
jupyter notebook
```
Open **`Project_Fixed.ipynb`** and run all cells sequentially.

---

## 📊 Key Insights

- Higher BMI, HbA1c level, and blood glucose are strongly associated with diabetes
- Certain geographic locations show significantly higher diabetes prevalence
- Patients with hypertension and older age groups are at elevated risk
- Proper MongoDB indexing significantly improves query performance

---

## 🛠️ Technologies Used

### Core Technologies
- **Apache Spark 4.1.1** – Distributed big data processing
- **MongoDB** – Flexible NoSQL database
- **PyMongo** – MongoDB Python driver
- **PySpark** – Python interface for Spark

### Supporting Tools
- **Pandas** – Data manipulation
- **Matplotlib & Seaborn** – Data visualization
- **Spark SQL** – Declarative querying

---

## 📈 Performance Highlights

- Successfully processed **100,000 patient records**
- Efficient batch ingestion into MongoDB
- Fast distributed transformations with Spark
- Optimized queries through proper indexing

---

## 👥 Team Members

- Tuo Yan
- Prayusha Poudel
- Dan Le

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Apache Spark and MongoDB open-source communities
- Kaggle for providing the Diabetes Healthcare Dataset
- All libraries and tools that made this pipeline possible

---

<div align="center">

**Built with ❤️ using Apache Spark, MongoDB, and Python**

![Spark](https://img.shields.io/badge/Apache_Spark-E25A1C?style=flat&logo=apachespark&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-47A248?style=flat&logo=mongodb&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)

</div>


