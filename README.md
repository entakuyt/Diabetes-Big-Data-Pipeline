# Big Data Pipeline for Diabetes Analysis

**Apache Spark + MongoDB**

## 📌 Overview

This project implements a **Big Data pipeline** to analyze a diabetes healthcare dataset using:

* **Apache Spark (PySpark)** for large-scale data processing
* **MongoDB** for flexible data storage
* **Python (Pandas, Matplotlib, Seaborn)** for visualization

The dataset includes patient information such as age, gender, BMI, blood glucose level, and diabetes status. 

[Diabetes Dataset (Kaggle)](https://www.kaggle.com/datasets/priyamchoksi/100000-diabetes-clinical-dataset)
---

## 🎯 Objectives

* Store healthcare data efficiently using MongoDB
* Process and analyse data using Apache Spark
* Perform queries using Spark SQL
* Improve performance using indexing
* Visualize key insights from the data 

---

## 📊 Dataset

* Source: Kaggle (Diabetes dataset)
* Size: ~100,000 records
* Key features:

  * Age
  * Gender
  * BMI
  * Blood glucose level
  * Smoking history
  * Diabetes (0/1) 

---

## 🏗️ System Architecture

```text
CSV Dataset → Pandas → MongoDB → Spark → Analysis → Visualization
```

---

## 🔄 Methodology

### 1. Data Ingestion

* Load CSV dataset using Pandas
* Convert to dictionary format
* Insert into MongoDB

### 2. Spark Processing

* Read data from MongoDB into Spark
* Clean and preprocess data:

  * Remove null values
  * Convert data types
  * Filter valid records

### 3. Data Analysis

Using Spark SQL:

* Count patients by location
* Calculate average BMI
* Compute diabetes percentage
* Rank patients by glucose level

### 4. Performance Optimization

* Created index on `location` field in MongoDB
* Compared query performance:

  * Without index → slower
  * With index → faster 

### 5. Visualization

* Age distribution vs diabetes
* Correlation heatmap
* BMI comparison

---

## 📈 Key Results

* Higher BMI is associated with higher diabetes risk
* Some locations show higher diabetes rates
* Blood glucose level is a strong indicator of diabetes
* MongoDB indexing significantly improves query performance 

---

## ⚙️ Technologies Used

* Python
* PySpark
* MongoDB
* Pandas
* Matplotlib
* Seaborn

---

## 🚀 How to Run
## Virtual Environment Setup (Recommended)

It is recommended to use a virtual environment to manage dependencies.

### 🪟 Windows

Create a virtual environment:
```bash
python -m venv venv
```

1. Install dependencies:

```bash
pip install pyspark pymongo pandas matplotlib seaborn
```
Activate it:

```bash
venv\Scripts\activate
```

### 🍎 macOS / 🐧 Linux

Create a virtual environment:

```bash
python3 -m venv venv
```
Activate it:

```bash
source venv/bin/activate
```
2. Start MongoDB

### 🪟 Windows

If MongoDB is installed as a service:
```bash
net start MongoDB
```
Or run manually:

```bash
"C:\\Program Files\\MongoDB\\Server\\<version>\\bin\\mongod.exe"
```

### 🍎 macOS

If installed with Homebrew:

```bash
brew services start mongodb-community
```

Or run manually:

```bash
mongod
```

3. Open and run:

```bash
Project_Fixed.ipynb
```

## 👥 Team Members

* Prayusha Poudel
* Tuo Yan
* Dan Le

---

## 📜 License

This project is for academic purposes.
