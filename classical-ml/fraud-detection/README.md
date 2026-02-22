# 🔍 Fraud Detection — Classical ML Pipeline

An end-to-end fraud detection project using the public credit card transactions dataset. Demonstrates classical ML approaches, reproducible preprocessing, feature engineering, modeling and evaluation reporting.

---

## 🎯 Business Objective

Detect fraudulent credit-card transactions to minimize financial loss and reduce false alerts. Useful for:
- Transaction monitoring systems
- Fraud alerting pipelines
- Risk scoring and investigations

---

## 📊 Data Source
This project uses the [Credit Card Fraud Detection Dataset](https://www.kaggle.com) from Kaggle.
To run this project:
1. Download `creditcard.csv` from the link above.
2. Place it in the `data/raw/` directory.

---

## 🧾 Dataset

- Source file: data/raw/creditcard.csv  
- Highly imbalanced binary classification: Fraud (1) vs Non-fraud (0)

---

## 🧠 Final Model

Following experimentation and baseline modelling, the reported model is:

### ✅ Logistic Regression (baseline / production-ready)
- Regularization: L2 (default)
- Solver: liblinear / sag (depending on implementation)
- Notes: class imbalance handled via resampling / class weights in training notebooks

(See detailed results in reports/logistic_regression_classification_report.txt)

---

## 🔧 Feature Engineering & Preprocessing

Key processing steps implemented in the pipeline and notebooks:
- Scaling of numerical features (StandardScaler)
- Handling class imbalance (class weights, resampling strategies explored)
- Feature selection based on EDA and correlation analysis
- All transformations wrapped in reproducible scripts / pipelines to avoid leakage

---

## 🧪 Evaluation

Metrics and artifacts:
- Precision, Recall, F1-score, ROC AUC — see reports/logistic_regression_classification_report.txt  
- Confusion matrix visual (referenced): reports/confusion_matrix.png

---

## 🏗 Project Structure
```text
fraud-detection/
├── data/
│   └── raw/
│       └── creditcard.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modelling.ipynb
├── src/
│   ├── data/
│   │   └── load_data.py
│   ├── features/
│   │   └── build_features.py
│   └── model/
│       └── train.py
├── models/
│   └── final_model.pkl
├── reports/
│   ├── logistic_regression_classification_report.txt
│   ├── logistic_regression_confusion_matrix.png
│   └── logistic_regression_roc.png
└── pyproject.toml
```

---

## 🚀 Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
# or
pip install .
```

2. Train model:
```bash
python -m src.model.train
```

3. Inspect evaluation:
- Classification report: reports/logistic_regression_classification_report.txt  
- Confusion matrix image: reports/confusion_matrix.png

---

## 🧭 Development Workflow

- Notebooks for exploration and experiments.
- Production scripts under src/ for reproducible runs.
- Save trained artifacts to models/ and evaluation snapshots to reports/.

---

## 🔁 Future Improvements

- Add advanced imbalance handling (SMOTE/ADASYN) and tuned tree-based models.
- Model monitoring and alerting pipeline.
- Containerized deployment and inference API.

---

## 📚 License & Author

- Dataset: Credit Card Fraud Detection (public dataset)  
- Author: (project owner) — Machine Learning Engineer
