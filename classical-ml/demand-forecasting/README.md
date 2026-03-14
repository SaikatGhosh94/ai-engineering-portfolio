# 📦 Demand Forecasting — Production Time-Series Pipeline

An end-to-end SKU-level demand forecasting pipeline: data ingestion, time-series feature engineering, model training, evaluation and reproducible artifacts.

---

## 🎯 Business Objective

Forecast SKU-level demand to improve inventory planning, reduce stockouts, and optimize replenishment.

---

## 🧾 Dataset

- Raw data (place here): data/raw/train.csv and data/raw/test.csv  
- Typical columns: date, store_id, sku, sales, price, promo, holiday_flag  
- Time index must be ISO date strings (YYYY-MM-DD)  
- As dataset could not be included into this repo, download the dataset from Kaggle and place train.csv and test.csv into data/raw/:
  - Kaggle dataset: https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data

---

## 🧠 Final Model

Production model chosen after experiments:

### ✅ LightGBM (gradient boosted trees)
- Use case: SKU-level regression with lag/rolling features and calendar flags  
- Target transform: train on log1p(sales); invert predictions with np.expm1()  
- Key hyperparameters (from reports/evaluation_report.json):
  - learning_rate: 0.1
  - max_depth: 15
  - n_estimators: 150
- Artifact: models/final_lightgbm_model.pkl (pipeline + metadata persisted)

Design note: LightGBM chosen for accuracy and inference speed; pipeline is group/time-aware to avoid leakage.

---

## 🔧 Feature Engineering

Implemented in src/features/feature_engineering.py and notebooks:
- Date features: day, week, month, weekday, is_month_end
- Lag features: lag_1, lag_7, lag_28
- Rolling statistics: roll_mean_7, roll_std_28
- External flags: promotions, holidays, price changes
- Expanding-window CV for validation

---

## 🧪 Evaluation & Artifacts

Saved evaluation artifacts:
- reports/evaluation_report.json — aggregated metrics (MAPE, RMSE, MAE)
- reports/model_predictions.csv — out-of-sample predictions per SKU

Recommended metrics: MAPE, RMSE, MAE (per-SKU + aggregate).

---

## 🏗 Project Structure
```text
demand-forecasting/
├── data/
│   └── raw/
│       ├── train.csv
│       └── test.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_gobal_modelling.ipynb
│   └── 04_time_series_modelling.ipynb
├── src/
│   ├── data/
│   │   └── load_data.py
│   ├── features/
│   │   └── feature_engineering.py
│   └── model/
│       ├── train.py
│       ├── evaluate.py
│       └── predict.py
├── models/                     # saved model artifacts (e.g., final_lightgbm_model.pkl)
├── reports/                    # evaluation outputs (evaluation_report.json, model_predictions.csv)
├── pyproject.toml
└── README.md
```

---

## 🚀 Quick Start

1. Install:
```bash
pip install -r requirements.txt
# or
pip install .
```

2. Place data:
- Put train.csv and test.csv in data/raw/ (or download from the Kaggle link above and place them there)

3. Train:
```bash
python -m src.model.train
```

4. Evaluate:
```bash
python -m src.model.evaluate
```

5. Predict:
```bash
python -m src.model.predict --input data/raw/test.csv --output reports/model_predictions.csv
```

---

## ✅ Tests & CI

- There are no dedicated test files in repo root; validate data and pipeline locally using notebooks.  
- If you add tests, run:
```bash
pytest -q
```

Reference data test file: data/raw/test.csv

---

## 📚 Notes & Future Work

- Add unit tests for feature builders and training loops.  
- Add model monitoring, drift detection, and automated retraining.  
- Containerize inference (FastAPI + Docker) for production deployment.

---

Author — Machine Learning Engineer
