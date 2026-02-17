# 🏠 House Price Prediction — Production-Ready ML Pipeline

An end-to-end machine learning system built to predict residential house prices using the **Ames Housing dataset**.

This project demonstrates not just model building, but proper **ML engineering practices**, including modular design, reproducible pipelines, evaluation reporting, and clean deployment-ready structure.

---

## 🎯 Business Objective

Accurately predict house prices based on structured tabular features such as:

- **Property characteristics:** Location (Neighborhood), Structural quality.
- **Area measurements:** Construction year, square footage, exterior details.

This predictive model assists in:
- Real estate price estimation
- Investment analysis
- Market valuation systems
- Automated appraisal tools

---

## 🧠 Final Model

After systematic experimentation with multiple regression models (Ridge, ElasticNet, RandomForest, Gradient Boosting), the best-performing model was:

### ✅ Support Vector Regression (SVR)
**Hyperparameters:**
- **Kernel:** `RBF`
- **C:** `0.5`
- **Gamma:** `auto`
- **Epsilon:** `0.01`

---

## 📈 Target Engineering

To stabilize variance and improve regression performance, the target variable is transformed:

### Transformation
```python
y = np.log1p(SalePrice)
```

###  Predictions are reversed using:
```python
np.expm1(predictions)
```

# 🔧 Feature Engineering
All transformations are implemented inside a Scikit-learn Pipeline to prevent data leakage and ensure reproducibility.
## Included Transformations
### Custom LotFrontage Imputation
- Missing values filled using mean LotFrontage per Neighborhood.
- Fallback to global median.
### Numerical Features
- Median imputation.
- Standard scaling (required for SVR performance).
### Categorical Features
- Most frequent imputation.
- One-hot encoding with safe handling of unseen categories.


# 🏗 Project Architecture
```text
house-prices-ml/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── final_model.pkl
│   └── evaluation_report.json
├── src/
│   ├── config.py
│   ├── features/
│   │   ├── custom_transformers.py
│   │   └── build_features.py
│   └── models/
│       ├── pipeline.py
│       ├── train.py
│       ├── evaluate.py
│       └── predict.py
└── notebooks/
```

# Design Principles
- Separation of Concerns: Logic is strictly split between feature engineering and modeling.
- Centralized Configuration: All paths and hyperparameters are managed in src/config.py.
- Modular Feature Pipeline: Reusable transformers ensure consistency across environments.
- CLI-based Execution: Operations are triggered via terminal commands for automation.
- Saved Evaluation Metadata: Performance snapshots are stored in models/evaluation_report.json.
- Fully Reproducible Training: End-to-end scripts ensure consistent results.
  
# 📊 Evaluation Strategy
Model performance is evaluated using:
- 5-Fold Cross Validation
- Metrics: RMSE, R² Score, Training RMSE.
  
# 🚀 Execution

### 1. Evaluate Model
```python
python -m src.models.evaluate
```

### 2. Train Final Model
```python
python -m src.models.train
```

### 3. Generate Predictions
```python
python -m src.models.predict
```

# 🧪 Development Workflow
- Phase 1 — Experimentation (Notebooks): Model comparison, hyperparameter tuning, and feature validation.
- Phase 2 — Production Code: Freezing the best model, moving to modular .py files, and generating final artifacts.

# 💡 Key Learnings & Future Improvements
## Key Learnings
- Designing scalable ML project structures.
- Writing custom sklearn transformers.
- Avoiding data leakage through pipelines.
## Future Improvements
- Tracking: MLflow experiment tracking.
- Deployment: REST API via FastAPI and Docker containerization.
- Automation: CI/CD integration for automated retraining.


# 📚 Dataset & Author
- Dataset: Ames Housing Dataset (Kaggle)
- Author: Saikat Ghosh — Machine Learning Engineer