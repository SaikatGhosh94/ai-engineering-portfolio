import pandas as pd
import joblib
import json
from datetime import datetime
import numpy as np
from sklearn.model_selection import cross_val_score
from src.utils.config import TRAIN_CSV_PATH,MODEL_OUTPUT_PATH,SVR_PARAMS,EVAL_REPORT_PATH
from sklearn.metrics import mean_absolute_error, r2_score

def evaluate():

    train_df = pd.read_csv(TRAIN_CSV_PATH)

    X = train_df.drop('SalePrice',axis=1)

    n_samples = X.shape[0]

    y = np.log1p(train_df['SalePrice'])

    pipeline = joblib.load(MODEL_OUTPUT_PATH)

    cv_score = cross_val_score(pipeline,X,y,scoring='neg_root_mean_squared_error',cv=5)

    cv_rmse_mean = -cv_score.mean()

    pipeline.fit(X,y)

    pred = pipeline.predict(X)

    train_rmse = mean_absolute_error(pred,y)

    r2 = r2_score(y,pred)

    residuals = y - pred

    report = { "model": "Support Vector Regression",
              "hyperparameters": SVR_PARAMS,
              "evaluation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
              "n_samples": int(n_samples),
              "cv_rmse_mean": float(cv_rmse_mean),
              "train_rmse": float(train_rmse),
              "r2_score": float(r2),
              "mean_residual": float(residuals.mean())}
    
    with open(EVAL_REPORT_PATH, "w") as f:
        json.dump(report,f,indent=4)

        print("===evaluation report+++")
        print(json.dumps(report, indent=4))

if __name__ == "__main__":
    evaluate()