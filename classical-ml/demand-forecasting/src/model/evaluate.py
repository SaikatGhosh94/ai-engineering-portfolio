from src.utils.config import LGBM_PARAMS,EVAL_REPORT_PATH
import json
from src.data.load_data import load_train_data
from datetime import datetime
from src.features.feature_engineering import create_features_for_global
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error, r2_score
from lightgbm import LGBMRegressor
def evaluate_model():

    train_df = load_train_data()

    train_df = create_features_for_global(train_df)
    split_point = int(len(train_df)*0.8)
    train_data = train_df.iloc[:split_point]
    validation_data = train_df[split_point:]

    model = LGBMRegressor(**LGBM_PARAMS)

    model.fit(train_data.drop(columns=['sales','date']),train_data['sales'])

    predictions = model.predict(validation_data.drop(columns=['sales','date']))

    mae = mean_absolute_error(validation_data['sales'], predictions)
    print(f"Mean absolute error : {mae}")
    rmse = root_mean_squared_error(validation_data['sales'],predictions)
    print(f"Root mean squared error : {rmse}")
    mape = mean_absolute_percentage_error(validation_data['sales'],predictions)
    print(f"Mean absolute percentage error : {mape}")
    r2 = r2_score(validation_data['sales'],predictions)
    print(f"R2 score : {r2}")



    report = { "model": "LightGBM Regressor",
              "hyperparameters": LGBM_PARAMS,
              "evaluation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
              "n_samples": int(len(validation_data)),
              "root mean squared error": float(rmse),
              "mean absolute error": float(mae),
              "mean absolute percentage error": float(mape),
              "r2_score": float(r2)}
    
    with open(EVAL_REPORT_PATH, "w") as f:
        json.dump(report,f,indent=4)

        print("===evaluation report+++")
        print(json.dumps(report, indent=4))

if __name__ == "__main__":
    evaluate_model()
