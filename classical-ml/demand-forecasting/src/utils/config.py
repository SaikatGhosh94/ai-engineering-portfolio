TRAIN_DATA_PATH="data/raw/train.csv"
TEST_DATA_PATH="data/raw/test.csv"

MODEL_OUTPUT_PATH = "models/lightgbm_model.pkl"

LGBM_PARAMS = {'learning_rate': 0.1, 'max_depth': 15, 'n_estimators': 150}

EVAL_REPORT_PATH = "reports/evaluation_report.json"

MODEL_PREDICTION_PATH = "reports/model_predictions.csv"