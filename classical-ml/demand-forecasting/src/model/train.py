from src.data.load_data import load_train_data
from src.features.feature_engineering import create_features_for_global
from src.utils.config import MODEL_OUTPUT_PATH,LGBM_PARAMS
from lightgbm import LGBMRegressor
import joblib
def train(model_output_path):

    train_df = load_train_data()
    train_df = create_features_for_global(train_df)
    model = LGBMRegressor(**LGBM_PARAMS)
    model.fit(train_df.drop(columns=['sales','date']),train_df['sales'])

    joblib.dump(model,model_output_path)



if __name__ == "__main__":
    train(MODEL_OUTPUT_PATH)


    