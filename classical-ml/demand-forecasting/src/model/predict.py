import joblib
import pandas as pd
from src.data.load_data import load_test_data,load_train_data
from src.features.feature_engineering import create_features_for_global
from src.utils.config import MODEL_OUTPUT_PATH,MODEL_PREDICTION_PATH
def predict(model_path,model_prediction_path):
    
    test_df = load_test_data()

    train_df = load_train_data()

    test_dates = sorted(test_df['date'].unique())
    df = pd.concat([train_df, test_df], sort=False)
    model = joblib.load(model_path)

    for current_date in test_dates:
        
        df = create_features_for_global(df)
        date_mask = df['date'] == current_date
        pred = model.predict(df[date_mask].drop(columns=['sales','date','id']))
        df.loc[date_mask, 'sales'] = pred
    
    df[df['date']>=test_dates[0]][['date','store','item','sales']].to_csv(model_prediction_path,index=False)

if __name__ == "__main__":
    predict(MODEL_OUTPUT_PATH,MODEL_PREDICTION_PATH)

