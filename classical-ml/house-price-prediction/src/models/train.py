import pandas as pd
import numpy as np
import joblib
from src.features.build_features import build_preprocessor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.features.custom_transformer import ColumnDropper, ColumnBasedMedianImputer
from sklearn.svm import SVR
from src.utils.config import DROP_COLS,TRAIN_CSV_PATH,MODEL_OUTPUT_PATH,SVR_PARAMS
def build_pipeline(numerical_cols,category_cols):

    preprocessor = build_preprocessor()

    scaler_encoder = ColumnTransformer([('scaler',StandardScaler(),numerical_cols),
                                    ('encoder',OneHotEncoder(handle_unknown='ignore',sparse_output=False,),category_cols)])

    pipeline = Pipeline([
                     ('drop_column',ColumnDropper(DROP_COLS)),
                     ('fill_LotFrontage',ColumnBasedMedianImputer(fill_column='LotFrontage',group_by_columns='Neighborhood')),
                     ('preprocessor',preprocessor),
                     ('scale_and_encode',scaler_encoder),
                     ('model', SVR(**SVR_PARAMS))])
    
    return pipeline

def train_model(train_path,model_output_path):
    train_df = pd.read_csv(train_path)

    # create new model pipeline
    X = train_df.drop('SalePrice',axis=1)

    ##as the y label is right skewed , we will log transform as regression expects normal distributed data
    y = np.log1p(train_df['SalePrice'])

    numerical_cols = X.select_dtypes(include=['int64','float64']).columns
    categorical_cols = X.select_dtypes(include='object').columns

    updated_num_cols = [c for c in numerical_cols if c not in DROP_COLS]
    updated_cat_cols = [c for c in categorical_cols if c not in DROP_COLS]

    pipeline = build_pipeline(updated_num_cols,updated_cat_cols)

    pipeline.fit(X,y)

    joblib.dump(pipeline,model_output_path)

    print("Model trained and saved successfully")

if __name__ == "__main__":
    train_model(TRAIN_CSV_PATH,MODEL_OUTPUT_PATH)