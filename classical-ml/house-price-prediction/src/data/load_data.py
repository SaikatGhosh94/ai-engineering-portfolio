import pandas as pd
from src.utils.config import TRAIN_CSV_PATH,TEST_CSV_PATH

def load_train_data():
    df = pd.read_csv(TRAIN_CSV_PATH)
    X = df.drop("SalePrice",axis=1)
    y = df["SalePrice"]
    return X,y

def load_test_data():
    return pd.read_csv(TEST_CSV_PATH)