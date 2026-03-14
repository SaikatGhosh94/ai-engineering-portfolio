import pandas as pd

from src.utils.config import TRAIN_DATA_PATH,TEST_DATA_PATH

def load_train_data():
    return pd.read_csv(TRAIN_DATA_PATH,parse_dates=['date'])

def load_test_data():
    return pd.read_csv(TEST_DATA_PATH,parse_dates=['date'])
