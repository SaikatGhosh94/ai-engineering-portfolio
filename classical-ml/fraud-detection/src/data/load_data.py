import pandas as pd

from src.utils.config import SOURCE_DATA_PATH

def load():
    df = pd.read_csv(SOURCE_DATA_PATH)
    X = df.drop("Class",axis=1)
    y = df["Class"]
    return X,y
