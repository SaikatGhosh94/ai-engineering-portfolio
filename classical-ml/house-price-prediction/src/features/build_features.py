import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.utils.config import FILL_NA_COLS,FILL_NONE_COLS,FILL_ZERO_COLS,FILL_MODE_COLS


def build_preprocessor():
    fill_zero_pipeline = Pipeline([('imputer',SimpleImputer(strategy='constant',fill_value = 0))])

    fill_most_frequent_pipeline = Pipeline([('imputer',SimpleImputer(strategy='most_frequent'))])

    fill_NA_pipeline = Pipeline([('imputer',SimpleImputer(strategy='constant',fill_value='NA'))])

    fill_None_pipeline = Pipeline([('imputer',SimpleImputer(strategy='constant',fill_value='None'))])

    fill_mode_pipeline = Pipeline([('imputer',SimpleImputer(strategy='most_frequent'))])

    preprocessor = ColumnTransformer(transformers=[
        ('fill_na',fill_NA_pipeline,FILL_NA_COLS),
        ('fill_zero',fill_zero_pipeline,FILL_ZERO_COLS),
        ('fill_none',fill_None_pipeline,FILL_NONE_COLS),
        ('fill_mode',fill_mode_pipeline,FILL_MODE_COLS)
    ],remainder = 'passthrough',verbose_feature_names_out=False)

    preprocessor.set_output(transform='pandas')

    return preprocessor
    