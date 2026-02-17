from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
class ColumnDropper(BaseEstimator,TransformerMixin):
    def __init__(self,columns):
        self.columns = columns
    def get_feature_names_out(self, input_features=None):
        """Allows set_output(transform='pandas') to work."""
        return input_features
    def fit(self,X , y = None):
        return self
    def transform(self,X):
        return X.drop(columns=self.columns,axis=1, errors = 'ignore')

class ColumnBasedMedianImputer(BaseEstimator,TransformerMixin):
    def __init__(self,fill_column,group_by_columns):
        self.fill_column = fill_column
        self.group_by_columns = group_by_columns
    
    def fit(self,X,y=None):
        X = pd.DataFrame(X).copy()

        self.mean_fill_column_by_group = X.groupby(self.group_by_columns)[self.fill_column].mean()


        self.global_mean = X[self.fill_column].mean()
        return self

    def get_feature_names_out(self, input_features=None):
        """Allows set_output(transform='pandas') to work."""
        return input_features
    
    def transform(self,X):
        X = pd.DataFrame(X).copy()

        # def impute(row):
        #     if pd.isna(row[self.fill_column]):
        #         return self.mean_fill_column_by_group.get(row[self.group_by_columns],self.global_mean)
        #     return row[self.fill_column]
        
        # X[self.fill_column] = X.apply(impute, axis=1)
        X[self.fill_column] = X[self.fill_column].fillna(X[self.group_by_columns].map(self.mean_fill_column_by_group))
        X[self.fill_column] = X[self.fill_column].fillna(self.global_mean)
        return X