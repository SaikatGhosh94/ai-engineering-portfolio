import pandas as pd
import numpy as np

def create_time_features(df):
    df['dayOfWeek'] = df['date'].dt.dayofweek
    df['dayOfMonth'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['isWeekend'] = df['dayOfWeek'].isin([5,6]).astype(int)

    return df

def create_lag_features(df):
    df['lag_1'] = df.groupby(['store','item'])['sales'].shift(1)
    df['lag_7'] = df.groupby(['store','item'])['sales'].shift(7)
    df['lag_14'] = df.groupby(['store','item'])['sales'].shift(14)
    df['lag_30'] = df.groupby(['store','item'])['sales'].shift(30)
    df['lag_365'] = df.groupby(['store','item'])['sales'].shift(365)

    return df

def create_rolling_features(df):
    df['rolling_mean_7'] = df.groupby(['store','item'])['sales'].shift(1).rolling(7).mean()
    df['rolling_mean_14'] = df.groupby(['store','item'])['sales'].shift(1).rolling(14).mean()
    df['rolling_mean_30'] = df.groupby(['store','item'])['sales'].shift(1).rolling(30).mean()
    df['rolling_std_7'] = df.groupby(['store','item'])['sales'].shift(1).rolling(7).std()

    return df

def create_features_for_global(df):

    df = create_time_features(df)

    df = create_lag_features(df)

    df = create_rolling_features(df)

    return df


def create_features_for_time_series(df,store_id,item_id):

    df = df[(df['store'] ==store_id) & (df['item']== item_id)].copy()[['sales','date']]

    df.set_index('date', inplace=True)

    return df


    