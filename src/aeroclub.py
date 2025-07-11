import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import make_column_transformer
from lightgbm import LGBMClassifier
from sklearn.pipeline import make_pipeline
from src.utils import *
from src.run import *


def duration_split(df, duration_cols):
    new_cols = []
    for col in duration_cols:
        time_split = df[col].str.split(':')
        hours_series = pd.to_numeric(time_split.str[0], errors='coerce')
        minutes_series = pd.to_numeric(time_split.str[1], errors='coerce')
        
        # Type: ignore to suppress linter warnings for pandas operations
        total_minutes = hours_series * 60 + minutes_series  # type: ignore
        df[f'{col}_total_minutes'] = total_minutes
        new_cols.append(f'{col}_total_minutes')
    return df, new_cols

def del_under_10_rows(df):
    valid_rankers = df['ranker_id'].value_counts()
    valid_rankers = valid_rankers[valid_rankers > 10].index
    # index: 조건을 만족하는 행들의 인덱스를 반환
    df = df[df['ranker_id'].isin(valid_rankers)]
    print('Lesser than 10 flight options/session deleted')

    return df

def add_duration_date_columns(df, date_cols):
    add_cols = []
    duration_cols = [col for col in date_cols if 'duration' in col]
    true_date_cols = [col for col in date_cols if 'duration' not in col]
    df, datetime_new_cols = extract_datetime_features(df, true_date_cols)
    df, duration_time_cols = duration_split(df, duration_cols)
    
    drop_cols = duration_cols + true_date_cols
    df = df.drop(columns=drop_cols)

    return df, datetime_new_cols, duration_time_cols

def del_columns(df):
    del_cols = ['Id']
    constant_cols = df.columns[df.nunique() == 1].tolist()
    del_cols = del_cols + constant_cols
    df = df.drop(columns=del_cols)
    return df, constant_cols



