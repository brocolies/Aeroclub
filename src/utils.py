import pandas as pd
import numpy as np


def get_column_types(df, cat_threshold=30, date_threshold=0.3):
    cat_cols = []
    num_cols = []
    date_cols = []
    id_cols = []
    
    for col in df.columns:
        if col.lower().endswith('id'):
            id_cols.append(col)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            date_cols.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            if df[col].nunique() > cat_threshold:
                num_cols.append(col)
            else:
                cat_cols.append(col)
        elif pd.api.types.is_object_dtype(df[col]):
            temp_date = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
            # date로 변환 가능 -> 변환
            # date로 변환 불가능 -> errors='coerce'가 NaT로 변환 
            temp_date_ratio = temp_date.count() / len(temp_date)
            if temp_date_ratio >= date_threshold:
                date_cols.append(col)
            else:
                cat_cols.append(col) 
        elif pd.api.types.is_bool_dtype(df[col]):
            cat_cols.append(col)

    return cat_cols, num_cols, date_cols, id_cols

def extract_datetime_features(df, date_columns):
    new_cols = []
    for col in date_columns:
        if col in df.columns:
            dt = pd.to_datetime(df[col])
            df[f'{col}_hour'] = dt.dt.hour
            df[f'{col}_dayofweek'] = dt.dt.dayofweek
            df[f'{col}_month'] = dt.dt.month
            df[f'{col}_day'] = dt.dt.day
            new_cols.extend([f'{col}_hour', f'{col}_day', f'{col}_month'])
    return df, new_cols

def safe_optimize_dtypes(df):
    for col in df.select_dtypes(include=['int64']).columns:
        if df[col].isna().any():
            continue
        
        col_min, col_max = df[col].min(), df[col].max()
        
        # 음수가 있으면 signed 타입만 사용
        if col_min < 0:
            if col_min >= -128 and col_max <= 127:
                df[col] = df[col].astype('int8')
            elif col_min >= -32768 and col_max <= 32767:
                df[col] = df[col].astype('int16')
        else:
            # 양수만 있을 때만 unsigned 사용
            if col_max < 255:
                df[col] = df[col].astype('uint8')
            elif col_max < 65535:
                df[col] = df[col].astype('uint16')
    
    return df