import pandas as pd
import numpy as np


def get_column_types(df, cat_threshold=30, date_threshold=0.7):
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
            temp_date = pd.to_datetime(df[col], errors='coerce')
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

            