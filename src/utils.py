import pandas as pd
import numpy as np
import lightgbm as lgb


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
            if col_max <= 255:
                df[col] = df[col].astype('uint8')
            elif col_max <= 65535:
                df[col] = df[col].astype('uint16')
    return df

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float32', 'float64']  # float16 제거
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min >= np.iinfo(np.int64).min and c_max <= np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print(f'Mem. usage decreased to {end_mem:5.2f} Mb ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    return df

def get_feature_importance(df, cat_cols, importance_type='gain', top_n=20):
    """
    전체 데이터로 LGBMRanker를 학습하고 feature importance를 반환
    
    Args:
        df: 데이터프레임 (selected, ranker_id 컬럼 포함)
        cat_cols: 카테고리 컬럼 리스트
        importance_type: 'gain', 'split', 'weight' 중 하나
        top_n: 상위 몇 개 feature를 출력할지
    
    Returns:
        feature_importance_df: feature importance DataFrame
        ranker: 학습된 모델
    """
    # 특성/타겟 분리
    X = df.drop(columns=['selected', 'ranker_id'])
    y = df['selected']
    
    # 카테고리 컬럼 타입 지정
    for col in cat_cols:
        if col in X.columns:
            X[col] = X[col].astype(str).astype('category')
    
    # 그룹 크기
    groups = df.groupby('ranker_id').size().values
    
    # 모델 학습
    ranker = lgb.LGBMRanker(
        objective='lambdarank',
        num_leaves=31,
        learning_rate=0.1,
        random_state=42,
        verbosity=-1,
        n_estimators=100
    )
    
    ranker.fit(X, y, group=groups, categorical_feature=cat_cols)
    
    # Feature importance 계산
    importance = ranker.booster_.feature_importance(importance_type=importance_type)
    
    feature_importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print(f"Top {top_n} features by {importance_type} importance:")
    print(feature_importance_df.head(top_n))
    
    return feature_importance_df, ranker
