import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import make_column_transformer
from lightgbm import LGBMClassifier
from sklearn.pipeline import make_pipeline

def calculate_hit_rate_at_k(y_pred_proba, y_true, ranker_ids, k=3):
    df = pd.DataFrame({
        'pred': y_pred_proba,
        'true': y_true,
        'ranker_id': ranker_ids
    })
    
    hits = 0
    valid_sessions = 0
    
    for session_id, group in df.groupby('ranker_id'):
            
        ranked = group.sort_values('pred', ascending=False)
        top_k_true = ranked['true'].head(k).sum()
        
        if top_k_true > 0:
            hits += 1
        valid_sessions += 1
    
    return hits / valid_sessions if valid_sessions > 0 else 0


def run_baseline_cv(train, cat_cols, negative_ratio=5, n_splits=5):
    """베이스라인 CV 실행"""
    # Negative sampling
    # train = train.drop(columns=date_cols)
    positive_samples = train[train['selected'] == 1]
    negative_samples = train[train['selected'] == 0]
    negative_sampled = negative_samples.sample(
        n=len(positive_samples) * negative_ratio, 
        random_state=42
    )
    train_balanced = pd.concat([positive_samples, negative_sampled])
    train_balanced = train_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 전처리
    train_df = train_balanced.copy()
    X = train_df.drop(columns='selected')
    y = train_df['selected']
    
    # 인코딩
    for col in cat_cols:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    
    # CV
    model = LGBMClassifier(random_state=42)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        X_train = X_train.drop(columns='ranker_id')
        X_val = X_val.drop(columns='ranker_id')
        
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        ranker_ids_val = train_df['ranker_id'].iloc[val_idx]
        score = calculate_hit_rate_at_k(y_pred_proba, y_val, ranker_ids_val)    
        cv_scores.append(score)
        print(f'Fold {fold+1}: {score: .5f}')
    
    mean_score = np.mean(cv_scores)
    std_score = np.std(cv_scores)
    print(f'Score: {mean_score: .5f}, Std: {std_score: .5f}')
    
    return mean_score, std_score, cv_scores

def duration_split(df, duration_cols):
    new_cols = []
    for col in duration_cols:
        time_split = df[col].str.split(':')
        hours = pd.to_numeric(time_split.str[0], errors='coerce').fillna(0).astype(int)
        minutes = pd.to_numeric(time_split.str[1], errors='coerce').fillna(0).astype(int)
        
        df[f'{col}_total_minutes'] = hours * 60 + minutes
        new_cols.append(f'{col}_total_minutes')
    return df, new_cols