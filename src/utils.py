import pandas as pd

def resumetable(df, target_col, missing_value=-1, ignore_cols=None, verbose=True):
    ignore_cols = ignore_cols or []
    if verbose:
        print(f'Data shape: {df.shape}')

    summary = pd.DataFrame(df.dtypes, columns=['Data Type'])
    summary['Missing'] = (df == missing_value).sum().values
    summary['Nunique'] = df.nunique().values
    summary['Feature Type'] = None

    for col in df.columns:
        if col in ignore_cols:
            continue
        if col == target_col:
            summary.loc[col, 'Feature Type'] = 'Target'
            continue 
        if df[col].nunique() == len(df):
            summary.loc[col, 'Feature Type'] = 'Id'
            continue
        if df[col].nunique() == 2:
            summary.loc[col, 'Feature Type'] = 'Binary'
            continue
        if np.issubdtype(df[col].dtype, 'object'):
            summary.loc[col, 'Feature Type'] = 'Categorical'
            continue
        if np.issubdtype(df[col].dtype, np.number):
            summary.loc[col, 'Feature Type'] = 'Needs_Review(Int)'
        
    summary = summary.sort_values(by='Feature Type')
    return summary
