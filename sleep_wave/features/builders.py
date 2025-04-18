# sleep_wave/featues/builders.py

import pandas as pd

def feat_bandpower_base(df:pd.DataFrame):
    '''
    Baseline feature set using only raw bandpower feats (delta, theta, alpha, beta)
    Note: omitting cycle on purpose-- it's an engineered feature and was a boost when it was created!
    '''
    return df[['delta', 'theta', 'alpha', 'beta', 'label', 'subject_id']]

def add_temporal_context(df: pd.DataFrame, feature_cols: list, shifts=[-1, 1]): 
    '''
    Add temporal context features by shifting original feature cols within each subject group

    Params: 
        df (pd.DataFrame): df with 'subject_id'
        feature_cols (list): List of features to shift
        shifts (list): Epoch shifts to apply (ex. [-1,1] for t-1 and t+1) 

    Returns: 
        df (pd.DataFrame): df with added shifted feature columns 
    '''

    df = df.copy()
    
    for shift in shifts:
        shifted = (
            df.groupby('subject_id')[feature_cols]
            .shift(shift)
            .rename(columns=lambda col: f'{col}_t{shift:+}') # Shifted col names
        )
        df = pd.concat([df, shifted], axis=1)
    
    return df.dropna() # Drop rows with NaN values (due to shifts)

def feat_temporal_bandpower_t1(df):
    '''
    Add temporal context features for bandpower data
    '''
    df = df[[col for col in df.columns if not any(x in col for x in ["_t-1", "_t+1"])]]
    return add_temporal_context(df, ['delta', 'theta', 'alpha', 'beta'], shifts=[-1, 1])

# ==================================================
def add_lag_feats(df:pd.DataFrame, feature_cols, lags=[1,2]):
    '''
    Adds lagged versions of bandpower features for each subject using group-wise shifting

    Parameters: 
        df: input df containing raw EEG feats and 'subject_id'
        feature_cols (list): list of col names to lag (ex. ['delta', 'theta',...]) 
        lasgs (list): int list of lag steps to apply 
        
    Returns: 
        df: df with new lagged feats (rows with NaN removed) 
    '''
    df = df.copy()

    for lag in lags: 
        lagged = (
            df.groupby('subject_id')[feature_cols] 
            .shift(lag) 
            .rename(columns=lambda col: f'{col}_lag{lag}') 
        )
        df = pd.concat([df, lagged], axis=1) 
    
    df = df.dropna().reset_index(drop=True) # Drop rows with NaN values (due to shifts)

    return df

def feat_bandpower_lags(df:pd.DataFrame): 
    ''' 
    Generates lag-1 and lag-2 feats for core EEG bands 

    Parameters: 
        df: df with raw bandpowers

    Returns: 
        df: updated df with lagged feats
    '''
    df = df[[col for col in df.columns if not col.endswith(tuple(f"_lag{i}" for i in range(1, 4)))]]
    return add_lag_feats(df, ['delta', 'theta', 'alpha', 'beta'], lags=[1,2]) 
