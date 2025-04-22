# sleep_wave/featues/builders.py

import pandas as pd

# ==================================================
# Baseline model for comparison-- no `cycle` 

def feat_bandpower_base(df:pd.DataFrame):
    '''
    Baseline feature set using only raw bandpower feats (delta, theta, alpha, beta)
    Note: omitting cycle on purpose-- it's an engineered feature and was a boost when it was created!
    '''
    return df[['delta', 'theta', 'alpha', 'beta', 'label', 'subject_id']]

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

def feat_band_rollmean(df:pd.DataFrame, windows=[3,5]): 
    '''
    Adds rollingm mean feats for each band
    
    Parameters: 
        df: df with raw bandpowers
        windows (list): list of window sizes in epochs 
        
    Returns: 
        df: updated with rolling mean feats
    '''
    df = df.copy()
    bands = ['delta', 'theta', 'alpha', 'beta']

    for band in bands: 
        for w in windows: 
            col_name = f'{band}_rollmean_{w}'
            # center = True -- aligns with current epoch
            # min_periods = 1 -- NaN avoidance
            df[col_name] = df[band].rolling(window=w, center=True, min_periods=1).mean()
        
    return df
    
def feat_band_diff(df:pd.DataFrame):
    '''
    Adds first-order difference for each band
    Highlights changes in each band across epochs 
    Thus, gives info on the direction and intensity of the band shift 

    Parameters:
        df: df with raw bandpowers

    Returns: 
        df: updated df with diff feats 
    '''
    df = df.copy()
    bands = ['delta', 'theta', 'alpha', 'beta']

    for band in bands:
        if band in df.columns: 
            df[f'{band}_diff'] = df[band].diff().fillna(0) 
    
    return df
