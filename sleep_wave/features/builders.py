# sleep_wave/featues/builders.py

import pandas as pd
import numpy as np

# ==================================================
# Baseline model for comparison

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

def feat_band_ratios(df:pd.DataFrame):
    '''
    Adds key freq band ratio feats
    Ratios selected based on sleep research and greatest impact
    
    Ratios included:
        - alpha / theta (drowsiness, cognitive decline)
        - alpha / delta (mental fatigue, sleepiness)
        - beta / alpha (wakefulness, arousal)
        - theta / beta (ADHD, REM onset marker)
        - beta / delta (alertness vs. deep sleep)

    *** Uses np.log1p() to stabilize the scale.

    Parameters:
        df: df with raw bandpowers
    
    Returns: 
        df: updated with band ratio feats
    '''
    eps = 1e-15

    df = df.copy()
    df['alpha_theta'] = np.log1p(df['alpha'] / (df['theta'] + eps))
    df['alpha_delta'] = np.log1p(df['alpha'] / (df['delta'] + eps))
    df['beta_alpha'] = np.log1p(df['beta']  / (df['alpha'] + eps))
    df['theta_beta'] = np.log1p(df['theta'] / (df['beta'] + eps))
    df['beta_delta'] = np.log1p(df['beta']  / (df['delta'] + eps))

    return df

def feat_band_rollstd(df: pd.DataFrame): 
    '''
    Adds rolling standard deviation feats over the bands 
    Computes for windows of 3 and 5 epochs 
    Essentially another technique for localized smoothing
    '''

    df = df.copy()
    band_cols = ['delta', 'theta', 'alpha', 'beta']
    windows = [3,5]

    for band in band_cols:
        for win in windows:
            new_col = f'{band}_rollstd{win}'
            df[new_col] = df[band].rolling(window=win, min_periods=1, center=True).std()

    return df

# More advanced feat-- Entropy
# How unpredictable the signal is 
# High Entropy: lots of random variation (Wake or REM)
# Low Entropy: stable, repetitive patterns (N3 most likely)

# Shannon Entropy requires it's own func
def shannon_entropy(arr: np.ndarray): 
    '''
    Computes Shannon entropy of a normalized array
    '''
    # Removes 0 to avoid log(0)
    arr = arr[arr>0] 
    return -np.sum(arr * np.log(arr)) 

def feat_band_entropy(df: pd.DataFrame): 
    '''
    Adds Shannon entropy feat across bands 
    '''
    df = df.copy()
    band_cols = ['delta', 'theta', 'alpha', 'beta']
    band_data = df[band_cols].values

    # Normalize each row to sum to 1
    # This turns it into a prob distribution! 
    band_probs = band_data / np.sum(band_data, axis=1, keepdims=True)

    # Handle divide-by-zero rows by setting probs to uniform
    band_probs[np.isnan(band_probs)] = 1.0 / len(band_cols) 

    # Finally, apply entropy per row
    df['band_entropy'] = np.apply_along_axis(shannon_entropy, 1, band_probs)

    return df

def feat_time_context(df: pd.DataFrame):
    '''
    Adds a normalized time-of-night context feature.
    For each subject_id, computes a value from 0 (start of night) to 1 (end of night).
    '''
    df = df.copy()
    df['time_index'] = df.groupby('subject_id').cumcount()
    df['norm_time'] = df['time_index'] / df.groupby('subject_id')['time_index'].transform('max')
    return df


