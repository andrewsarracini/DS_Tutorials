# sleep_wave/featues/builders.py

import pandas as pd

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
