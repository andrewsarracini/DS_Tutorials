import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def detect_na(df: pd.DataFrame) -> pd.DataFrame:
    ''' Returns a df showing NA counts for each col'''
    na_summary = df.isna().sum().reset_index()
    na_summary.columns = ['Columns', 'NA_count']
    return na_summary

def plot_missing_values(df: pd.DataFrame):
    ''' Plots a heatmap showing missing values in the dataset.'''
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis", yticklabels=False)
    plt.title("Missing Values Heatmap")
    plt.show()


def remove_nas(df: pd.DataFrame) -> pd.DataFrame:
    ''' Blanket removes NA values'''
    df_no_nas = df.dropna()
    print(f'Original row length: {len(df)}\n')
    print(f'Remaing rows: {len(df_no_nas)}')
    return df_no_nas 

def impute_nas(df: pd.DataFrame, strat: str = 'mean') -> pd.DataFrame:
    ''' Imputes missing values based on specified strategy mean, median, mode'''
    for col in df.select_dtypes(include=['int', 'float']).columns:
        if strat == 'mean':
            df[col].fillna(df[col].mean(), inplace=True) 
        elif strat == 'mode':
            df['col'].fillna(df[col].mode(), inplace=True) 
        elif strat == 'median':
            df['col'].fillna(df[col].median(), inplace=True)

    print(f'Missing values converted to {col} {strat}')
    return df

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    '''Converts column names to lowercase with underscores '''
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
    return df