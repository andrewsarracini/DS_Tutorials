import pandas as pd

def detect_na(df: pd.DataFrame) -> pd.DataFrame:
    '''Cleans and preprocesses the data'''

    # detect NA values
    return df.isna().sum()

# More to come