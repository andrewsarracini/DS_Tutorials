import pandas as pd

def load_data(filepath: str) -> pd.DataFrame:
    '''Loads data from a .csv file'''
    
    try: 
        return pd.read_csv(filepath)
    except Exception as ex:
        print(f'Error loading file: {filepath}')
        raise ex
    
    