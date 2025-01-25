import pandas as pd
import json

def load_csv(filepath: str) -> pd.DataFrame:
    '''Loads data from a .csv file'''
    try: 
        return pd.read_csv(filepath)
    except Exception as ex:
        print(f'Error loading file: {filepath}')
        raise ex
    
def load_json(filepath: str) -> dict:
    '''Loads data from a JSON file.'''
    try:
        with open(filepath, 'r') as file:
            data = json.load(file)
        print(f"Loaded JSON data from {filepath}")
        return data
    except Exception as ex:
        print(f"Error loading JSON file: {filepath}")
        raise ex