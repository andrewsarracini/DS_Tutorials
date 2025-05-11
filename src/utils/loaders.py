from src.paths import DATA_DIR

import pandas as pd
from pathlib import Path


def load_eeg_data(filename='eeg_hypno.csv'):
    local_path = DATA_DIR / filename 
    remote_url = f'https://raw.githubusercontent.com/andrewsarracini/DS_Tutorials/main/data/{filename}'

    if local_path.exists():
        print(f"[INFO] Loading from local: {local_path}")
        return pd.read_csv(local_path)
    
    try:
        print(f"[WARNING] Local file not found. Trying GitHub: {remote_url}")
        return pd.read_csv(remote_url)
    except Exception as e:
        print(f"[ERROR] Could not load file. You may need to generate or download `{filename}` manually.")
        print(f"Details: {e}")
        raise
