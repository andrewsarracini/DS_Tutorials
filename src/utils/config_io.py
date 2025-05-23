from pathlib import Path
import json

def save_best_params(params, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(params, f, indent=4)

def load_best_params(path):
    with open(path, 'r') as f:
        return json.load(f)