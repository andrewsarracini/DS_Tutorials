# sleep_wave/utils/file_resolver.py

from pathlib import Path
from src.paths import DATA_DIR

def get_edf_paths(subject_id: str):
    subject_tag = f"ST{subject_id}J"
    psg_path = next(DATA_DIR.glob(f"{subject_tag}*-PSG.edf"), None)
    hypnogram_path = next(DATA_DIR.glob(f"{subject_tag}*-Hypnogram.edf"), None)
    if not psg_path:
        raise FileNotFoundError(f"PSG file for subject {subject_id} not found.")
    return psg_path, hypnogram_path

