# sleep_wave/utils/file_resolver.py

from pathlib import Path
from src.paths import DATA_DIR

def get_edf_paths(subject_id: str):
    '''
    Given a subject ID (ex. 7011), returns matching PSG and Hypnogram EDF paths
    '''

    subject_tag = f"ST{subject_id}J"
    subject_dir = DATA_DIR / 'sleep_waves'

    # Match PSG
    psg_candidates = list(subject_dir.glob(f"{subject_tag}*-PSG.edf"))
    if not psg_candidates:
        raise FileNotFoundError(f"PSG file for subject {subject_id} not found.")
    psg_path = psg_candidates[0]

    # Match Hypno
    hypnogram_candidates = list(subject_dir.glob(f"{subject_tag}*-Hypnogram.edf"))
    hypnogram_path = hypnogram_candidates[0] if hypnogram_candidates else None

    return psg_path, hypnogram_path
