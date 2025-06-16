# sleep_wave/utils/file_resolver.py

from pathlib import Path
from src.paths import DATA_DIR

def get_edf_paths(subject_id: str):
    '''
    Given a subject ID (ex. 7011), returns matching PSG and Hypnogram EDF paths
    '''

    subject_tag = f'ST{subject_id}'

    # Match PSG
    psg_candidates = list(DATA_DIR.glob(f'{subject_tag}*-PSG.edf'))
    if not psg_candidates:
        raise FileNotFoundError(f'PSG file for s{subject_id} not found')
    if len(psg_candidates) > 1:
        print(f'[WARN] Multiple PSG files found for s{subject_id}, using first: {psg_candidates[0]}')
    psg_path = psg_candidates[0] 

    # Match Hypno
    hypnogram_candidates = list(DATA_DIR.glob(f'{subject_tag}*-Hypnogram.edf'))
    if not hypnogram_candidates:
        print(f'[WARN] No Hypnogram found for subject s{subject_id}. No labels applied')
        hypnogram_path = None
    else:
        hypnogram_path = hypnogram_candidates[0]
        if len(hypnogram_candidates) > 1: 
            print(f'[WARN] Multiple Hypnogram files found for s{subject_id}, using first: {hypnogram_candidates[0]}')
        
    return psg_path, hypnogram_path