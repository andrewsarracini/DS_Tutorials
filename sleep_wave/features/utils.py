# sleep_wave/features/utils.py

import numpy as np

def bandpower(psds, freqs, band):
    idx = (freqs >= band[0]) & (freqs <= band[1])
    
    return psds[:, :, idx].mean(axis=-1).mean(axis=1)

