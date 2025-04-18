# sleep_wave/features/registry.py 

from sleep_wave.features.builders import feat_bandpower_lags

def register_all_features():
    '''
    Register all feature functions in the module
    '''
    return [
        {
            'name': 'EEG band lag-1 and lag-2',
            'func': feat_bandpower_lags,
            'notes': 'Adds 1- and 2-epoch lag feats for each band'
        }
    ] # Add more features here! 