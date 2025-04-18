# sleep_wave/features/registry.py 

from sleep_wave.features.builders import feat_bandpower_lags, feat_temporal_bandpower_t1

def register_all_features():
    '''
    Register all feature functions in the module
    '''
    return [
        {
            'name': '+- temporal bandpower',
            'func': feat_temporal_bandpower_t1,
            'notes': 'Adds t-1 and t+1 EEG bandpower features'
        },
        {
            'name': 'EEG band lag-1 and lag-2',
            'func': feat_bandpower_lags,
            'notes': 'Adds 1- and 2-epoch lag feats for each band'
        }
    ]