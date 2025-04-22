# sleep_wave/features/registry.py 

from sleep_wave.features.builders import feat_band_diff, feat_band_rollmean, feat_bandpower_lags

def register_all_features():
    '''
    Register all feature functions in the module
    '''
    return [
        {
            'name': 'Band lag1 lag2',
            'func': feat_bandpower_lags,
            'notes': 'Adds 1- and 2-epoch lag feats for each band'
        },

        {
            'name': 'band_rollmean',
            'func': feat_band_rollmean,
            'notes': 'Adds rolling average (windows 3,5 epochs) for all bands'
        },

        {
            'name': 'band_diff',
            'func': feat_band_diff,
            'notes': 'Adds rate of change feat for bands'
        }
    ]