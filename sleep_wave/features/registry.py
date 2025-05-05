# sleep_wave/features/registry.py 

from sleep_wave.features.builders import feat_band_diff, feat_band_entropy, feat_band_ratios, feat_band_rollmean, feat_band_rollstd, feat_bandpower_base, feat_bandpower_lags, feat_time_context

def register_all_features():
    '''
    Register all feature functions in the module
    '''
    return [
        {
            'name': 'band_base',
            'func': feat_bandpower_base,
            'notes': 'Baseline band feats (delta, theta, alpha, beta, cycle, label, subject_id)'
        },

        {
            'name': 'band_lag1_2',
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
        }, 

        {
            'name': 'band_ratio',
            'func': feat_band_ratios,
            'notes': 'Adds band ratios (delta/theta, alpha/theta, etc.)'
         }, 

         {
             'name': 'band_rollstd', 
             'func': feat_band_rollstd,
             'notes': 'Adds rolling std (3, 5 window) for bands'
         }, 

         {
             'name': 'band_entropy', 
             'func': feat_band_entropy, 
             'notes': 'Shannon entropy across normalized bands'
         }, 

         {
             'name': 'time_frac', 
             'func':feat_time_context,
             'notes': 'Normalized time-of-night context'
         }
    ]