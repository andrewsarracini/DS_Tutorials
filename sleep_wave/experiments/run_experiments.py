# sleep_wave/experiments/run_experiments.py

# Applies a single feature function (from the registry) to your DataFrame
# Runs loso_full() with a specified model
# Adds metadata like feature name and subject
# Returns the evaluation results

import pandas as pd 
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import argparse
import os

from sleep_wave.features.registry import register_all_features
from src.paths import DATA_DIR, LOG_DIR
from src.models.loso import loso_full

def run_feature_experiment_loso(
        df:pd.DataFrame,
        feature_entry, 
        target_subject,
        model_class, 
        model_name, 
        n_trials=30, 
        save_plot=True): 
    '''
    Applies a feature func, runs LOSO and logs results (RESULT LOG)

        - load df_edf
        = for each feature:
            - apply feature func to df
            - run loso_full() on that subject with that model
            - record results
        - save all results to CSV

    Parameters: 
        base_df (pd.DataFrame):  df_edf (raw)
        feature_entry (dict): Feat registry ('name', 'func', 'notes', etc.)
        target_subject (int): Subject ID to leave out and test on
        model_class (sklearn): ex. RandomForestClassifier
        model_name (str): Name for model tracking
        n_trials (int): Number of trials for Optuna
        save_plot (bool): Whether or not to save plots

    Returns: 
        results (dict): Dictionary with feature metadata
    '''
    # Apply feature transformation: 
    df_feat = feature_entry['func'](df.copy())

    results = loso_full(
        df=df_feat,
        model_class=model_class,
        model_name=model_name,
        n_trials=n_trials,
        save_plot=save_plot,
        target_subject=target_subject
    )

    results['feature_name'] = feature_entry['name']
    results['notes'] = feature_entry.get('notes', '')
    results['target_subject'] = target_subject
    results['model_used'] = model_class.__name__

    return results

# Running the script from the Command Line
# =========================================================
def main(): 
    '''
    CLI entrypoint for running LOSO experiments over one or all engineered feats
    '''
    parser = argparse.ArgumentParser() 
    parser.add_argument('--subject', type=int, default=7242, help='Target subject for LOSO') 
    parser.add_argument('--single', action='store_true', help='Run single feature func only')
    parser.add_argument('--model', type=str, default='lgbm', choices=['lgbm, rf'], help='Model selection')
    args = parser.parse_args()

    df_edf = pd.read_csv(DATA_DIR / 'eeg_hypno.csv')
    all_features = register_all_features() 
    results_log = []

    features_to_run = [all_features[0]] if args.single else all_features
    # LGBM is default, select RF to toggle
    model_class = LGBMClassifier if args.model == 'lgbm' else RandomForestClassifier

    for feature_entry in features_to_run:
        model_name = f"{model_class.__name__}_loso_{feature_entry['name'].replace(' ', '_')}"

        print(f"\n🔁 Running: {feature_entry['name']} on subject {args.subject}")
        result = run_feature_experiment_loso(
            df=df_edf,
            feature_entry=feature_entry,
            target_subject=args.subject,
            model_class=model_class,
            model_name=model_name,
            n_trials=30
        )

        results_log.append(result)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    feature_code = features_to_run[0]['name'].lower().split()[0] if args.single else 'all'
    model_code = model_class.__name__.lower()
    subject_code = f"s{args.subject}"
    
    filename = f'loso-{feature_code}-{model_code}-{subject_code}-{timestamp}.csv'

    pd.DataFrame(results_log).to_csv(LOG_DIR / filename, index=False)
    print("\n✅ Experiment(s) complete. Results saved.")

if __name__ == '__main__':
    main()