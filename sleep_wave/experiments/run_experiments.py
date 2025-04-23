# sleep_wave/experiments/run_experiments.py

# Applies a single feature function (from the registry) to your DataFrame
# Runs loso_full() with a specified model
# Adds metadata like feature name and subject
# Returns the evaluation results

from pyexpat import features
import pandas as pd 
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import argparse

from sleep_wave.features.registry import register_all_features
from src.paths import DATA_DIR, LOG_DIR
from src.models.loso import loso_full
from sleep_wave.cli.cli_utils import get_common_arg_parser, resolve_feats_to_run

def run_feature_experiment_loso(
        df:pd.DataFrame,
        feature_entry, 
        target_subject,
        model_class, 
        model_name, 
        n_trials=10, 
        save_plot=True,
        model_params=None): 
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
        model_params (dict): Optional kwargs to pass into model constructor

    Returns: 
        results (dict): Dictionary with feature metadata
    '''
    # Apply feature transformation: 
    df_feat = feature_entry['func'](df.copy())
    
    # Sanity check before training
    if df_feat.isna().any().any():
        raise ValueError(f"⚠️ NaNs detected after applying {feature_entry['name']}")

    stds = df_feat.drop(columns=['label', 'subject_id'], errors='ignore').std()
    low_variance = stds[stds < 1e-15]
    if not low_variance.empty:
        raise ValueError(f"⚠️ Near-zero variance in columns: {list(low_variance.index)} from {feature_entry['name']}")
    
    # Ensure model_params exists and suppress LightGBM verbosity if applicable
    if model_class == LGBMClassifier:
        model_params = {"verbose": -1, "class_weight": "balanced", "random_state": 10, **(model_params or {})}
    else:
        model_params = model_params or {"class_weight": "balanced", "random_state": 10}

    loso_result, best_params, final_metrics = loso_full(
        df=df_feat,
        model_class=model_class,
        model_name=model_name,
        n_trials=n_trials,
        save_plot=save_plot,
        target_subject=target_subject,
        model_params=model_params
    )

    return {
        'feature_name': feature_entry['name'],
        'notes': feature_entry.get('notes', ''),
        'target_subject': target_subject,
        'model_used': model_class.__name__,
        'accuracy': final_metrics.get('accuracy'),
        'weighted_f1': final_metrics.get('weighted_f1'),
        'precision': final_metrics.get('precision'),
        'recall': final_metrics.get('recall'),
        # Looks inside final_metrics dict (from eval_classification) 
        # for the sub-dict; per_class_f1
        # gets weighted_f1 for each class
        **{
            f"{cls}_f1": final_metrics.get("per_class_f1", {}).get(cls)
            for cls in ("N1", "N2", "N3", "REM", "Wake")
        },
        'n_trials': n_trials
    }

# Running the script from the Command Line
# =========================================================
def main(): 
    '''
    CLI entrypoint for running LOSO experiments over one or all engineered feats
    '''
    parser = get_common_arg_parser()
    args = parser.parse_args()

    df_edf = pd.read_csv(DATA_DIR / 'eeg_hypno.csv')
    all_features = register_all_features() 
    results_log = []

    features_to_run = resolve_feats_to_run(all_features, args)
   
    model_class = LGBMClassifier if args.model == 'lgbm' else RandomForestClassifier
    model_params = {'verbosity':-1} if args.model == 'lgbm' else None

    for feature_entry in features_to_run:
        model_name = f"{model_class.__name__}_{feature_entry['name'].replace(' ', '_')}"

        print(f"\n🔁 Running: {feature_entry['name']} on subject {args.subject}")
        result = run_feature_experiment_loso(
            df=df_edf,
            feature_entry=feature_entry,
            target_subject=args.subject,
            model_class=model_class,
            model_name=model_name,
            n_trials=args.trials, 
            model_params=model_params
        )

        results_log.append(result)

    # Save results
    timestamp = datetime.now().strftime("%m%d")
    
    if args.baseline:
        feature_code = 'baseline'
    elif args.last: 
        feature_code = 'last'
    else: 
        feature_code = 'allfeats;'

    model_code = 'lgbm' if args.model == 'lgbm' else 'rf' 
    subject_code = f"s{args.subject}"
    
    filename = f'{feature_code}-{model_code}-{subject_code}-{timestamp}.csv'

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results_log).to_csv(LOG_DIR / filename, index=False)
    print("\n✅ Experiment(s) complete. Results saved.")

if __name__ == '__main__':
    main()