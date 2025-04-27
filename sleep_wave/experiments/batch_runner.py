import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

from sleep_wave.features.registry import register_all_features
from src.paths import DATA_DIR, LOG_DIR
from src.models.loso import loso_full
from sleep_wave.experiments.run_experiments import run_feature_experiment_loso
from sleep_wave.cli.cli_utils import get_common_arg_parser, resolve_feats_to_run

def main(): 
    parser = get_common_arg_parser()
    args = parser.parse_args()

    df_edf = pd.read_csv(DATA_DIR / 'eeg_hypno.csv')
    all_features = register_all_features() 
    results_log = []

    features_to_run = resolve_feats_to_run(all_features, args)
   
    model_class = LGBMClassifier if args.model == 'lgbm' else RandomForestClassifier
    model_params = {'verbosity':-1, 'random_state':10} if args.model == 'lgbm' else {'random_state':10}

    models_to_try = ['lgbm', 'rf'] if args.batch_models else [args.model]

    for model_choice in models_to_try: 
        model_class = LGBMClassifier if model_choice == 'lgbm' else RandomForestClassifier
        model_params = {'verbosity':-1, 'random_state':10} if model_choice == 'lgbm' else {'random_state':10}

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
                model_params=model_params, 
                target_column='binary_label' if args.binary else 'label'
            )

            results_log.append(result)

    # Save results
    timestamp = datetime.now().strftime("%m%d")
    
    # csv naming: feats included
    if args.baseline:
        feature_code = 'baseline'
    elif args.last: 
        feature_code = 'last'
    else: 
        feature_code = 'allfeats'

    # csv naming: model 
    model_code = 'multi' if args.batch_models else args.model

    subject_code = f"s{args.subject}"
    
    filename = f'{feature_code}-{model_code}-{subject_code}-{timestamp}.csv'

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results_log).to_csv(LOG_DIR / filename, index=False)
    print("\n✅ Experiment(s) complete. Results saved.")

if __name__ == '__main__':
    main()