import argparse 
import pandas as pd 

from src.paths import DATA_DIR
from src.tune import optuna_lstm_tuner

def main(): 
    parser = argparse.ArgumentParser(description='Run Optuna tuning across subjects') 
    parser.add_argument('--trials', type=int, default=10, help='Number of Optuna trials')
    parser.add_argument('--dataset', type=str, default='eeg_hypno.csv')
    parser.add_argument('--binary', action='store_true', help='Use binary REM/NREM labels')
    parser.add_argument('--subject', type=int, nargs='*', default=None, help='Subject(s) to tune on. Default: all subjects')
    args = parser.parse_args()

    df = pd.read_csv(DATA_DIR / args.dataset) 
    all_subjects = sorted(df['subject_id'].unique())
    subjects_to_tune = args.subject if args.subject is not None else all_subjects

    target_col = 'binary_label' if args.binary else 'label'
    non_feat_cols = {target_col, 'label', 'binary_label', 'subject_id'}
    feature_cols = [col for col in df.columns if col not in non_feat_cols]

    static_config = {
        'df': df, 
        'feature_cols': feature_cols, 
        'label_col': target_col, 
        'is_binary': args.binary, 
        'verbose': False,
        'auto_thresh': True, 
        'plot_thresholds': False
    }
    
    print(f"\n🚀 Tuning LSTM | Trials: {args.trials} | Subjects: {subjects_to_tune}\n")

    best_params = study = optuna_lstm_tuner(
        n_trials=args.trials, 
        random_state=10, 
        static_config=static_config, 
        subject_list = subjects_to_tune
    )

    print('\n✅ Best params found:')
    for k, v in best_params.items(): 
        print(f'  {k}: {v}')

if __name__ == '__main__':
    main()