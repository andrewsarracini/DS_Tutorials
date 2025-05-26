import argparse 
import pandas as pd 
from datetime import datetime

from src.paths import DATA_DIR, REPORT_DIR
from src.tune import optuna_lstm_tuner
from src.helper import write_study_summary_md

def main(): 
    parser = argparse.ArgumentParser(description='Run Optuna tuning across subjects') 
    parser.add_argument('--trials', type=int, default=10, help='Number of Optuna trials')
    parser.add_argument('--dataset', type=str, default='eeg_hypno.csv')
    parser.add_argument('--binary', action='store_true', help='Use binary REM/NREM labels')
    parser.add_argument('--subject', type=int, nargs='*', default=None, help='Subject(s) to tune on. Default: all subjects')
    parser.add_argument("--export-csv", action="store_true", help="Export subject scores to CSV")
    parser.add_argument("--dryrun", action="store_true", help="Run a minimal tuning pass for debugging")
    parser.add_argument("--analyze", action="store_true", help="Generate Optuna study analysis plots after tuning")


    args = parser.parse_args()

    df = pd.read_csv(DATA_DIR / args.dataset) 
    all_subjects = sorted(df['subject_id'].unique())
    subjects_to_tune = args.subject if args.subject is not None else all_subjects

    if args.dryrun:
        print("⚠️ Running in DRYRUN mode (1 trial, 1 subject)")
        args.trials = 1
        subjects_to_tune = [7011]

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

    best_params, study = optuna_lstm_tuner(
        n_trials=args.trials, 
        random_state=10, 
        static_config=static_config, 
        subject_list = subjects_to_tune, 
        export_csv=args.export_csv
    )

    print('\n✅ Best params found:')
    for k, v in best_params.items(): 
        print(f'  {k}: {v}')

    timestamp = datetime.now().strftime('%Y-%m-%d')
    subject_str = args.subject[0] if args.subject and len(args.subject) == 1 else 'Multiple Subjects'
    folder_name = f"study_{subject_str}_{timestamp}"
    output_dir = REPORT_DIR / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)


    # Write Markdown Summary
    write_study_summary_md(
        study=study, 
        subject=subject_str,
        out_dir=output_dir
    )

    if args.analyze:
        from src.tune import analyze_study  
        analyze_study(study, output_dir=output_dir)

if __name__ == '__main__':
    main()