import argparse
import pandas as pd
from pathlib import Path
import json 

from src.paths import DATA_DIR, CONFIG_DIR
from src.helper import eval_best_config

# Add more tools later!

def main():
    parser = argparse.ArgumentParser(description='SleepWave CLI Tool')
    subparsers = parser.add_subparsers(dest='command', required=True) 

    # Subcommand: eval_best_config
    eval_parser = subparsers.add_parser('eval-best', help='Evaluate saved best configs on a target subject')
    eval_parser.add_argument('--config', type=str, required=True, help='Name of saved config file (ex. best_config_7242.json)')
    eval_parser.add_argument('--subject', type=int, required=True, help='Subject ID to eval on')
    eval_parser.add_argument('--binary', action='store_true', help='Use binary REM/NREM labels')
    eval_parser.add_argument('--save', action='store_true', help='Save results to md and csv')

    args = parser.parse_args()

    # Eval: 

    if args.command == 'eval_best':
        config_path = CONFIG_DIR / args.config
        df = pd.read_csv(DATA_DIR / 'eeg_hypno.csv')

    with open(config_path, 'r') as f:
        config = json.load(f)

    target_col = 'binary_label' if args.binary else 'label'
    non_feat_cols = {'label', 'binary_label', 'subject_id'}
    feature_cols = [col for col in df.columns if col not in non_feat_cols]

    static_config = {
        'df': df,
        'feature_cols': feature_cols,
        'label_col': target_col,
        'is_binary': args.binary
    }

    subject_list = [args.subject]

    results_df = eval_best_config(
        config=config,
        subject_ids=subject_list,
        static_config=static_config,
        save_md=args.save,
        save_csv=args.save
    )

    for r in results_df.iterrows():
        print(f"\n📊 Subject {r['subject']} | F1: {r['f1']:.4f} | Accuracy: {r['accuracy']:.4f} | Threshold: {r['threshold']}")

if __name__ == '__main__':
    main()
    