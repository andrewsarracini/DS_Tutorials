import argparse
import pandas as pd
import json
from pathlib import Path

from src.paths import DATA_DIR, CONFIG_DIR, TUNED_PARAMS_DIR
from src.helper import eval_best_config

def main():
    parser = argparse.ArgumentParser(description='SleepWave CLI Tool')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Subcommand: eval_best_config
    eval_parser = subparsers.add_parser('eval-best', help='Evaluate saved best config on a subject')
    eval_parser.add_argument('--config', type=str, required=True, help='Name of saved config file (e.g., best_config_7242.json)')
    eval_parser.add_argument('--subject', type=int, required=True, help='Subject ID to evaluate')
    eval_parser.add_argument('--binary', action='store_true', help='Use binary REM/NREM labels')
    eval_parser.add_argument('--save', action='store_true', help='Save results to .md and .csv')

    # Subcommand: train_lstm
    train_parser = subparsers.add_parser('train-lstm', help='train LSTM on a subject')
    train_parser.add_argument('--subject', type=int, required=True, help='Subject ID to train on')
    train_parser.add_argument('--binary', action='store_true', help='Use binary REM/NREM labels')
    train_parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    train_parser.add_argument('--save', action='store_true', help='Save trained model to .pt file')

    args = parser.parse_args()

    if args.command == 'eval-best':
        config_path = TUNED_PARAMS_DIR / args.config
        df = pd.read_csv(DATA_DIR / 'eeg_hypno.csv')

        # Load saved config
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Static config setup
        target_col = 'binary_label' if args.binary else 'label'
        non_feat_cols = {'label', 'binary_label', 'subject_id'}
        feature_cols = [col for col in df.columns if col not in non_feat_cols]

        static_config = {
            'df': df,
            'feature_cols': feature_cols,
            'label_col': target_col,
            'is_binary': args.binary
        }

        # Run evaluation
        results_df = eval_best_config(
            config=config,
            subject_ids=[args.subject],
            static_config=static_config,
            save_md=args.save,
            save_csv=args.save
        )

        # Print results to terminal:
        for _, r in results_df.iterrows():
            print(f"\n📊 Subject {r['subject']} | F1: {r['f1']:.4f} | Accuracy: {r['accuracy']:.4f} | Threshold: {r['threshold']}")

    elif args.command == 'train-lstm':
        import torch
        import torch.nn as nn
        from src.models.train_lstm import train_lstm
        from src.models.lstm_model import SleepLSTM
        from src.helper import build_dataloaders
        from src.paths import MODEL_DIR

        subject = args.subject
        dataloaders = build_dataloaders(subject, binary=args.binary)
        
        model = SleepLSTM(
            input_size=4,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            is_binary=args.binary
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.BCEWithLogitsLoss() if args.binary else nn.CrossEntropyLoss()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = train_lstm(model, dataloaders, optimizer, loss_fn, device,
                           n_epochs=args.epochs, is_binary=args.binary)

        if args.save:
            path = MODEL_DIR / f"best_lstm_{subject}.pt"
            torch.save(model.state_dict(), path)
            print(f"✅ Saved model to {path}")

if __name__ == '__main__':
    main()
