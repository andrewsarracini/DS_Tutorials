# scripts/tool_runner.py

import argparse
import pandas as pd
import json
from pathlib import Path
import torch
import torch.nn as nn

from src.paths import DATA_DIR, CONFIG_DIR, TUNED_PARAMS_DIR, MODEL_DIR
from src.helper import eval_best_config, split_subject_data
from src.utils.lstm_utils import build_dataloaders
from src.models.lstm_model import SleepLSTM
from src.models.train_lstm import train_lstm


def main():
    parser = argparse.ArgumentParser(description='SleepWave CLI Tool')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Subcommand: Evaluate best config
    eval_parser = subparsers.add_parser('eval-best', help='Evaluate saved best config on a subject')
    eval_parser.add_argument('--config', type=str, required=True, help='Name of saved config file (e.g., best_config_7242.json)')
    eval_parser.add_argument('--subject', type=int, required=True, help='Subject ID to evaluate')
    eval_parser.add_argument('--binary', action='store_true', help='Use binary REM/NREM labels')
    eval_parser.add_argument('--save', action='store_true', help='Save results to .md and .csv')

    # Subcommand: Train and save LSTM
    train_parser = subparsers.add_parser('train-lstm', help='Train LSTM model and save .pt')
    train_parser.add_argument('--config', type=str, required=True, help='Name of saved config file (e.g., LSTM_best_7242.json)')
    train_parser.add_argument('--subject', type=int, required=True, help='Subject ID to train on')
    train_parser.add_argument('--binary', action='store_true', help='Use binary REM/NREM labels')
    train_parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')

    args = parser.parse_args()

    if args.command == 'eval-best':
        config_path = TUNED_PARAMS_DIR / args.config
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

        results_df = eval_best_config(
            config=config,
            subject_ids=[args.subject],
            static_config=static_config,
            save_md=args.save,
            save_csv=args.save
        )

        for _, r in results_df.iterrows():
            print(f"\n📊 Subject {r['subject']} | F1: {r['f1']:.4f} | Accuracy: {r['accuracy']:.4f} | Threshold: {r['threshold']}")

    elif args.command == 'train-lstm':
        config_path = TUNED_PARAMS_DIR / args.config
        with open(config_path, 'r') as f:
            config = json.load(f)

        df = pd.read_csv(DATA_DIR / 'eeg_hypno.csv')
        target_col = 'binary_label' if args.binary else 'label'
        non_feat_cols = {'label', 'binary_label', 'subject_id'}
        feature_cols = [col for col in df.columns if col not in non_feat_cols]

        # Split subject train/test
        df_train = df[df['subject_id'] != args.subject]
        df_test  = df[df['subject_id'] == args.subject]

        # ✅ Encode labels
        from src.utils.lstm_utils import encode_labels
        df_train, df_test, le, encoder_path = encode_labels(
            df_train, df_test, label_col=target_col, subject_id=args.subject
        )

        # Build DataLoaders
        from src.utils.lstm_utils import build_dataloaders
        dataloaders = build_dataloaders(
            df_train=df_train,
            df_test=df_test,
            feature_cols=feature_cols,
            label_col=target_col,
            window_size=config['seq_len'],
            stride=config['stride'],
            batch_size=32
        )

        # Initialize model
        from src.models.lstm_model import SleepLSTM
        model = SleepLSTM(
            input_size=len(feature_cols),
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            bidirectional=config['bidirectional']
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        loss_fn = nn.BCEWithLogitsLoss() if args.binary else nn.CrossEntropyLoss()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Establishing the save path earlier
        model_save_path = MODEL_DIR / f"best_lstm_{args.subject}.pt"

        # Train
        from src.models.train_lstm import train_lstm
        trained_model = train_lstm(
            model=model,
            dataloaders=dataloaders,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            n_epochs=args.epochs,
            verbose=True,
            is_binary=args.binary,
            threshold=config.get('threshold', 0.5),
            save_path = model_save_path
        )

        # Save model checkpoint
        torch.save(trained_model.state_dict(), model_save_path)
        print(f"✅ Model saved to {model_save_path}")


if __name__ == '__main__':
    main()
