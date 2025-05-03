import argparse
from unicodedata import bidirectional 
import torch

from src.paths import DATA_DIR
from src.models.loso_lstm import loso_lstm
from sleep_wave.cli.cli_utils import get_common_arg_parser

import pandas as pd

def main(): 
    parser = get_common_arg_parser() 

    # Now adding LSTM-specific options!
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs') 
    parser.add_argument('--seq_len', type=int, default=20, help='Length of input sequences')
    parser.add_argument('--hidden_size', type=int, default=128, help='LSTM hidden size')
    parser.add_argument('--lr', type=int, default=1e-3, help='Learning rate')
    parser.add_argument('--bidirectional', action='store_true', help='Use bidirectional LSTM')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout between LSTM layers (only used if num_layers > 1)')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of stacked LSTM layers')

    args = parser.parse_args()

    print('\nStarting LSTM LOSO experiment...')

    # Load the data!
    df = pd.read_csv(DATA_DIR / 'eeg_hypno.csv')

    target_col = 'binary_label' if args.binary else 'label'
    non_feat_cols = {target_col, 'label', 'binary_label', 'subject_id'}
    feature_cols = [col for col in df.columns if col not in non_feat_cols]

    loso_lstm(
        df=df, 
        feature_cols=feature_cols,
        target_subject=args.subject,
        label_col=target_col,
        window_size=args.seq_len, 
        model_params={
            'hidden_size': args.hidden_size,
            'dropout': args.dropout, 
            'num_layers': args.num_layers
        },
        bidirectional=args.bidirectional,
        n_epochs=args.epochs,
        lr=args.lr
    )

if __name__ == '__main__':
    main()