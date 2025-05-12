import argparse
import torch
import torch.nn as nn 

from src.models.loso_lstm import loso_lstm
from sleep_wave.cli.cli_utils import get_common_arg_parser
from src.utils.loaders import load_eeg_data

import pandas as pd


def main(): 
    parser = get_common_arg_parser()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Now adding LSTM-specific options!
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs') 
    parser.add_argument('--seq_len', type=int, default=20, help='Length of input sequences')
    parser.add_argument('--hidden_size', type=int, default=128, help='LSTM hidden size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--bidirectional', action='store_true', help='Use bidirectional LSTM')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout between LSTM layers (only used if num_layers > 1)')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of stacked LSTM layers')
    parser.add_argument('--stride', type=int, default=None, help='Stride for LSTM windowing, default:None')
    parser.add_argument('--threshold', type=float, default=0.5, help='Thresh for sigmoid output in binary classification')
    parser.add_argument('--plot_thresholds', action='store_true', help='Plot ROC/PR curves after training (binary only)')
    parser.add_argument('--auto_thresh', action='store_true', help='Find threshold that maximizes F1 on validation set')


    args = parser.parse_args()

    print('\nStarting LSTM LOSO experiment...')

    # Load the data!
    # df = pd.read_csv(DATA_DIR / 'eeg_hypno.csv')
    df = load_eeg_data('eeg_hypno.csv')

    target_col = 'binary_label' if args.binary else 'label'
    non_feat_cols = {target_col, 'label', 'binary_label', 'subject_id'}
    feature_cols = [col for col in df.columns if col not in non_feat_cols]

    loso_lstm(
        df=df, 
        feature_cols=feature_cols,
        target_subject=args.subject,
        label_col=target_col,
        window_size=args.seq_len, 
        stride=args.stride,
        model_params={
            'hidden_size': args.hidden_size,
            'dropout': args.dropout, 
            'num_layers': args.num_layers
        },
        bidirectional=args.bidirectional,
        n_epochs=args.epochs,
        lr=args.lr, 
        loss_fn=None,
        is_binary=args.binary,
        threshold=args.threshold,
        plot_thresholds=args.plot_thresholds
    )

if __name__ == '__main__':
    main()