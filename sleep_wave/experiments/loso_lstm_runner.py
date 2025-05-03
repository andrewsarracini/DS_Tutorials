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

    args = parser.parse_args()

    print('\nStarting LSTM LOSO experiment... \n')
    print("✅ Device:", torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU")

    # Load the data!
    df = pd.read_csv(DATA_DIR / 'eeg_hypno.csv')

    target_col = 'binary_label' if args.binary else 'label'

    loso_lstm(
        df=df, 
        target_subject=args.subject,
        target_col=target_col,
        seq_len=args.seq_len, 
        hidden_size=args.hidden_size,
        bidirectional=args.bidirectional,
        n_epochs=args.epochs,
        lr=args.lr
    )

if __name__ == '__main__':
    main()