import torch
import torch.nn as nn 
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score, f1_score

from src.models.lstm_model import SleepLSTM
from src.models.train_lstm import train_lstm
from src.datasets.sequence_dataset import LSTMDataset
from src.logger_setup import logger

import numpy as np
import pandas as pd

def loso_lstm(df:pd.DataFrame, feature_cols, label_col='label',
              model_params=None, window_size=10, batch_size=32,
              lr=1e-3, n_epochs=10, target_subject=None, 
              verbose=True, device=None, bidirectional=False):
    '''
    Performs Leave-One-Subject-Out (LOSO) training and eval using LSTM

    Args: 
        df: full df with 'subject_id' features and label_col 
        feature_cols (list): List of cols to use as input feats
        label_col (str): Name of label col
        model_params (dict): Params passed into SleepLSTM 
        window_size (int): Number of timesteps in each seq 
        batch_size (int): Batch size for training
        lr (float): Learning rate
        n_epochs (int): Number of training epochs 
        target_subject (int): If provided, runs only that subject
        verbose (bool): Whether to print progress 
        device (str): 'cuda' or 'cpu' 

    Returns: 
        results (dict): Maps subject_id ot performance metrics
    '''
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    results = {} 

    subjects = [target_subject] if target_subject is not None else df['subject_id'].unique()

    for subject in subjects: 
        if verbose: 
            print(f'\n🚪 Leaving out Subject {subject} for LSTM training...\n')

        df_train = df[df['subject_id'] != subject].reset_index(drop=True) 
        df_test = df[df['subject_id'] == subject].reset_index(drop=True) 

        le = LabelEncoder()
        df_train[label_col] = le.fit_transform(df_train[label_col])
        df_test[label_col] = le.fit_transform(df_test[label_col]) 

        # Dataset + DataLoaders
        train_ds = LSTMDataset(df_train, feature_cols, label_col, window_size)
        test_ds = LSTMDataset(df_test, feature_cols, label_col, window_size) 

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False) 

        dataloaders = {'train': train_loader, 'val': test_loader}

        # Initialize the model! 
        input_size = len(feature_cols)
        num_classes = len(np.unique(df[label_col])) 

        lstm_model = SleepLSTM(
            input_size=input_size, 
            hidden_size=model_params.get('hidden_size', 64), 
            num_layers=model_params.get('num_layers', 1),
            num_classes=num_classes, 
            dropout=model_params.get('dropout', 0.0), 
            bidirectional=bidirectional
        ) 
        
        optimizer = torch.optim.Adam(lstm_model.parameters(), lr=lr) 
        loss_fn = nn.CrossEntropyLoss() 

        # Now train the model using `train_lstm` 
        lstm_model = train_lstm(
            model=lstm_model, 
            dataloaders=dataloaders,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device, 
            n_epochs=n_epochs,
            verbose=verbose
        )

        # Final Eval on the test set
        lstm_model.eval()
        all_preds, all_targets = [], []

        with torch.no_grad():
            for batch in test_loader:
                x, y = batch
                x, y = x.to(device), y.to(device) 

                logits = lstm_model(x)
                preds = torch.argmax(logits, dim=1) 

                all_preds.extend(preds.cpu().numpy()) 
                all_targets.extend(y.cpu().numpy()) 

        acc = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0) 

        if verbose: 
            print(f"✅ Subject {subject} | Acc: {acc:.4f} | F1: {f1:.4f}")

        results[subject] = {
            'subject_id': subject, 
            'accuracy': acc, 
            'weighted_f1': f1, 
            'all_preds': all_preds, 
            'all_targets': all_targets
        }

    return results 
              