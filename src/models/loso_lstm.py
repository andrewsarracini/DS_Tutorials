import torch
import torch.nn as nn 
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

from src.models.lstm_model import SleepLSTM
from src.models.train_lstm import train_lstm
from src.datasets.sequence_dataset import LSTMDataset
from src.helper import print_eval_summary
from src.logger_setup import logger

import numpy as np
import pandas as pd
import joblib
import tqdm

from src.paths import ENCODER_DIR

def loso_lstm(df:pd.DataFrame, feature_cols, label_col='label',
              model_params=None, window_size=10, stride=None, 
              batch_size=32, lr=1e-3, n_epochs=10, target_subject=None, 
              verbose=True, device=None, bidirectional=False,
              dropout=0.0, num_layers=1):
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
        df_test[label_col] = le.transform(df_test[label_col]) 

        # Saving encoder
        ENCODER_DIR.mkdir(parents=True, exist_ok=True)

        encoder_path = ENCODER_DIR / f"label_encoder_s{subject}.pkl"
        joblib.dump(le, encoder_path)

        # Save a readable class mapping .txt
        class_map_path = ENCODER_DIR / f'classes_s{subject}.txt'
        with open(class_map_path, 'w') as f:
            for i, label in enumerate(le.classes_):
                f.write(f'{i} = {label}\n')

        class_weights = None
        if df_train[label_col].nunique() > 2: # if multiclass: 
            classes = np.unique(df_train[label_col])
            class_weights = compute_class_weight(class_weight='balanced',
                                                 classes=classes,
                                                 y=df_train[label_col])
            class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
            if verbose:
                print(f'[DEBUG] Class Weights (multi): {class_weights}')

        # Dataset + DataLoaders
        train_ds = LSTMDataset(df_train, feature_cols, label_col, window_size, stride) 
        test_ds = LSTMDataset(df_test, feature_cols, label_col, window_size, stride) 

        print(f"[INFO] Train seqs: {len(train_ds)} | Val seqs: {len(test_ds)}")

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False) 

        dataloaders = {'train': train_loader, 'val': test_loader}

        # Initialize the model! 
        input_size = len(feature_cols)
        num_classes = len(np.unique(df_train[label_col])) 

        lstm_model = SleepLSTM(
            input_size=input_size, 
            hidden_size=model_params.get('hidden_size', 64), 
            num_layers=model_params.get('num_layers', 1),
            num_classes=num_classes, 
            dropout=model_params.get('dropout', 0.0), 
            bidirectional=bidirectional
        ) 
        
        optimizer = torch.optim.Adam(lstm_model.parameters(), lr=lr) 
        loss_fn = nn.CrossEntropyLoss(weight=class_weights) 

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
                preds = torch.argmax(logits, dim=-1) 

                all_preds.append(preds.cpu().numpy().reshape(-1))
                all_targets.append(y.cpu().numpy().reshape(-1))

        # Flatten to match sklearn expectations!
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        report = classification_report(
            all_targets,
            all_preds,
            target_names=le.classes_,
            zero_division=0, 
            digits=4, 
            output_dict=True
        )

        acc = report['accuracy']
        f1 = report['weighted avg']['f1-score']

        if verbose: 
            logger.info('\n--- Per-Class F1 Scores ---') 
            for label in le.classes_:
                stats = report[label]
                logger.info(f"{label:<5} | F1: {stats['f1-score']:.4f} | Precision: {stats['precision']:.4f} | Recall: {stats['recall']:.4f} | Support: {int(stats['support'])}")

            print_eval_summary(all_preds, all_targets, encoder_path)

        results[subject] = {
            'subject_id': subject, 
            'accuracy': acc, 
            'weighted_f1': f1, 
            'all_preds': all_preds, 
            'all_targets': all_targets,
            'encoder_path': str(encoder_path),
            'report': report
        }

    return results 
              