# === Standard Library ===
import json
from pathlib import Path
from collections import Counter

# === Third-Party Libraries ===
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.calibration import LabelEncoder
from sklearn.metrics import f1_score, precision_recall_curve, confusion_matrix

# === Local Modules ===
from src.logger_setup import logger

def save_best_params(params, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(params, f, indent=4)

def load_best_params(path):
    with open(path, 'r') as f:
        return json.load(f)
    
# from helper.py
from src.datasets.sequence_dataset import LSTMDataset
from torch.utils.data import DataLoader

def build_dataloaders(df_train, df_test, feature_cols, label_col, window_size, stride, batch_size):
    train_ds = LSTMDataset(df_train, feature_cols, label_col, window_size, stride)
    test_ds = LSTMDataset(df_test, feature_cols, label_col, window_size, stride)

    return {
        'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        'val': DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    }

from src.paths import ENCODER_DIR, REPORT_DIR

# from helper.py
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path
from src.paths import ENCODER_DIR

def encode_labels(df_train, df_test, label_col, subject_id, save_dir=ENCODER_DIR):
    """
    Applies LabelEncoder to train/test splits and saves the encoder.

    Returns:
        df_train (DataFrame): with encoded labels
        df_test (DataFrame): with encoded labels
        le (LabelEncoder): fitted encoder
        encoder_path (Path): path to saved encoder
    """
    le = LabelEncoder()
    df_train[label_col] = df_train[label_col].copy()
    df_test[label_col] = df_test[label_col].copy()
    
    df_train[label_col] = le.fit_transform(df_train[label_col])
    df_test[label_col] = le.transform(df_test[label_col])

    encoder_path = save_dir / f'le_s{subject_id}.pkl'
    joblib.dump(le, encoder_path)

    return df_train, df_test, le, encoder_path
        
# from helper.py 
# Eval-- plugs directly into plot_threshold_curves
def find_best_threshold(y_true, y_probs, metric='f1', plot=True):
    '''
    Finds optimal classification thresh for a given metric (defauly f1)

    Args: 
        y_true: Ground truth binary labels
        y_probs: Predicted probs for target class
        metric: f1 or recall
        plot: Whether to plot the metric vs thresh curve

    Returns: 
        best_thresh: Thresh that maximizes f1 (or recall)
    '''

    thresholds = np.linspace(0.0, 1.0, 100)
    scores = []

    for t in thresholds: 
        y_pred = (y_probs >= t).astype(int) 
        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0) 
        elif metric == 'recall':
            precision, recall, _ = precision_recall_curve(y_true, y_probs)
            idx = np.argmin(np.abs(thresholds - t)) 
            score = recall[idx] if idx < len(recall) else 0
        else: 
            raise ValueError('Metric must be f1 or recall')
        scores.append(score)

    best_idx = np.argmax(scores)
    best_thresh = thresholds[best_idx]

    if plot:
        plt.figure(figsize=(8,4))
        plt.plot(thresholds, scores, label=f'{metric} score') 
        plt.axvline(best_thresh, color='red', linestyle='--', label=f'Best Threshold = {best_thresh:.2f}')
        plt.xlabel('Threshold') 
        plt.ylabel(f'{metric.upper()} Score')
        plt.title(f'{metric.upper()} vs Threshold')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.show() 

    return best_thresh

def print_eval_summary(preds, targets, encoder_path): 
    '''
    Prints a readable confusion matrix and class distribution summary

    Args: 
        preds (list[int]): Predicted label indices
        target (list[int]): Ground-truth label indices
        encoder_path (str or Path): Path to saved LabelEncoder .pkl
    '''
    le = joblib.load(Path(encoder_path)) 
    decoded_preds = le.inverse_transform(preds)
    decoded_targets = le.inverse_transform(targets)

    # 1. === Confusion Matrix === 
    print("--- Confusion Matrix ---")
    cm = confusion_matrix(decoded_targets, decoded_preds, labels=le.classes_)
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    print(cm_df.to_string())

    # === 2. Class distribution
    pred_counts = Counter(decoded_preds)
    true_counts = Counter(decoded_targets)

    dist_df = pd.DataFrame({
        "Label": le.classes_,
        "Predicted Count": [pred_counts.get(label, 0) for label in le.classes_],
        "Actual Count": [true_counts.get(label, 0) for label in le.classes_]
    })

    logger.info("\n--- Class Distribution ---\n" + dist_df.to_string(index=False))
    print("\n--- Class Distribution ---\n" + dist_df.to_string(index=False))

# from helper 
# circular import made me put it here:
def flatten_outputs_targets(outputs, targets, is_binary): 
    '''
    Reshape outputs and targets for loss calc 

    Args: 
        outputs (Tensor): Model outputs
        targets (Tensor): Ground truth labels 
        is_binary (Bool): Whether binary classification

    Returns
        tuple (Tensor, Tensor): Flattened outputs and targets
    '''

    if is_binary: 
        outputs = outputs.squeeze(-1)
        outputs = outputs.view(-1) 
        targets = targets.view(-1).float()
    else: 
        outputs = outputs.view(-1, outputs.size(-1)) 
        targets = targets.view(-1)
    return outputs, targets