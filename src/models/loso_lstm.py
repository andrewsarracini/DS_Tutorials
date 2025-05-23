import torch
import torch.nn as nn 
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

from src.models.lstm_model import SleepLSTM
from src.models.train_lstm import train_lstm
from src.utils.lstm_utils import build_dataloaders, encode_labels, find_best_threshold, print_eval_summary
from src.logger_setup import logger
from src.paths import PLOT_DIR

# -----------------------------------------------------------------------
# OLD loso_lstm() implementation

# def loso_lstm(df: pd.DataFrame, feature_cols, label_col='label',
#               model_params=None, window_size=10, stride=None,
#               batch_size=32, lr=1e-3, n_epochs=10, target_subject=None,
#               verbose=True, device=None, bidirectional=False,
#               dropout=0.0, num_layers=1, loss_fn=None,
#               is_binary=False, threshold=0.5, plot_thresholds=False,
#               auto_thresh=False):
# ------------------------------------------------------------------------

def loso_lstm(config):
    # === Unpack config ===
    # Config is passed in through src/tune

    df = config["df"]
    feature_cols = config["feature_cols"]
    label_col = config.get("label_col", "label")
    target_subject = config.get("target_subject")
    verbose = config.get("verbose", True)

    hidden_size = config["hidden_size"]
    num_layers = config["num_layers"]
    dropout = config["dropout"]
    bidirectional = config["bidirectional"]
    lr = config["learning_rate"]
    stride = config["stride"]
    window_size = config["seq_len"]
    n_epochs = config["epochs"]
    batch_size = config["batch_size"]
    weight_decay = config.get("weight_decay", 0.0)
    is_binary = config.get("is_binary", False)
    threshold = config.get("threshold", 0.5)
    plot_thresholds = config.get("plot_thresholds", False)
    auto_thresh = config.get("auto_thresh", False)
    

    # device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    device = config.get('device', None)
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"[INFO] Using device: {device}")

    subjects = [target_subject] if target_subject is not None else df['subject_id'].unique()

    for subject in subjects:
        if verbose:
            print(f'\n🚪 Leaving out Subject {subject} for LSTM training...\n')

        df_train = df[df['subject_id'] != subject].reset_index(drop=True)
        df_test = df[df['subject_id'] == subject].reset_index(drop=True)
        df_train, df_test, le, encoder_path = encode_labels(df_train, df_test, label_col, subject)

        # Class weights (multiclass only)
        class_weights = None
        if df_train[label_col].nunique() > 2:
            classes = np.unique(df_train[label_col])
            class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=df_train[label_col])
            class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
            if verbose:
                print(f'[DEBUG] Class Weights (multi): {class_weights}')

        dataloaders = build_dataloaders(df_train, df_test, feature_cols, label_col, window_size, stride, batch_size)

        input_size = len(feature_cols)
        num_classes = 1 if is_binary else len(np.unique(df_train[label_col]))

        lstm_model = SleepLSTM(
            input_size=input_size,
            hidden_size=hidden_size, 
            num_layers=num_layers,
            num_classes=num_classes, 
            dropout=dropout,
            bidirectional=bidirectional
        )
        
        optimizer = torch.optim.Adam(lstm_model.parameters(),
                                      lr=lr, 
                                      weight_decay=weight_decay)

        # Binary or Multiclass loss
        if is_binary:
            class_counts = df_train[label_col].value_counts().to_dict()
            neg_count = class_counts.get(0, 1)
            pos_count = class_counts.get(1, 1)
            imbalance_ratio = neg_count / pos_count
            if verbose:
                print(f'[INFO] Subject {subject}: Class Imbalance Ratio: {imbalance_ratio:.2f} (neg:pos)')
            pos_weight = torch.tensor([imbalance_ratio]).to(device)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        # === Train === 
        lstm_model = train_lstm(
            model=lstm_model,
            dataloaders=dataloaders,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            n_epochs=n_epochs,
            verbose=verbose,
            is_binary=is_binary,
            threshold=threshold  
        )

        # === Eval === 
        lstm_model.eval()
        all_targets, all_probs = [], []

        with torch.no_grad():
            for batch in dataloaders['val']:
                x, y = batch
                x, y = x.to(device), y.to(device)
                logits = lstm_model(x)

                if is_binary:
                    probs = torch.sigmoid(logits).squeeze(-1)
                    all_probs.append(probs.cpu().numpy().reshape(-1))
                else:
                    preds = torch.argmax(logits, dim=-1)
                    all_targets.append(y.cpu().numpy().reshape(-1))

        all_targets = np.concatenate([y.cpu().numpy().reshape(-1) for _, y in dataloaders['val']])
        all_probs = np.concatenate(all_probs) if is_binary else None

        # Auto threshold logic
        if is_binary and auto_thresh:
            # only import if necessary
            from src.eval import find_best_thresh

            best_thresh, best_score = find_best_thresh(all_targets, all_probs, metric='f1')
            threshold = best_thresh

        # Now apply threshold to get Final Preds
        if is_binary:
            all_preds = (all_probs > threshold).astype(int)
        else:
            all_preds = np.concatenate([
                torch.argmax(lstm_model(x.to(device)), dim=-1).cpu().numpy().reshape(-1)
                for x, _ in dataloaders['val']
            ])

        # Plot threshold curves if requested
        if is_binary and plot_thresholds and all_probs is not None:

            from src.eval import plot_threshold_curves

            plot_path = PLOT_DIR / f'thresh_s{subject}.png'
            plot_threshold_curves(
                y_true=all_targets,
                y_probs=all_probs,
                model_name=f'LSTM_s{subject}',
                highlight_threshold=threshold,
                save_path=plot_path
            )

        # Reporting
        report = classification_report(
            all_targets, all_preds,
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

    print(f'\n[METRICS] | Weighted F1: {f1:.4f} | Accuracy: {acc:.4f}\n')

    return {
        'val_targets': all_targets, 
        'val_probs': all_probs if is_binary else None,
        'val_preds': all_preds, 
        'f1_weighted': f1, 
        'accuracy': acc,
        'subject_id': subject,
        'threshold': threshold
    }
