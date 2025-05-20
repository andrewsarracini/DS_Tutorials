import os 
import json
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, LabelEncoder
from sklearn.metrics import f1_score, precision_recall_curve, log_loss
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from src.datasets.sequence_dataset import LSTMDataset

# from tune.py
def save_best_params(best_params, model_name, save_dir='../tuned_params'):
    '''
    Saves best hyperparameters into a file one level back called tuned_params

    Example Usage:
        save_best_params(best_params, model.__class__.__name__)
    '''

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{model_name}_best_params.json') 
    with open(save_path, 'w') as f:
        json.dump(best_params, f, indent=4) 
    print(f"💾 Saved best params for {model_name} to {save_path}\n")
    print('=' * 60, '\n')

# from train.py
def load_best_params(model_name, load_dir='../tuned_params'):
    '''
    Loads in best params for a given model (model name in file).
    Makes it so grand_tuner() only has to be used once.

    Sample Usage:
        best_rf_params = load_best_params('RandomForestClassifier')
        models = {
            'RandomForest': (RandomForestClassifier, best_rf_params)
        }

        trained_models = train_model(X_train_full, y_train_full, models)
    '''
    load_path = os.path.join(load_dir, f'{model_name}_best_params.json')
    with open(load_path, 'r') as f:
        return json.load(f)
    
# from eval.py
def serialize_params(params):
    relevant_params = {}
    for k, v in params.items():
        try:
            json.dumps(v)
            relevant_params[k] = v
        except (TypeError, OverflowError, ValueError):
            # Catch more exceptions including nan
            if isinstance(v, float) and np.isnan(v):
                relevant_params[k] = 'NaN'
            else:
                relevant_params[k] = str(v)
    return relevant_params


# from tune.py
def stratified_sample(X, y, sample_frac=0.1, random_state=10, min_samples_per_class=5, verbose=True):
    '''
    Stratified sample of X, y for tuning

    Args: 
        X: full feature set
        y: full labels
        sample_frac: fraction of data used to sample (10%)
        random_state: reproducibility seed (10) 

    Returns: 
        X_sample, y_sample
    '''

    # Class distribution in full data
    class_counts = pd.Series(y).value_counts()
    num_classes = len(class_counts) 

    # Estimate sample size 
    expected_sample_size = int(len(X) * sample_frac) 

    # Estimate per-class sample counts: 
    estimated_per_class = expected_sample_size / num_classes

    # Safeguard: Too small to support all classes
    if estimated_per_class < min_samples_per_class:
        raise ValueError(
            f"Sample too small! Estimated ~{estimated_per_class:.2f} samples per class, "
            f"but min required is {min_samples_per_class}. "
            f"Increase sample_frac (currently {sample_frac})."
        )
    
    # If we're good, actually sample:
    X_sample, _, y_sample, _ = train_test_split(X, y, train_size=sample_frac, stratify=y, random_state=random_state)
    
    # Show class balance after sampling
    sampled_class_counts = pd.Series(y_sample).value_counts()

    if verbose:
        print(f"🔍 Sampled {len(X_sample)} rows ({sample_frac*100:.1f}%)")
        print(f"Class distribution:\n{sampled_class_counts.to_string()}")
        print("="*60,'\n')
    
    return X_sample, y_sample


# from tune.py
param_spaces = {
    "RandomForestClassifier": {
        "classifier__n_estimators": [50, 100, 200, 300, 500],
        "classifier__max_depth": [None, 10, 20, 30, 50],
        "classifier__min_samples_split": [2, 5, 10, 20],
        "classifier__min_samples_leaf": [1, 2, 5, 10],
        "classifier__max_features": ["sqrt", "log2", None]
    },
    "XGBClassifier": {
        "classifier__n_estimators": [50, 100, 200, 300],
        "classifier__max_depth": [3, 6, 9, 12],
        "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "classifier__subsample": [0.5, 0.7, 1.0],
        "classifier__colsample_bytree": [0.5, 0.7, 1.0]
    },
    "LogisticRegression": {
        "classifier__C": [0.01, 0.1, 1, 10, 100],
        "classifier__penalty": ["l1", "l2"],
        "classifier__solver": ["liblinear", "saga"]
    }, 
    "LGBMClassifier": {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [5, 10, 15, -1],
        'classifier__learning_rate': [0.01, 0.05, 0.1],
        'classifier__num_leaves': [31, 63, 127]
    }
}

def dynamic_param_grid(model, best_params):
    '''
    Refines hyperparameter search speace for GridSearchCV, based on model type
    
    Args:
        model: trained model object (RandomForest, XGBoost, LinearRegression)
        best_params: best paramters found from RandomizedSearchCV
        
    Returns:
        redefined_param_grid: param grid that is dependent on the `model`
    '''

    model_name = model.__class__.__name__
    if model_name not in param_spaces:
        raise ValueError(f"Model {model_name} is not yet supported by the `grand_tuner`.")
    
    refined_grid = {}

    try:
        # RF -- n_estimators and max_depth
        if model_name == "RandomForestClassifier":
            refined_grid = {
                "classifier__n_estimators": [
                    max(best_params["classifier__n_estimators"] - 50, 50),  
                    best_params["classifier__n_estimators"],  
                    best_params["classifier__n_estimators"] + 50
                ],
                "classifier__max_depth": [
                    best_params["classifier__max_depth"] - 10 if best_params["classifier__max_depth"] else None,
                    best_params["classifier__max_depth"],
                    best_params["classifier__max_depth"] + 10 if best_params["classifier__max_depth"] else None
                ]
            }
        
        # XGB -- n_estimators, learning_rate
        elif model_name == "XGBClassifier":
            refined_grid = {
                "classifier__n_estimators": [
                    max(best_params["classifier__n_estimators"] - 50, 50),  
                    best_params["classifier__n_estimators"],  
                    best_params["classifier__n_estimators"] + 50
                ],
                "classifier__learning_rate": [
                    max(best_params["classifier__learning_rate"] - 0.01, 0.01),
                    best_params["classifier__learning_rate"],
                    min(best_params["classifier__learning_rate"] + 0.01, 0.5)
                ]
            }

        # LR -- C, penalty
        elif model_name == "LogisticRegression":
            refined_grid = {
                "classifier__C": [
                    max(best_params["classifier__C"] / 10, 0.001),
                    best_params["classifier__C"],
                    min(best_params["classifier__C"] * 10, 1000)
                ],
                "classifier__penalty": [best_params["classifier__penalty"]] 
            }

        # LGBM -- n_estimators, max_depth, learning_rate, num_leaves
        elif model_name == "LGBMClassifier":
            refined_grid = {
                "classifier__n_estimators": [
                    max(best_params["classifier__n_estimators"] - 100, 50),
                    best_params["classifier__n_estimators"],
                    best_params["classifier__n_estimators"] + 100
                ],
                "classifier__max_depth": [
                    best_params["classifier__max_depth"] - 5 if best_params["classifier__max_depth"] not in [None, -1] else -1,
                    best_params["classifier__max_depth"],
                    best_params["classifier__max_depth"] + 5 if best_params["classifier__max_depth"] not in [None, -1] else -1
                ],
                "classifier__learning_rate": [
                    round(max(best_params["classifier__learning_rate"] - 0.01, 0.01), 3),
                    best_params["classifier__learning_rate"],
                    round(min(best_params["classifier__learning_rate"] + 0.01, 0.3), 3)
                ],
                "classifier__num_leaves": [
                    max(best_params["classifier__num_leaves"] - 16, 8),
                    best_params["classifier__num_leaves"],
                    best_params["classifier__num_leaves"] + 16
                ]
            }
    
    # If all else fails, revert to param_spaces (default) 
    except KeyError as e:
        print(f"⚠️ Missing expected param in best_params: {e}")
        print("Using default grid for fallback.")

        refined_grid = param_spaces[model_name]

    return refined_grid

# for train.py improvement
def strip_classifier_prefix(params_dict, prefix='classifier__'):
    '''
    Strips prefix "clasifier__" from param keys for raw model instantiation. 
    '''
    return {
        k.replace(prefix, '') if k.startswith(prefix) else k: v
        for k, v in params_dict.items()
    }

def detect_class_imbalance(y, threshold=0.15):
    '''
    Detects class imbalance, up to a toggleable percentage (default 15-85 split)
    04/7 -- Now supports both string and int target labels

    Args: 
        y (array-like): Class labels (str or int) 
        threshold (float): Minority class ratio threshold

    Returns: 
        is_imbalanced (bool): Whether the dataset is imbalanced 
        minority_ratio (float): Ratio of least-represented class
    '''

    if not np.issubdtype(np.array(y).dtype, np.integer):
        le = LabelEncoder()
        y = le.fit_transform(y) 

    counts = np.bincount(y)
    class_ratios = counts / counts.sum()
    minority_ratio = min(class_ratios)
    is_imbalanced = minority_ratio < threshold

    return  is_imbalanced, minority_ratio

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
        plt.axvline(best_thresh, color='red', linestyle='--', label=f'Best Threshold = {find_best_threshold:.2f}')
        plt.xlabel('Threshold') 
        plt.ylabel(f'{metric.upper()} Score')
        plt.title(f'{metric.upper()} vs Threshold')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.show() 

    return best_thresh

# EDA-- prevents data leakage
def map_target_column(df, target_col, positive='Yes', negative='No'):
    '''
    Maps string labels to binary (1 for positive, 0 for negative).
    Handles capitalization and leading/trailing whitespace.
    '''
    return df[target_col].str.strip().str.lower().map({
        positive.lower(): 1,
        negative.lower(): 0
    })

# Eval-- model calibration
def calibrate_model(model, X_val, y_val, method='sigmoid', cv=None):
    '''
    Calibrates a classifier's predicted probs using CalibratedClassifierCV

    Args: 
        model: already fitted classifier 
        X_calib: feats for calibration
        y_calib: target lables for calibration
        method: calibration method ('sigmoid' or 'isotonic')
        cv = number of cv folds or None (means prefit) 

    Returns: 
        calibrated_model: Calibrated classifier with predict_proba support
    '''

    if cv == 'prefit':
        calibrated_model = CalibratedClassifierCV(estimator=model, method=method, cv=None)
        calibrated_model.fit(X_val, y_val)
    else: 
        calibrated_model = CalibratedClassifierCV(estimator=model, method=method, cv=5)
        calibrated_model.fit(X_val, y_val)
    
    print(f"✅ Model calibrated using {method} method")
    return calibrated_model

# Eval-- more model calibration
def compare_probs(model, calibrated_model, X_val, y_val): 
    '''
    Compares predicted probs before and after model calibration

    Also displays predicted prob dist from before and after calibration
    '''
    # First get probs
    pre_probs = model.predict_proba(X_val)[:,1]
    post_probs = calibrate_model.predict_proba(X_val)[:,1]

    # Summaries
    print("\n📊 Probability Summary — Before Calibration:")
    print(pd.Series(pre_probs).describe())

    print("\n📊 Probability Summary — After Calibration:")
    print(pd.Series(post_probs).describe())

    print(f"\n🔍 Log Loss Before: {log_loss(y_val, pre_probs):.4f}")
    print(f"🔍 Log Loss After : {log_loss(y_val, post_probs):.4f}")

    # Plot the dists
    plt.hist(pre_probs, bins=30, alpha=0.5, label='Before Calibration')
    plt.hist(post_probs, bins=30, alpha=0.5, label='After Calibration')
    plt.title('Predicted Probability Distribution (Class 1)')
    plt.xlabel('Probability')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()


# EEG_FILE = resolve_path('data', 'eeg_hypno.csv')


from sklearn.metrics import confusion_matrix
from collections import Counter
import pandas as pd
from pathlib import Path
from src.logger_setup import logger

import joblib

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

from src.paths import ENCODER_DIR, REPORT_DIR

# loso_lstm --- refactor
def encode_labels(df_train, df_test, label_col, subject_id, save_dir=ENCODER_DIR):
    le = LabelEncoder()
    df_train[label_col] = le.fit_transform(df_train[label_col])
    df_test[label_col] = le.transform(df_test[label_col])
    
    save_dir.mkdir(parents=True, exist_ok=True)
    encoder_path = save_dir / f'le_s{subject_id}.pkl'
    joblib.dump(le, encoder_path) 

    with open(save_dir / f'classes_s{subject_id}.txt', 'w') as f:
        for i, label in enumerate(le.classes_):
            f.write(f'{i} = {label}\n')

    return df_train, df_test, le, encoder_path

# loso_lstm --- refactor 
from src.datasets.sequence_dataset import LSTMDataset
from torch.utils.data import DataLoader

def build_dataloaders(df_train, df_test, feature_cols, label_col, window_size, stride, batch_size):
    train_ds = LSTMDataset(df_train, feature_cols, label_col, window_size, stride)
    test_ds = LSTMDataset(df_test, feature_cols, label_col, window_size, stride)

    return {
        'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        'val': DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    }


# Directly interfaces with src/tune-- optuna_tuner stuff:
from datetime import datetime

def write_study_summary(study, subject=None, out_dir=None, top_n=5): 
    '''
    Writes a clean markdown summary of an Optuna study
        - Best trial
        - Top Trial Table (TTT)
        - Paths to the saved viz 
    
    Args: 
        study (optuna): Copmleted Optuna study object-- optuna CLI runner (for more info)
        stubject (int): Optional subject ID for clarity in study
        out_dir (str, Path): Optional folder to save the .md file 
        top_n (int): number of top trials to include in the TTT 
    '''
    timestamp = datetime.now().strftime('%Y-%m-%d') 
    name = f'study_{subject}_{timestamp}' if subject else f'study_{timestamp}'

    report_path = Path(out_dir) if out_dir else REPORT_DIR / 'optuna' / name
    report_path.mkdir(parents=True, exist_ok=True) 

    df = study.trials_dataframe()
    df_sorted = df.sort_values('value', ascending=False).reset_index(drop=True)

    best = df_sorted.iloc[0]
    best_params = {
        k.replace('params', ''): v
        for k, v in best.items()
        if k.startswith('params_')
    }
    
    markdown = [] 

    # --- Header ---
    markdown.append(f'#Optuna LSTM Tuning Summary')
    markdown.append(f'Date: {timestamp}') 
    if subject: 
        markdown.append(f'Subject: {subject}')
    markdown.append(f'Trials: {len(df)}') 
    markdown.append(f'Objective: Maximize F1 Score') 
    markdown.append('\n---\n')

    # --- Best Trial --- 
    markdown.append(f'## Best Trial') 
    markdown.append(f'- **F1 Score**: {best['value']:.4f}')
    markdown.append(f'- **Threshold**: {best.get('user_attrs_best_thresh', 'N/A')}')
    markdown.append(f'- **Accuracy**: {best.get('user_attrs_accuracy', 'N/A')}')
    markdown.append(f'- **Params**')
    for k, v in best_params.items():
        markdown.append(f"  - `{k}`: {v}")
    markdown.append("\n---\n")

        # --- Top Trials Table ---
    markdown.append(f"## Top {top_n} Trials")
    markdown.append(f"| Trial | F1 Score | Threshold | Accuracy |")
    markdown.append(f"|-------|----------|-----------|----------|")


    for i, row in df_sorted.head(top_n).iterrows():
        f1 = row["value"]
        thresh = row.get("user_attrs_best_thresh", "N/A")
        acc = row.get("user_attrs_accuracy", "N/A")
        markdown.append(f"| {row['number']} | {f1:.4f} | {thresh} | {acc} |")

    markdown.append("\n---\n")
    
    # --- Visualizations ---
    markdown.append("## Visualizations")
    markdown.append("- `f1_importance_barplot.png`")
    markdown.append("- `corr_heatmap.png`")
    markdown.append("\n---\n")

    # --- Notes ---
    markdown.append(f'## Notes')

    # --- Write to file --- 
    summary_path = out_dir / f'optuna_summary_{timestamp}.md'
    summary_path.write_text('\n'.join(markdown))

    print(f'✅ Markdown summary saved to: {summary_path.resolve()}')