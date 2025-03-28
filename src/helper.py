import os 
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

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
    '''
    counts = np.bincount(y)
    class_ratios = counts / counts.sum()
    minority_ratio = min(class_ratios)
    return minority_ratio < threshold, minority_ratio

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
def map_target_column(df, target_col, positive, negative): 
    '''
    Converts a binary categorical target col to 1 (pos) and 0 (neg) 

    Args:
        df (pd.DataFrame): DataFrame containing the target column.
        target_col (str): Name of the column to map.
        positive (str): Value to map to 1.
        negative (str): Value to map to 0.
        
    Returns:
        pd.Series: Mapped binary target series.
    '''
    return df[target_col].map({positive:1, negative:0})
