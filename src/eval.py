import pandas as pd
import numpy as np
import logging
from logging.handlers import RotatingFileHandler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, precision_recall_curve, roc_curve, auc
)
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib 
matplotlib.use('Agg')
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
import seaborn as sns


from src.helper import serialize_params
from src.logger_setup import logger

def eval_classification(model, X_test, y_test, threshold=0.5,
                         label_encoder=None, verbose=True):
    '''
    Evaluates a classification model and prints/logs performance metrics.
    '''

    is_pipeline = hasattr(model, 'named_steps')  # True if model is a pipeline

    # Get the classifier for parameter extraction
    if is_pipeline:
        classifier = model.named_steps['model']
        model_name = type(classifier).__name__
    else:
        classifier = model
        model_name = type(model).__name__

    # If threshold param exists, use probability-based thresholding
    # Second check ensures binary classification-- otherwise SKIP thresholding
    if hasattr(model, "predict_proba") and len(np.unique(y_test)) == 2: 
        try:
            y_probs = model.predict_proba(X_test)[:, 1]
            y_pred = (y_probs >= threshold).astype(int)

            if threshold != 0.5:
                print(f"\n🔧 Custom threshold applied: {threshold}")
                    
        except Exception as e:
            print(f"\n⚠️ Error using threshold: {e}")
            y_pred = model.predict(X_test)
    else:
        y_pred = model.predict(X_test)

    # Decode if label encoder is provided and labels are int-encoded
    if label_encoder is not None:
        try:
            if isinstance(y_test[0], str):
                if verbose:
                    print("⏭️ Label decoding skipped — y_test appears to already be decoded.")
            else:
                if verbose:
                    print(f"\nLabel decoder active:")
                    print("  y_test unique:", np.unique(y_test))
                    print("  Classes:", label_encoder.classes_)
                y_test = label_encoder.inverse_transform(y_test)
                y_pred = label_encoder.inverse_transform(y_pred)
        except Exception as e:
            if verbose:
                print(f"⚠️ Label decoding failed: {e}")

    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=[f"Actual {i}" for i in range(len(cm))], columns=[f"Predicted {i}" for i in range(len(cm))])

    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)

    try:
        raw_params = classifier.get_params()
        model_params = serialize_params(raw_params)
    except Exception as e:
        model_params = {}
        logger.error(f"Error retrieving model parameters: {e}")

    log_message = (
        "\n" + "-"*80 + "\n"
        f"Model: {model_name}\n"
        f"Hyperparameters: {json.dumps(model_params, indent=4)}\n\n"
        "Overall Metrics:\n"
        f"Accuracy: {accuracy:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall: {recall:.4f}\n"
        f"F1-score: {f1:.4f}\n\n"
        "Confusion Matrix:\n"
        f"{cm_df.to_string()}\n"
        + "-"*80 + "\n"
    )

    try:
        logger.info(log_message)
    except Exception as e:
        logger.error(f"Error writing to log file: {e}")
        print(log_message)  # Print to console as a fallback

    if verbose:
        # Print results for readability
        print(f"\nEvaluating Model: {model_name}")
        print("Hyperparameters:")
        print(json.dumps(model_params, indent=4))  
        
        print("\nOverall Metrics:")
        print(pd.DataFrame({
            "Metric": ["Accuracy", "Weighted Precision", "Weighted Recall", "Weighted F1-Score"],
            "Score": [accuracy, precision, recall, f1]
        }).round(4).to_markdown(index=False))

        print("\nClass-Specific Metrics:")
        class_keys = sorted([k for k in report.keys() if k not in ('accuracy', 'macro avg', 'weighted avg')])
        print(pd.DataFrame({
            "Class": class_keys,
            "Precision": [report[k]["precision"] for k in class_keys],
            "Recall": [report[k]["recall"] for k in class_keys],
            "F1-Score": [report[k]["f1-score"] for k in class_keys],
            "Support": [report[k]["support"] for k in class_keys]
        }).round(4).to_markdown(index=False))

    per_class_f1 = {
        label: report[label]['f1-score']
        for label in report 
        if label not in ('accuracy', 'macro avg', 'weighted avg')
    }

    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "weighted_f1": f1,
        "model_params": model_params, 
        "confusion_matrix": cm_df, 
        "report": report,
        'per_class_f1': per_class_f1,
        }

def eval_regression(model, X_test, y_test):
    '''
    Evaluates a regression model and prints performance metrics 

    Params: 
    model-- trained model
    X_test-- test features
    y_test-- true labels

    Returns: 
    dict: Dictionary containing eval metrics
    '''

    y_pred = model.predict(X_test, y_test) 

    metrics = {
        'MAE' : np.mean(abs(y_test - y_pred)), 
        'MSE' : np.mean((y_test - y_pred) **2),
        'RMSE' : np.sqrt(metrics['MSE']),
        'R2' : model.score(X_test, y_test) 
    }

    print('\nEval Metrics')
    for key, value in metrics.items():
        print(f'f{key}: {value:.4f}')

    return metrics

# Potting the Threshold Curves
def plot_threshold_curves(y_true, y_probs, model_name='Model', highlight_threshold=0.25, save_path=None):
    '''
    Plots the ROC and Precision-Recall curves with optional annotations

    Args: 
        y_true: Ground truth binary labels
        y_probs: Predicted probs of the positive class
        model_name: Name of the model
        highlight_threshold: Optional thresh to annotate on both curves
        save_path: Optional filepath to save the plots
    '''
    # Roc Data
    # fpr -- False Positive Rate 
    # tpr -- True Positive Rate
    # roc_thresholds -- differing threshs used to generated fpr, tpr pairs :D 

    # auc(fpr, tpr) calcs total area under the ROC curve.
    # This is the model's ability to discriminate between classes

    # Reminder: Area Under Curve (AUC) 
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr) 

    # PR Data
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall, precision)

    # Find closest index to highlight thresh
    idx_roc = np.argmin(np.abs(roc_thresholds - highlight_threshold))
    idx_pr = np.argmin(np.abs(pr_thresholds - highlight_threshold)) 

    # Plot
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,5))

    # ROC Curve
    ax1.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    ax1.plot([0,1], [0,1], linestyle='--', color='gray') # Represents random classification (FPR = TPR)
    ax1.scatter(fpr[idx_roc], tpr[idx_roc], color='red', label=f'Thresh = {highlight_threshold}')
    ax1.set_title(f'ROC Curve - {model_name}')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate') 
    ax1.legend()

    ax2.plot(recall, precision, label=f'AUC = {pr_auc:.2f}') 
    ax2.scatter(recall[idx_pr], precision[idx_pr], color='red', label=f'Thresh = {highlight_threshold}')
    ax2.set_title(f'Precision-Recall Curve - {model_name}')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision') 
    ax2.legend()

    fig.suptitle(f"Threshold Analysis for {model_name}", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f'Plot saved to {save_path}\n')
    
    # REMOVING THIS FOR NOW
    # TKINTER YOU DID THIS!
    # plt.show() 
    
    # =============================
    # Calibration Funcs-- maybe overkill?

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

def eval_predictions(y_pred, y_true, label_encoder=None): 
    '''
    Evaluates raw predicted labels against true labels
    Optionally decodes if label encoder is provided

    Args: 
        y_pred (array-like): Predicted labels (str or int) 
        y_true (array-like): Ground truth labels
        label_encoder (LabelEncoder, optional): IF used during modeling, will decode y_pred and y_true
    '''

    if label_encoder: 
        if hasattr(y_pred[0], '__index__'): # encoded ints
            y_pred = label_encoder.inverse_transform(y_pred)
        if hasattr(y_true[0], '__index__'):
            y_true = label_encoder.inverse_transform(y_true) 

    print('Classification Report') 
    print(classification_report(y_true, y_pred, digits=3)) 

    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=np.unique(y_true), yticklabels=np.unique(y_true), cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title('Confusion Matrix')
    plt.show()

# to plug into loso_lstm
def find_best_thresh(y_true, y_probs, metric='f1', step=0.01): 
    '''
    Finds the best threshold that maximizes a specific metric 

    Args: 
        y_true: Ground truth labels
        y_probs: Model's predicted probabilities (after loss_fn "sigmoid")
        metric: One of 'f1', 'precision', 'recall'
        step: Incremental step for thresholds 
    
    Returns: 
        best_thresh (float): Thresh that delivers highest metric result
        best_score (float): Corresponding metric score
    '''
    metric = metric.lower()

    if metric == 'f1':
        metric = f1_score
    elif metric == 'acc':
        metric = accuracy_score
    elif metric == 'recall': 
        metric = recall_score
    else: 
        raise ValueError(f'Unsupported metric {metric}')

    best_thresh = 0.5
    best_score = 0.0

    thresholds = np.arange(0.05, 0.95, step) 

    for t in thresholds:
        preds = (y_probs > t).astype(int) 
        score = metric(y_true, preds, zero_division=0)

        if score > best_score:
            best_score = score
            best_thresh = t
    
    return round(best_thresh, 4), best_score

