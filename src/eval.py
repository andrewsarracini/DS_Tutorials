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

from src.helper import serialize_params
from src.logger_setup import logger

def eval_classification(model, X_test, y_test, threshold=0.5):
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
    if hasattr(model, "predict_proba"):
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
        model_params = "Parameters not available"
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
    print(pd.DataFrame({
        "Class": list(report.keys())[:-3],  # Exclude avg/total rows
        "Precision": [report[k]["precision"] for k in report.keys() if k.isdigit()],
        "Recall": [report[k]["recall"] for k in report.keys() if k.isdigit()],
        "F1-Score": [report[k]["f1-score"] for k in report.keys() if k.isdigit()],
        "Support": [report[k]["support"] for k in report.keys() if k.isdigit()]
    }).round(4).to_markdown(index=False))
    
    print("\nConfusion Matrix:")
    print(cm_df.to_markdown(index=True))


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
    ax1.set_ylabel('True Positive Raet') 
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
        print(f'Plot saved to {save_path}')
    
    plt.show() 
    