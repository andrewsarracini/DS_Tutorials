import pandas as pd
import numpy as np
import logging
from logging.handlers import RotatingFileHandler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
)
import json
import os

def setup_logger():
    """
    Sets up a rotating log file in the logs directory at the base of the repo.
    """

    # Go up one level (to the repo base) and create the 'logs' folder there
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
    os.makedirs(log_dir, exist_ok=True)  # Ensure logs directory exists

    log_file = os.path.join(log_dir, "model_eval.log")

    handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=10)  # 5MB per file, keep 10 files max
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[handler]
    )
    return logging.getLogger("ModelEvaluation")

# Initialize logger
logger = setup_logger()

def serialize_params(params):
    '''Filters, cleans, and serializes model parameters for logging.'''
    relevant_params = {}
    for k, v in params.items():
        if v is None:
            continue
        try:
            json.dumps(v)  # Check if it's serializable
            relevant_params[k] = v
        except (TypeError, OverflowError):
            relevant_params[k] = str(v)  # Convert non-serializable objects to string
    return relevant_params

def eval_classification(model, X_test, y_test):
    '''
    Evaluates a classification model and prints/logs performance metrics.
    
    Params:
    - model: Trained model
    - X_test: Test features
    - y_test: True labels
    '''

    model_name = type(model.steps[-1][1]).__name__

    y_pred = model.predict(X_test)

    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])

    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    try:
        model_params = serialize_params(model.get_params())
    except Exception as e:
        model_params = "Parameters not available"
        logger.error(f"Error retrieving model parameters: {e}")

    log_entry = {
        "model_name": model_name,
        "hyperparameters": model_params,
        "metrics": {
            "accuracy": accuracy,
            "weighted_precision": precision,
            "weighted_recall": recall,
            "weighted_f1": f1
        },
        "confusion_matrix": cm.tolist()
    }

    try:
        logger.info(json.dumps(log_entry, indent=4))
    except Exception as e:
        logger.error(f"Error logging JSON: {e}")
        print(json.dumps(log_entry, indent=4))  # Print for debugging if logging fails

    # Print results for readability
    formatted_params = json.dumps(model_params, indent=4)

    # Print results for readability
    print(f"\nEvaluating Model: {model_name}")
    print("Hyperparameters:")
    print(formatted_params)  # Print formatted JSON
    
    print("\nOverall Metrics:")
    print(pd.DataFrame({
        "Metric": ["Accuracy", "Weighted Precision", "Weighted Recall", "Weighted F1-Score"],
        "Score": [accuracy, precision, recall, f1]
    }).round(4))

    print("\nClass-Specific Metrics:")
    print(pd.DataFrame({
        "Class": ["0", "1"],
        "Precision": [report["0"]["precision"], report["1"]["precision"]],
        "Recall": [report["0"]["recall"], report["1"]["recall"]],
        "F1-Score": [report["0"]["f1-score"], report["1"]["f1-score"]],
        "Support": [report["0"]["support"], report["1"]["support"]]
    }).round(4))
    
    print("\nConfusion Matrix:")
    print(cm_df)

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