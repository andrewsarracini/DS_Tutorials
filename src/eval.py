import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def eval_classification(model, X_test, y_test):
    '''
    Evaluates a classification model and prints performance metrics

    Params: 
    model-- trained model 
    X_test-- test features
    y_test-- true labels

    Returns:
    dict: Dictionary containing eval metrics
    '''

    y_pred = model.predict(X_test)

    metrics = {
        'accuracy' : accuracy_score(y_test, y_pred),
        'precision' : precision_score(y_test, y_pred, average='weighted', zero_division=0), 
        'recall' : recall_score(y_test, y_pred, average='weighted', zero_division=0), 
        'f1_score' : f1_score(y_test, y_pred, average='weighted', zero_division=0), 
        'confusion_matrix' : confusion_matrix(y_test, y_pred).tolist(), # -- for for better readability!
        'classification_rep' : classification_report(y_test, y_pred, output_dict=True) 
    }

    print(f'\nEvaluation Metrics: ') 
    for key, value in metrics.items():
        if key not in ['confusion_matrix', 'classification_rep']:
            print(f'{key.capitalize()}: {value:.4f}')

    print('\nConfusion Matrix') 
    print(pd.DataFrame(metrics['confusion_matrix']))
    return metrics

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