import os
import pandas as pd
from sklearn import pipeline
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import make_scorer, f1_score
from sklearn.preprocessing import StandardScaler
import joblib
import os 
import json

from src.logger_setup import logger
from src.helper import strip_classifier_prefix

def train_model(X_train, y_train, models, verbose=True):
    '''
    Trains one or multiple models using an sklearn pipeline.
    *** Requires train/test split to occur beforehand 

    X_train -- features
    y_train -- target
    models (dict)-- Dictionary where keys are model names (str) and values are tuples (model_class, optional model params)
            Ex: 
        {
            'RandomForest': {RandomForestClassifier, {n_estimators:100}),
            'LogisticReg': {LogisticRegression, {'C':0.1})
        } 
    save_dir (str)-- Directory to save trained models (defauly is '../models') 


    *** Through this, pass in any model and corresponding params (as a result of hyperparam tuning) 

    Returns:
        dict: Trained models, names as keys and fitted pipelines as values

    --------------------------------------------------
    Exapmle Usage:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

    models = {
        'RandomForest': (RandomForestClassifier, {'class_weight':'balanced'}),
        'XGBoost': (XGBClassifier, None)
    }

    trained_models = train_model(X_train, y_train, models)

    # Accessing the trained model:
    rf_model = trained_models['RandomForest']
    lr_model = trained_models['LogisticReg']
    '''

    trained_models = {}

    if verbose:
        print(f"[TRAINING] Starting model training...\n")

    for model_name, model_tuple in models.items():
        if len(model_tuple) == 3: 
            model_class, model_params, use_scaler = model_tuple
        else: 
            model_class, model_params = model_tuple
            use_scaler = True # default to scaling unless specified! 
        
        if verbose:
            print(f"→ Training {model_name} with params:")
            print(json.dumps(model_params, indent=4))
            print(f'Scaling: {'ON' if use_scaler else 'OFF'}') 

        # Clean params for model init
        clean_params = strip_classifier_prefix(model_params or {}) 

        # Initialize model with params
        model = model_class(**clean_params)

        # Handle XGBoost separately — no pipeline
        if model_name == 'XGBoost':
            model.fit(X_train, y_train)
            trained_model = model  # Direct model, no pipeline
        else:
            steps = []
            if use_scaler: 
                steps.append(('scaler', StandardScaler())) 
            steps.append(('model', model)) 
            
            trained_model = Pipeline(steps) 
            trained_model.fit(X_train, y_train)

        if verbose:
            # print(f"\n✅ {model_name} trained | Saved to {save_path}")
            print(f'\n✅ {model_name} Trained!')
            print("="*60)

        trained_models[model_name] = trained_model

    return trained_models