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


def train_model(X_train, y_train, models, save_dir='../models'):
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

    os.makedirs(save_dir, exist_ok=True)

    print('Training Models...')
    print('=============================') 

    for model_name, (model_class, model_params) in models.items():
        print(f'Training {model_name}({model_class.__name__}) with params: {model_params or 'default settings'}...') 

        # Initialize model with params
        model = model_class(**(model_params or {})) 

        # Handle XGBoost separately — no pipeline
        if model_name == 'XGBoost':
            model.fit(X_train, y_train)
            trained_model = model  # Direct model, no pipeline
        else:
            trained_model = Pipeline([
                ('scaler', StandardScaler()), 
                ('model', model) 
            ])
            trained_model.fit(X_train, y_train)

        # Actually training the model
        trained_model.fit(X_train, y_train)

        # Trying out joblib! 
        # Save the trained pipeline
        save_path = os.path.join(save_dir, f'{model_name}.pkl')
        joblib.dump(trained_model, save_path) 

        print(f"✅ {model_name} training complete! Model saved to {save_path}\n")

        trained_models[model_name] = trained_model

    return trained_models