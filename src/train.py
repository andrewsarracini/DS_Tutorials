from random import Random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

# Hyperparm Tuning:
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV # Bayseian Optimization
from tpot import TPOTClassifier # Automated ML, Classification Tasks

def extract_target(df: pd.DataFrame, target: str): 
    '''
    Splits the dataset into features (X) and target (y)
    
    df -- input Dataframe
    target -- Name of the target variable column
    '''
    X = df.drop(columns=[target])
    y = df[target]
    return X, y

def train_model(X_train, y_train, models):
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

    Through this, pass in any model and corresponding params (as a result of hyperparam tuning) 

    Returns:
        dict: trained models, names as keys and fitted pipelines as values

    --------------------------------------------------
    Exapmle Usage:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

    models = {
        'RandomForest': (RandomForestClassifier, {'n_estimators':100, random_state=10}),
        'LogisticReg': (LogisticRegression, {'C':0.1})
    }

    trained_models = train_model(X_train, y_train, models)

    # Accessing the trained model:
    rf_model = trained_models['RandomForest']
    lr_model = trained_models['LogisticReg']
    '''

    trained_models = {}

    for model_name, (model_class, model_params) in models.items():
        model = model_class(**(model_params or {})) 

        trained_model = Pipeline([
            ('scaler', StandardScaler()), 
            ('model', model) 
        ])

        # Actually training the model
        trained_model.fit(X_train, y_train)
        print(f'{model.__name__} training complete!')

        # Trying out joblib! 
        # Save the trained pipeline
        joblib.dump(trained_model, f'../models/{model_name}.pkl')

        trained_models[model_name] = trained_model

    return trained_models

# Hyperparameter Tuning:
def grid_search(model, param_grid, X_train, y_train, cv=5, scoring='accuracy'):
    ''' 
    Performs exhaustive grid search hyperparam tuning
    '''
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=-1) 
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_

def random_search_tuning(model, param_dist, X_train, y_train, cv=5, n_iter=50, scoring='accuracy'):
    ''' 
    Performs random search hyperparam tuning
    '''
    random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=n_iter, cv=cv, scoring=scoring, n_jobs=-1)
    random_search.fit(X_train, y_train)
    return random_search.best_estimator_, random_search.best_params_

def bayesian_opt_tuning(model, search_spaces, X_train, y_train, cv=5, n_iter=50, scoring='accuracy'):
    ''' 
    Performs Bayesian optimization param tuning
    '''
    bayes_search = BayesSearchCV(model, search_spaces=search_spaces, n_iter=n_iter, cv=cv, scoring=scoring, n_jobs=-1)
    bayes_search.fit(X_train, y_train)
    return bayes_search.get_params(deep=True)

#------------------------------------------------
# This will take hours/days to run, USE WITH CAUTION!!! 
def tpot_opt_tuning(X_train, y_train, X_test, y_test, generations=5, population_size=20, cv=5):
    '''
    Uses TPOT to find the best params
    '''
    tpot = TPOTClassifier(generations=generations, population_size=population_size, cv=cv, verbosity=2, n_jobs=-1)
    tpot.fit(X_train, y_train)

    # BE AWARE: this generates a brand new python file that will have optimal params!
    tpot.export('tpot_pipeline.py') 
    return tpot.score(X_test, y_test)

