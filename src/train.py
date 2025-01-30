import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

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

