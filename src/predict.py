import joblib
import pandas as pd
import numpy as np

def load_trained_model(model_path): 
    '''
    Loads a trained model from a joblib file!
    '''
    return joblib.load(model_path) 

def make_preds(model, X): 
    '''
    Makes predictions using a trained model
    
    Args: 
        model-- the trained model
        X-- feature data (np array or df) for prediction

    Returns: 
        np array of preds
    '''

    if isinstance(X, pd.DataFrame): 
        # converting any df to np array
        X = X.values 
    return model.predict(X)

def batch_pred(model, X_batch): 
    '''
    This function is useful for processing large datasets at once, optimizing performance
    especially for models that benefit from batch processing (e.g., neural networks or
    large-scale ML models). Keeping this function separate allows for future scalability,
    such as adding batch size control or parallel computation.
    
    
    Args: 
        model-- trained model
        X_batch-- feature data(np array or df) for pred
        
    Returns: 
        List of preds
    '''
    return make_preds(model, X_batch).tolist()

