import optuna
from src.tune import grand_tuner, optuna_tuner
from src.train import train_model
from src.helper import detect_class_imbalance, stratified_sample, detect_class_imbalance
from src.eval import eval_classification
from src.paths import MODEL_DIR

import joblib 
from sklearn.preprocessing import LabelEncoder
import numpy as np
import random
import pickle
import json

from .paths import MODEL_DIR, TUNED_PARAMS_DIR

np.random.seed(10)
random.seed(10)

def tune_and_train_full(model_class, model_name, X_train, y_train,
                        sample_frac=0.1, model_params=None,
                        X_test=None, y_test=None, dev_mode=False,
                        scoring='f1_weighted', use_scaler = True, 
                        n_trials=50, cv=5, verbose=True,
                        **tuner_kwargs): 
    
    """
    Full workflow:
    1. Sample data
    2. Tune on sample
    3. Save best params
    4. Train final model on full data

    Args:
        model_class: The estimator class (e.g., RandomForestClassifier)
        model_name: Name used for saving and reporting
        X_train: Training features
        y_train: Training labels
        sample_frac: Fraction of data to use for tuning (default 0.1)
        model_params: Initial model parameters (if any)
        X_test: Optional test features for final eval
        y_test: Optional test labels for final eval
        dev_mode: If True, trains on the sample instead of full set
        scoring: Metric used during tuning ('accuracy', 'f1_weighted', etc.)
        **tuner_kwargs: Additional parameters passed to tuning functions

    Returns: 
        trained_model: optimized, trained and ready to predict on new data model
        best_params: fresh results from grand_tuner's efforts (sent to json as well)
    """
    
    dev_mode = tuner_kwargs.pop("dev_mode", dev_mode)
    label_encoder = tuner_kwargs.pop("label_encoder", None)  # Extract before grand_tuner()! 

    if scoring == 'roc_auc' and len(np.unique(y_train)) > 2: 
        raise ValueError("'roc_auc' is only valid for binary classification! Use 'accuracy' or 'weighted_f1' for multiclass")

    X_sample, y_sample = stratified_sample(X_train, y_train, sample_frac=sample_frac)

    # Instantiate model
    base_model = model_class(**(model_params or {}))


    # INITIATE OPTUNA
    best_model, best_params, _ = optuna_tuner(
        model_class=model_class, 
        X=X_sample,
        y=y_sample,
        scoring=scoring,
        n_trials=n_trials,
        cv=cv, 
        verbose=verbose, 
        study_name=model_name,
    )

    # Load and train on full data
    # Merge custom added model_params + best_params
    merged_params = (model_params or {}).copy()
    merged_params.update(best_params)

    # DEV_MODE ONLY
    X_train_final = X_sample if dev_mode else X_train
    y_train_final = y_sample if dev_mode else y_train

    trained_models = train_model(X_train_final, y_train_final, {
        model_name: (model_class, merged_params, use_scaler)
    })

    trained_model = trained_models[model_name]

    # === Save trained model to pkl ===
    basename = model_name.lower().replace('classifier', '')
    basename = basename.replace('_', '-').replace(' ', '-')
    filename = f'{basename}.pkl' 
    save_path = MODEL_DIR / filename 

    joblib.dump(trained_model, save_path) 
    
    if verbose: 
        print(f'✅ Saved to {save_path}')

    if X_test is not None and y_test is not None:
        print(f'\n Running evaluation on test set...')

        is_imbalanced, minority_ratio = detect_class_imbalance(y_test)
        threshold = 0.25 if is_imbalanced else 0.5

        if is_imbalanced:
            print(f"⚠️ Imbalanced test set detected (Minority class = {minority_ratio:.2%})")
        else:
            print(f"Balanced test set detected, using default threshold: {threshold}")

        final_metrics = eval_classification(trained_model, 
                            X_test, y_test, 
                            threshold,
                            label_encoder=label_encoder
        )
    else: 
        print("⚠️ No test set provided, skipping evaluation.\n")

    

    return trained_model, best_params, final_metrics

# Quick Load-- trying to cut away stupid-long wait times! 
def quick_load(model_class, model_name, X_train, y_train, force_retrain=False):
    '''
    Loads a trained model from .pkl if it exists, or trains from the JSON best params
    If both are missing *or* retrain is forced, then it trains from scratch via `train_model`

    Returns: 
        trained_model (sklearn-like): Model is ready for prediction! 
    '''

    model_path = MODEL_DIR / '{model_name}.pkl'
    params_path = TUNED_PARAMS_DIR / '{model_name}_best_params.json'

    if model_path.exists() and not force_retrain: 
        print(f'📦 Loading existing model from {model_path}')
        with open(model_path, 'rb') as f:
            return pickle.load(f) 
        
    if params_path.exists(): 
        print(f'⚙️ Training model using best params from {params_path}') 
        with open(params_path, 'r') as f:
            best_params = json.load(f)

        trained_models = train_model(X_train, y_train, {
            model_name: (model_class, best_params)
        }) 

        model = trained_models[model_name] 

        with open(model_path, 'wb') as f:
            pickle.dump(model, f) 
            print(f'💾 Saved trained model to {model_path}')

        return model
    
    raise FileNotFoundError('No model .pkl or param .json found-- please run full pipeline first!')
