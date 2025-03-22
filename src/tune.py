from imblearn.pipeline import Pipeline as imbpipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold, train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import time

import os
import json

# THIS IS INTENTIONALLY OUTSIDE
# Might make this global later, still deciding
param_spaces = {
    "RandomForestClassifier": {
        "classifier__n_estimators": [50, 100, 200, 300, 500],
        "classifier__max_depth": [None, 10, 20, 30, 50],
        "classifier__min_samples_split": [2, 5, 10, 20],
        "classifier__min_samples_leaf": [1, 2, 5, 10],
        "classifier__max_features": ["sqrt", "log2", None]
    },
    "XGBClassifier": {
        "classifier__n_estimators": [50, 100, 200, 300],
        "classifier__max_depth": [3, 6, 9, 12],
        "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "classifier__subsample": [0.5, 0.7, 1.0],
        "classifier__colsample_bytree": [0.5, 0.7, 1.0]
    },
    "LogisticRegression": {
        "classifier__C": [0.01, 0.1, 1, 10, 100],
        "classifier__penalty": ["l1", "l2"],
        "classifier__solver": ["liblinear", "saga"]
    }
}

def dynamic_param_grid(model, best_params):
    '''
    Refines hyperparameter search speace for GridSearchCV, based on model type
    
    Args:
        model: trained model object (RandomForest, XGBoost, LinearRegression)
        best_params: best paramters found from RandomizedSearchCV
        
    Returns:
        redefined_param_grid: param grid that is dependent on the `model`
    '''

    model_name = model.__class__.__name__
    if model_name not in param_spaces:
        raise ValueError(f"Model {model_name} is not yet supported by the `grand_tuner`.")
    
    refined_grid = {}

    try:
        # RF -- n_estimators and max_depth
        if model_name == "RandomForestClassifier":
            refined_grid = {
                "classifier__n_estimators": [
                    max(best_params["classifier__n_estimators"] - 50, 50),  
                    best_params["classifier__n_estimators"],  
                    best_params["classifier__n_estimators"] + 50
                ],
                "classifier__max_depth": [
                    best_params["classifier__max_depth"] - 10 if best_params["classifier__max_depth"] else None,
                    best_params["classifier__max_depth"],
                    best_params["classifier__max_depth"] + 10 if best_params["classifier__max_depth"] else None
                ]
            }
        
        # XGB -- n_estimators, learning_rate
        elif model_name == "XGBClassifier":
            refined_grid = {
                "classifier__n_estimators": [
                    max(best_params["classifier__n_estimators"] - 50, 50),  
                    best_params["classifier__n_estimators"],  
                    best_params["classifier__n_estimators"] + 50
                ],
                "classifier__learning_rate": [
                    max(best_params["classifier__learning_rate"] - 0.01, 0.01),
                    best_params["classifier__learning_rate"],
                    min(best_params["classifier__learning_rate"] + 0.01, 0.5)
                ]
            }

        # LR -- C, penalty
        elif model_name == "LogisticRegression":
            refined_grid = {
                "classifier__C": [
                    max(best_params["classifier__C"] / 10, 0.001),
                    best_params["classifier__C"],
                    min(best_params["classifier__C"] * 10, 1000)
                ],
                "classifier__penalty": [best_params["classifier__penalty"]] 
            }
    
    # If all else fails, revert to param_spaces (default) 
    except KeyError as e:
        print(f"⚠️ Missing expected param in best_params: {e}")
        print("Using default grid for fallback.")

        refined_grid = param_spaces[model_name]

    return refined_grid


def grand_tuner(model, param_grid, X, y, cv=5, scoring='roc_auc', use_smote=True, n_iter=20):
    '''
    Performs hyperparameter tuning using a two-step approach:
    1. RandomizedSearchCV to explore a broad parameter space
    2. GridSearchCV to fine-tune the best found region
    ** Saves best params to ../tuned_params

    Args:
        model: the classifier to be tuned
        param_grid: dictionary of hyperparameters
        X: features
        y: target labels
        cv: number of Cross-Validation folds
        scoring: evaluation metric (roc_auc)
        use_smote: whether to apply SMOTE for class-balancing
        n_iter: number of iterations for RandomizedSearchCV (20)

    Returns: 
        best_model: model with optimal hyperparams
        best_params: dictionary of best hyperparams
    '''

    print(f"\nStarting Grand Tuner with {cv}-fold Cross-Validation...")
    print(f"Model: {model.__class__.__name__}")
    print(f"Scoring metric: {scoring}")
    print(f"SMOTE Enabled: {use_smote}")
    print(f"Running RandomizedSearchCV with {n_iter} iterations...")

    if param_grid is None:
        param_grid = param_spaces.get(model.__class__.__name__, {})
        print(f"Using default param grid for {model.__class__.__name__}: {param_grid}")

    # Pipeline with optional SMOTE
    steps = []
    if use_smote:
        steps.append(('smote', SMOTE(random_state=10)))

    steps.extend([('scaler', MinMaxScaler()), ('classifier', model)])
    pipeline = imbpipeline(steps)

    # Cross-validation strategy
    stratified_kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=10)

    # Step 1: RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring=scoring,
        cv=stratified_kfold,
        n_jobs=-1,
        verbose=1,
        random_state=10
    )
   
    # Timing the Random Search
    start_random = time.time()
    random_search.fit(X, y)
    end_random = time.time()
    print(f"⏱️ RandomizedSearchCV completed in {(end_random - start_random)/60:.2f} minutes")


    # Get best parameters from RandomizedSearch
    best_random_params = random_search.best_params_
    print(f"\n🎲 Best Parameters from RandomizedSearch: {best_random_params}")

    # === Inject all best params into pipeline before GridSearch ===
    model_specific_params = {k: v for k, v in best_random_params.items() if k.startswith('classifier__')}
    pipeline.set_params(**model_specific_params)

    refined_grid = dynamic_param_grid(model, best_random_params)
    print(f"\n🛠️ Running GridSearchCV with refined parameters: {refined_grid}")

    # Step 2: GridSearchCV 
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=refined_grid, 
        scoring=scoring, 
        cv=stratified_kfold, 
        n_jobs=4, 
        verbose=1
    ) 

    # Timing the Grid Search
    start_grid = time.time()
    grid_search.fit(X, y)
    end_grid = time.time()
    print(f"⏱️ GridSearchCV completed in {(end_grid - start_grid)/60:.2f} minutes")


    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    cv_results = grid_search.cv_results_

    print("=" * 50)
    print(f"\n✅ Best Model Found: {best_model}")
    print(f"🏆 Best Hyperparameters: {best_params}")
    print(f"📊 Best {scoring}: {grid_search.best_score_:.4f}")

    # Helper Function
    # Saves best params to disk for ease of storage
    save_best_params(best_params, model.__class__.__name__)

    return best_model, best_params, cv_results

def save_best_params(best_params, model_name, save_dir='../tuned_params'):
    '''
    Saves best hyperparameters into a file one level back called tuned_params

    Example Usage:
        save_best_params(best_params, model.__class__.__name__)
    '''

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{model_name}_best_params.json') 
    with open(save_path, 'w') as f:
        json.dump(best_params, f, indent=4) 
    print(f"💾 Saved best params for {model_name} to {save_path}")

def stratified_sample(X, y, sample_frac=0.1, random_state=10):
    '''
    Stratified sample of X, y for tuning

    Args: 
        X: full feature set
        y: full labels
        sample_frac: fraction of data used to sample (10%)
        random_state: reproducibility seed (10) 

    Returns: 
        X_sample, y_sample
    '''

    X_sample, y_sample = train_test_split(X, y, train_size=sample_frac, stratify=y, random_state=random_state)
    return X_sample, y_sample