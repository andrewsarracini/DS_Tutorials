from imblearn.pipeline import Pipeline as imbpipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold, train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import time

from src.helper import save_best_params, load_best_params
from src.logger_setup import logger 
from src.helper import stratified_sample, dynamic_param_grid, param_spaces

import os
import json

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