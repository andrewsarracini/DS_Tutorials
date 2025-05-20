import csv
import datetime
from os import mkdir
import matplotlib.pyplot as plt
from imblearn.pipeline import Pipeline as imbpipeline
from imblearn.over_sampling import SMOTE
from matplotlib.pyplot import step
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold, train_test_split
from torch import lstm, threshold
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd 
from datetime import datetime
from pathlib import Path

import warnings
import time

from src.helper import save_best_params, load_best_params
from src.logger_setup import logger 
from src.helper import stratified_sample, dynamic_param_grid, param_spaces

import json

from src.paths import LOG_DIR

def grand_tuner(model, param_grid, X, y, cv=5, scoring='roc_auc', use_smote=True, n_iter=20, verbose=True):
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

    if verbose:
        print(f"Starting Grand Tuner | CV: {cv} | Scoring: {scoring}")
        print(f"Model: {model.__class__.__name__} | SMOTE: {use_smote} | Random Iterations: {n_iter}")
        print("="*60, '\n')

    if param_grid is None:
        param_grid = param_spaces.get(model.__class__.__name__, {})
        print(f"**{model.__class__.__name__} param grid is None, using default\n")

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
    print(f"⏱️ RandomizedSearchCV completed in {(end_random - start_random)/60:.2f} minutes\n")


    # Get best parameters from RandomizedSearch
    best_random_params = random_search.best_params_
    # print(f"\n🎲 Best Parameters from RandomizedSearch: {best_random_params}")

    # === Inject all best params into pipeline before GridSearch ===
    model_specific_params = {k: v for k, v in best_random_params.items() if k.startswith('classifier__')}
    pipeline.set_params(**model_specific_params)

    refined_grid = dynamic_param_grid(model, best_random_params)
    # print(f"\n🛠️ Running GridSearchCV with refined parameters: {refined_grid}")

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
    print(f"⏱️ GridSearchCV completed in {(end_grid - start_grid)/60:.2f} minutes\n")


    best_model = grid_search.best_estimator_
    best_grid_params = grid_search.best_params_

    # Merge all params: RandomSearch + GridSearch overwrite
    all_best_params = best_random_params.copy()
    all_best_params.update(best_grid_params)

    cv_results = grid_search.cv_results_

    print("=" * 50)
    # print(f"\n✅ Best Model Found: {best_model}")
    print(f"🏆 Best Hyperparameters: {json.dumps(all_best_params, indent=2)}")
    print(f"\n📊 Best {scoring}: {grid_search.best_score_:.4f}")

    # Helper Function
    # Saves best params to disk for ease of storage
    save_best_params(all_best_params, model.__class__.__name__)

    return best_model, all_best_params, cv_results

# New Age: Optuna Tuner
# --------------------------------------------------------
# Optuna Tuner for sklearn-compatible models

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score

import optuna

warnings.filterwarnings("ignore") 

def get_optuna_search_space(trial, model_class):
    '''
    Returns a dictionary of hyperparameters for the given model_class using Optuna's suggestion API.
    '''
    model_name = model_class.__name__

    if model_name == "RandomForestClassifier":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 5, 50),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }

    elif model_name == "XGBClassifier":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }

    elif model_name == "LGBMClassifier":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 5, 50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        }

    elif model_name == "LogisticRegression":
        return {
            "C": trial.suggest_loguniform("C", 1e-3, 1e2),
            "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
            "solver": trial.suggest_categorical("solver", ["liblinear", "saga"]),
        }

    elif model_name == "GradientBoostingClassifier":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
        }

    elif model_name == "AdaBoostClassifier":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0),
        }

    else:
        raise ValueError(f"Optuna tuning not yet supported for {model_name}.")

def optuna_tuner(model_class, X, y, scoring='f1_weighted',
                 n_trials=50, cv=5, random_state=10, verbose=True,
                 study_name=None):
    '''
    Optuna-based tuner for sklearn-compatibile models

    Args: 
        model_class: Classifier class (ex. RandomForestClassifier, LightGBMClassifier)
        X (pd.Dataframe): Training Features
        y (array): Training Labels
        scoring (str): Metric to optimize ('accuracy', 'f1_weighted', etc.)
        n_trials (int): Number of trials for Optuna
        cv = Cross-Validation folds
        random_state: Reproducibility
    
    Returns: 
        tuple: 
            best_model: trained sklearn best model
            best_params: dict of best params
            study: Optuna study object containing trial results
    '''

    def objective(trial):
        params = get_optuna_search_space(trial, model_class)
        model = model_class(**params)

        # Cross-validation
        scores = cross_val_score(model, X, y, scoring=scoring, cv=cv).mean()
        return scores
    
    progress_bar = TqdmProgressBar(n_trials)
    optuna.logging.set_verbosity(optuna.logging.WARNING)  # Suppress Optuna logs
    
    study = optuna.create_study(direction="maximize", study_name=study_name, sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, n_jobs=-1, callbacks=[progress_bar])
    progress_bar.close()

    best_params = study.best_params
    best_model = model_class(**best_params)
    best_model.fit(X, y)

    return best_model, best_params, study

# MAKING A PROGRESS BAR! 
from tqdm import tqdm

class TqdmProgressBar:
    def __init__(self, total_trials):
        self.pbar = tqdm(total=total_trials, desc="🔍 Tuning", ncols=100)

    def __call__(self, study, trial):
        self.pbar.update(1)

    def close(self):
        self.pbar.close()

# --------------------------------------------------------

def lstm_search_space(trial):
    return {
        "hidden_size": trial.suggest_categorical("hidden_size", [64, 128, 256]),
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "bidirectional": trial.suggest_categorical("bidirectional", [True, False]),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-4, 1e-2),
        "stride": trial.suggest_categorical("stride", [1, 2, 4]),
        "seq_len": trial.suggest_categorical("seq_len", [32, 64, 128]),
        "epochs": trial.suggest_int("epochs", 5, 15),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "weight_decay": trial.suggest_loguniform("weight_decay", 1e-5, 1e-2),
    }

from src.models.loso_lstm import loso_lstm
from src.eval import eval_probs, find_best_thresh
import numpy as np

def optuna_lstm_tuner(n_trials=30, random_state=10,
                      static_config=None, subject_list=None,
                      export_csv=True, export_path=None): 
    
    def lstm_full_loso_objective(trial): 
        trial_config = lstm_search_space(trial)
        f1_scores = []
        subject_results = {}

        for subject in subject_list:
            config = {
                **static_config, 
                **trial_config, 
                'target_subject': subject
            }

            result = loso_lstm(config)
            f1 = result['f1_weighted']
            f1_scores.append(result['f1_weighted']) 

            subject_results[str(subject)] = {
                'f1': f1, 
                'thresh': result.get('threshold'), 
                'acc': result.get('accuracy') 
            }

        avg_f1 = np.mean(f1_scores) 
        trial.set_user_attr('avg_f1', avg_f1) 
        trial.set_user_attr('subject_scores', subject_results) 
        trial.set_user_attr('params', trial_config)

        return avg_f1
                    
    progress_bar = TqdmProgressBar(n_trials)
    optuna.logging.set_verbosity(optuna.logging.WARNING)  # Suppress Optuna logs

    study = optuna.create_study(
        direction='maximize', 
        sampler=optuna.samplers.TPESampler(seed=random_state)
    )
    study.optimize(lstm_full_loso_objective, n_trials=n_trials, callbacks=[progress_bar])
    progress_bar.close() 

    if export_csv: 
        export_subject_scores(study, path=export_path) 

    return study.best_params, study


def export_subject_scores(study, path=None):
    rows = []
    for trial in study.trials:
        trial_id = trial.number
        avg_f1 = trial.user_attrs.get('avg_f1') 
        params = trial.user_attrs.get('params', {})
        subject_scores = trial.user_attrs.get('subject_scores', {})

        for subject_id, metrics in subject_scores.items(): 
            rows.append({
                'trial': trial_id, 
                'subject': subject_id, 
                'f1': metrics['f1'], 
                'thresh': metrics['thresh'],
                'acc': metrics['acc'], 
                'avg_f1': avg_f1,
                **params
            })

        df = pd.DataFrame(rows)

        if path is None:
            timestamp = datetime.now().strftime("%m-%d")
            path = LOG_DIR / f'optuna_lstm_{timestamp}.csv' 

        Path(path).parent.mkdir(parents=True, exist_ok=True) 
        df.to_csv(path, index=False) 
        print(f"✅ Exported subject scores to {path}")
        
# ----------------------------------------
# Leveraging Optuna's Wealth of Built-Ins! 

from optuna.importance import get_param_importances
from optuna.visualization import plot_parallel_coordinate, plot_contour

def analyze_study(study, output_dir = LOG_DIR / 'optuna_analysis'): 
    '''
    Generates a visual and statistical analysis of an Optuna study, including:
        - Hyperparam importance
        - Correlation heatmaps
        - Paralell coordinate plot
        - Contour plots for key pairs

    Args: 
        study (optuna): Completed Optuna study object 
        output_dir (str or Path): Directory to save the output plots
    '''
    output_dir = Path(output_dir)
    output_dir = mkdir(parents=True, exist_okay=True) 

    # --- Important Bar Plot ---
    importances = get_param_importances(study)

    plt.figure(figsize=(8,4)) 
    plt.bar(importances.keys(), importances.values())
    plt.title('Hyperparameter Importance (F1)') 
    plt.ylabel('Optuna Importance Score') 
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'f1_importance_barplot.png') 

    # --- Correlation Heatmap --- 
    df = study.trials_dataframe() 
    filtered = df[[col for col in df.columns if col.startswith('params_')] + ['value']].dropna()

    corr = filtered.corr()
    matrix = np.triu(corr) 

    plt.figure(figsize=(6, len(corr)//2)) 
    sns.heatmap(corr[['value']].sort_values(by='value', ascending=False), 
                annot=True, cmap='coolwarm', mask=matrix)
    plt.title('Correlation with F1') 
    plt.tight_layout()
    plt.savefig(output_dir / 'corr_heatmap.png')
    plt.close()

    # --- Paralell Coordinate Plot ---  
    fig = plot_parallel_coordinate(study) 
    fig.write_image(str(output_dir / 'paralell_coords.png')) 

    # --- Contour Plots for Top 2 Numerical Params --- 
    top_numeric = [k for k in importances if isinstance(importances[k], (int, float))][:2]

    if len(top_numeric) == 2: 
        fig = plot_contour(study, params=top_numeric)
        fig.write_image(str(output_dir / 'contour_plot.png')) 

    print(f'✅ Optuna Analysis report saved to {output_dir.resolve()}') 