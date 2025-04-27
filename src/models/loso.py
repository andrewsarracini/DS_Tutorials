# src/models/loso.py

import matplotlib
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Use 'Agg' only in non-CLI mode 
if not hasattr(sys, 'ps1'):
    matplotlib.use('Agg')

from src.run_pipeline import tune_and_train_full

from matplotlib.colors import ListedColormap

DEFAULT_PLOT_DIR = Path(__file__).absolute().parent.parent / "sleep_plots"

def plot_subject_sequence(subject_id, model, model_name, df: pd.DataFrame, 
                          label_encoder, label_order=None, n_epochs=50, save_path=None):
    '''
    Visualizes sleep stage preds vs ground truth for a specific subject

    Args: 
        subject_id (int): subject ID to filter rows
        model: Trained classifier with .predict() 
        model_name (str): Name of the model for the plot title
        df (pd.DataFrame): full dataset with 'label' and 'subject_id'
        label_encoder: Fitted label encoder (decoding preds)
        label_order (list): Optional fixed order of labels 
        n_epochs (int): Number of epochs to plot (default=50)
    '''
    # Filter for the subject: 
    df_subj = df[df['subject_id'] == subject_id].copy()

    # Predict: 
    X_subj = df_subj.drop(columns=['label','binary_label', 'subject_id'], errors='ignore')
    y_true = df_subj['label'].values

    y_pred = label_encoder.inverse_transform(model.predict(X_subj))

    # For safety, clip to n_epochs available
    n_epochs = min(n_epochs, len(df_subj))
    y_true, y_pred = y_true[:n_epochs], y_pred[:n_epochs] 
    df_subj = df_subj.iloc[:n_epochs]
    
    # Label Mapping
    if label_order is None: 
        label_order = sorted(list(set(y_true) | set(y_pred)))

    label_to_int = {label: i for i, label in enumerate(label_order)} 
    y_true_idx = [label_to_int[y] for y in y_true]
    y_pred_idx = [label_to_int[y] for y in y_pred]

    # Add a thin white row separator between Ground Truth and Classifier
    data = np.array([y_true_idx, y_pred_idx])

    stage_colors = {
        "Wake": "#9b59b6",  # Soft lavender
        "N1": "#f1c40f",     # Yellow
        "REM": "#e67e22",    # Rich orange (darker than N1)
        "N2": "#1abc9c",     # Teal
        "N3": "#2c3e50"      # Dark blue-gray
    }

    cmap = ListedColormap([stage_colors[stage] for stage in label_order])

    fig, ax = plt.subplots(figsize=(16, 3))
    ax.imshow(data, 
              cmap=cmap, 
              aspect='auto', 
              interpolation='none', 
              extent=[0, data.shape[1], 0, data.shape[0]])

    # Cycle markers with annotations
    if 'cycle' in df_subj.columns:
        cycle_seq = df_subj['cycle'].values[:n_epochs]
        for i in range(1, len(cycle_seq)):
            if cycle_seq[i] != cycle_seq[i - 1]:
                ax.axvline(i - 0.5, color='black', linestyle='--', linewidth=3)
                ax.text(i, 1.08, f"Cycle {cycle_seq[i]}", rotation=0,
                        verticalalignment='top', horizontalalignment='center', 
                        fontsize=14, fontweight='bold')

    # X-axis ticks every 200 epochs
    ax.set_xticks(np.arange(0, n_epochs + 1, 200)) 

    # Draw white horizontal separator line
    ax.axhline(y=1, color='white', linewidth=9)

    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels([f'{model_name}', 'Ground Truth'], fontsize=14)
    ax.tick_params(axis='x', labelsize=12)
    ax.set_title(f"Sleep Stage Sequence – Subject {subject_id}", fontsize=26)

    handles = [plt.Line2D([0], [0], color=stage_colors[stage], lw=6) for stage in label_order]
    ax.legend(handles, label_order, bbox_to_anchor=(1.01, 1), loc='upper left', title="Stages", fontsize=12, title_fontsize=13)

    plt.tight_layout()
    plt.show()

# =========================================================================

def loso_full(df: pd.DataFrame, model_class, model_name, n_trials=30,
                          save_plot=False, target_subject=None, plot_dir=DEFAULT_PLOT_DIR,
                          model_params=None, target_col='label'):
    '''
    Performs leave-one-subject-out cross-validation

    Args:
        df (pd.DataFrame): Dataset including 'label' and 'subject_id'
        model_class: Classifier to train (e.g., LGBMClassifier)
        model_name (str): Name used for reporting
        n_trials (int): Number of Optuna trials for tuning
        save_plot (bool): Toggle to save plots 
        plot_dir (str): Directory to save plots

    Returns:
        results (dict): Dictionary mapping subject_id to evaluation result
    '''

    results = {}
    if target_subject is not None:
        subjects = [target_subject]
    else:
        subjects = df['subject_id'].unique()

    for left_out in subjects:
        print(f"\n🚪 Leaving out Subject {left_out}...\n")

        df_train = df[df['subject_id'] != left_out].copy()
        df_test = df[df['subject_id'] == left_out].copy()

        if df_test.empty:
            print(f"⚠️ No data found for subject {left_out}, skipping.")
            continue

        le = LabelEncoder()
        le.fit(df_train[target_col]) 

        y_train = le.fit_transform(df_train[target_col])
        y_test = le.transform(df_test[target_col])

        # (always drop both label-type cols)
        X_train = df_train.drop(columns=['label', 'binary_label', 'subject_id'], errors='ignore')
        X_test = df_test.drop(columns=['label', 'binary_label', 'subject_id'], errors='ignore')

        #=====================================================
        # TEMP-- for testing! 
        print("\n📦 Training features:", list(X_train.columns))
        #=====================================================

        trained_model, best_params, final_metrics = tune_and_train_full(
            model_class=model_class,
            model_name=f"{model_name}_{left_out}", # Noting that this is here...
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test, 
            model_params=model_params or {'class_weight': 'balanced'},
            sample_frac=0.99,
            scoring='f1_weighted',
            use_scaler=False,
            n_trials=n_trials,
            label_encoder=le)
        
        results[left_out] = {
            "model": trained_model,
            "best_params": best_params,
            "label_encoder": le,
            **{k: final_metrics.get(k) for k in ("accuracy", "precision", "recall", "weighted_f1")},
            **{
                f"{cls}_f1": score
                for cls, score in final_metrics.get("per_class_f1", {}).items()
            },
            "all_metrics": final_metrics
        }


        # Optional-- plot or save:
        if save_plot: 
            plot_dir.mkdir(parents=True, exist_ok=True)
            save_path = plot_dir / f'{model_name}_{left_out}.png'
        else: 
            save_path = None

        plot_subject_sequence(
            subject_id = left_out,
            model = trained_model,
            model_name = f'{model_name}_s{left_out}',
            df = df, 
            label_encoder= le, 
            n_epochs= len(df_test),
            save_path=save_path
        )

        return trained_model, best_params, final_metrics

    