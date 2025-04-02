import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
import math

# --- EDA: 7 STEPS ---
# 1. Data Cleaning and Preprocessing
# 2. Descriptive Stats
# 3. Univariate Analysis
# 4. Bivariate Analysis 
# 5. Multivariate Analysis 
# 6. Feature Engineering
# 7. Visualization

# 1. Data Cleaning and Preprocessing
def check_missing_vals(df: pd.DataFrame): 
    '''
    Checks for missing values
    '''
    missing = df.isnull().sum()
    print('\nMissing Values:')
    print(missing[missing > 0])
    return missing

def detect_constant_feats(df: pd.DataFrame):
    ''' 
    Detects and returns features with only one unique value.
    An alert is triggered if a constant feature is detected 
    '''
    const_feats = [col for col in df.columns if df[col].nunique() == 1] 
    if const_feats > 0: 
        print(f'\n Constant Features:, {const_feats}')
    else: 
        print(f'\n No constant features detected!') 

# 2. Get Descriptive Stats
# Adding this simiply to stay in order for a good EDA! 

# 3. Univariate Analysis
def plot_dists(df: pd.DataFrame, features: list):
    ''' 
    Plots distributions of specified features
    
    - If 1-2 features: individual plots
    - if 3+ features: grid layout
    '''

    rows = math.ceil(len(features) / 3) # max 3 plots per row
    cols = min(len(features), 3) # 3 columns for readability

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).flatten() 

    # Now onto plot generation

    for i, feature in enumerate(features):
        sns.histplot(df[feature], kde=True, color='blue', ax=axes[i]) 
        axes[i].set_title(f'Distribution of {feature}')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Count')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

# 4. Bivariate Analysis
def compare_class_means(df: pd.DataFrame, target: str):
    '''
    Compares mean values across classes and returns top differentiators 
    Provides insight into potentially informative features
    Large differences indicate that there's a signal within the data

    Args: 
        df: Input dataframe
        target (str): Name of target column

    Returns:
        top_diff (df): Top differentiating features by mean difference
    '''

    grouped_means = df.groupby(target).mean().T
    grouped_means['diff'] = abs(grouped_means[0] - grouped_means[1]) 
    top_diff = grouped_means.sort_values(by='diff', ascending=False).head(10)

    print(f'\nTop Differentiating Features (mean diff):')
    print(top_diff[['diff']])

    # Plot top differentiators
    top_feats = top_diff.index.tolist()
    for feat in top_feats:
        plt.figure(figsize=(6, 3))
        sns.boxplot(data=df, x=target, y=feat)
        plt.title(f"{feat} by {target}")
        plt.tight_layout()
        plt.show()

    return top_diff

# 5. Multivariate Analysis: 
def compute_mutual_info(df: pd.DataFrame, target: str):
    ''' 
    Calculates mutual information scores between features and target
    MI captures non-linear relationships 
    Scores that are close to 0 across the board indicate problems

    Args: 
        df: Input dataframe
        target (str): Name of target col

    Returns: 
        mi_series (pd.Series): Mutual info scores
    '''

    X = df.drop(columns=target)
    y = df[target]

    mi = mutual_info_classif(X, y, discrete_features='auto', random_state=10)
    mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)

    print(f'\nMutual Information Scores:')
    print(mi_series.head(10))

    # Plot mutual info
    plt.figure(figsize=(8, 5))
    sns.barplot(x=mi_series.head(15), y=mi_series.head(15).index)
    plt.title("Top Mutual Information Scores")
    plt.xlabel("Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

    return mi_series

def detect_low_var_feats(df: pd.DataFrame, threshold=0.01):
    '''
    Detects features with low variance

    Args: 
        df: input dataframe
        threshold (float): Minimum variance threshold

    Returns: 
        low_var_feats (list): List of low variance feature names
    '''

    low_var_feats = [col for col in df.columns if df[col].var() < threshold]
    print(f"\nLow Variance Features (< {threshold}):", low_var_feats)

    return low_var_feats
