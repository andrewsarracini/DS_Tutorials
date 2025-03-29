import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor 
from sklearn.preprocessing import StandardScaler
import math
from sklearn.linear_model import ElasticNetCV
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold


# Note! If data contains non-numerical data, make sure One-Hot Encoding occurs BEFORE running eda functions
# Note, use pd.get_dummies(df, drop_first = True) 

def plot_correlation_heatmap(df: pd.DataFrame):
    '''Plots a heatmap of feature correlations.'''
    corr_matrix = df.corr()
    # Makes the heatmap triangular-- cuts off redundant parts!
    matrix = np.triu(corr_matrix)
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", mask=matrix)
    plt.title("Correlation Heatmap")
    plt.show()

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

    # for feature in features: 
    #     plt.figure(figsize=(8,4))
    #     sns.histplot(df[feature], kde=True, color='blue')
    #     plt.title(f'Distribution of {feature}')
    #     plt.xlabel(feature)
    #     plt.ylabel("Count") 
    #     plt.show() 

def extended_describe(df: pd.DataFrame): 
    '''
    Computes advanced statistics beyond pandas `.describe()`.
    Includes skewness, kurtosis, and percentiles for numerical features.

    Updated to only include numeric features as `df_num`

    - Skewness: Measures the asymmetry of the data distribution.
        - Positive skew: A longer tail on the right (right-skewed).
        - Negative skew: A longer tail on the left (left-skewed).
        - Use case: Identifies if the data is symmetric or skewed, which can guide transformations like log or square root.

    - Kurtosis: Measures the "tailedness" of the data distribution.
        - High kurtosis: Heavy tails and outliers (leptokurtic).
        - Low kurtosis: Light tails (platykurtic).
        - Use case: Detects the presence of outliers and the likelihood of extreme values.

    - Percentiles: Show the value below which a given percentage of data falls.
        - Example: 25th percentile (Q1), 75th percentile (Q3), etc.
        - Use case: Helps identify the spread and central tendency of the data, and calculate interquartile range (IQR) for outlier detection.
    
    Returns:
        pd.DataFrame: A DataFrame with detailed statistics for numerical features.
    '''
    df_num = df.select_dtypes('number')

    stats = df_num.describe().T
    stats['skewness'] = df_num.skew()
    stats['kurtosis'] = df_num.kurtosis()
    # percentiles = [5, 25, 75, 95]
    # for p in percentiles:
    #     stats[f'{p}th percentile'] = df_num.quantile(q = p/100) 
    return stats


def calc_vif(df: pd.DataFrame):
    '''
    Computes Variance Inflation Factor (VIF) for numerical features.
    
    - VIF > 10: Severe multicollinearity (consider removing the feature).
    - VIF > 5: High multicollinearity (investigate further).
    - VIF ≤ 5: Low multicollinearity (acceptable).

    Args:
        df (pd.DataFrame): The dataset.

    Returns:
        pd.DataFrame: A DataFrame showing VIF values for each numeric feature.
    '''
    
    # Select only numeric columns
    num_cols = df.select_dtypes(include=['number']).columns

    # Protect against empty selection
    if len(num_cols) == 0:
        raise ValueError("No numeric columns found in the dataset.")

    # Compute VIF
    vif_data = pd.DataFrame()
    vif_data['Feature'] = num_cols
    vif_data['VIF'] = [variance_inflation_factor(df[num_cols].values, i) for i in range(len(num_cols))]

    return vif_data

# Under construction, merging lasso_feat_select and elastic_feat_select into feat_select

# def feat_select(df: pd.DataFrame, target:str, imbalance_thresh=0.15, smote_enabled=True):
#     '''
#     - Performs automatic feature selection using Elastic Net regression (but searches params that qualify as full lasso or full ridge)
#     - Handles class imbalance via SMOTE (if it exists)

#     Args: 
#         df (DataFrame): dataset that has already been OHE
#         target (str): name of target column
#         imbalance _thresh: if minority class is below this threshold, apply SMOTE

#     Returns: 
#         - selected_feats: list of selected feature names
#         - best_alpha: best alpha value chosen by ElasticNetCV
        
#     '''

#     # Check for imbalanced target class
#     class_counts = np.bincount(df[target])
#     min_class_ratio = class_counts.min() / class_counts.sum()

#     if min_class_ratio < imbalance_thresh and smote_enabled:
#         print(f'⚠️ Imbalanced data detected \n(Minority class = {min_class_ratio:.2%}) \nApplying SMOTE...\n')
#         smote = SMOTE(random_state=10)
#         X, y = smote.fit_resample(df.drop(columns=[target]), df[target])
#     else: 
#         X, y = df.drop(columns=[target]), df[target]

#     original_feat_count = X.shape[1]

#     # Scaling the data
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)  

#     # pre-alpha work because E-Net is finicky
#     # Now alpha is tied to signal strength!
#     min_alpha = (1.e-7) * np.median(np.abs(X_scaled.T @ y)) 

#     # pre-l1_ratio work because E-Net is finicky
#     # Hopefully this solves the SMOTE-enabled problem of killing all features
#     # By leaning more towards ridge
#     if smote_enabled:
#         l1_ratio = 0.6
#     else: 
#         l1_ratio = [0.1, 0.5, 0.9]

#     # ElasticNetCV Regularization
#     cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=10) 
#     elastic_model = ElasticNetCV(
#         # alphas = np.logspace(-6, -3, 50), 
#         alphas = np.geomspace(min_alpha, min_alpha * 100, 50),
#         l1_ratio = l1_ratio,
#         cv = cv, 
#         random_state=10
#     )
#     elastic_model.fit(X_scaled, y) 

#     # Finally, feature selection!
#     selected_feats = X.columns[elastic_model.coef_ != 0].to_list()
#     selected_feat_count = len(selected_feats)

#     print("ElasticNet Feature Selection Summary")
#     print("=====================================")
#     if smote_enabled: 
#         print('SMOTE enabled')
#     else: 
#         print('SMOTE disabled')
#     print(f"Original feature count: {original_feat_count}")
#     print(f"Selected feature count: {selected_feat_count} (🔻{original_feat_count - selected_feat_count} trimmed)")
#     print(f"Best Alpha: {elastic_model.alpha_:.2e}")
#     print(f"Best L1 Ratio: {elastic_model.l1_ratio_:.2f}")
#     print(f"Final Selected Features:")
#     print(selected_feats)

#     print('----------------')
#     nonzero_count = np.sum(elastic_model.coef_ != 0)
#     print(f"Non-zero Coefficients: {nonzero_count}/{len(elastic_model.coef_)}")
#     print("All Coefficients:", elastic_model.coef_)

#     return selected_feats

# --------------------------------------------------------

def feat_select(df_encoded: pd.DataFrame, target, smote_enabled=True, imbal_thresh=0.15):
    '''
    Feature selection using ElasticNetCV on an encoded (quickly with get_dummies) df

    Args: 
        df (DataFrame): dataset that has already been OHE
        target (str): name of target column
        smote_enabled: toggle for SMOTE oversampling
        imbal_thresh: if minority class is below this threshold, apply SMOTE

    Returns: 
        selected: List of selected feature names
    '''

    X = df_encoded.drop(columns=[target])
    y = df_encoded[target]

    # Check for imbalance
    class_counts = y.value_counts(normalize=True) 
    min_class_ratio = class_counts.min()

    if smote_enabled and min_class_ratio < imbal_thresh: 
        print(f"⚠️ Imbalanced data detected ({min_class_ratio:.2%}). Applying SMOTE...\n")
        X, y = SMOTE(random_state=10).fit_resample(X, y) 

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X) 

    min_alpha = 1e-7 * abs((X_scaled.T @ y)).mean()

    elastic_model = ElasticNetCV(
        alphas = pd.Series(np.geomspace(min_alpha, min_alpha*100, 50)), 
        l1_ratio=0.6 if smote_enabled else [0.1, 0.5, 0.9],
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=10),
        random_state=10
        ) 
    
    elastic_model.fit(X_scaled, y)

    selected = X.columns[elastic_model.coef_ != 0].tolist()

    print("📌 ElasticNet Feature Selection Summary")
    print("======================================")
    print(f"SMOTE enabled: {smote_enabled}")
    print(f"Selected feature count: {len(selected)} (🔻{X.shape[1] - len(selected)} trimmed)")
    print(f"Best Alpha: {elastic_model.alpha_:.2e}")
    print(f"Best L1 Ratio: {elastic_model.l1_ratio_:.2f}")
    print("======================================\n")

    return selected