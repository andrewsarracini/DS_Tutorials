import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor 
from sklearn.preprocessing import StandardScaler

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
    ''' Plots distributions of specified features'''
    for feature in features: 
        plt.figure(figsize=(8,4))
        sns.histplot(df[feature], kde=True, color='blue')
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel("Count") 
        plt.show() 

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
    df_num = df.select_dtypes(include=['number']).columns

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