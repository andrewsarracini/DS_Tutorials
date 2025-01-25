import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation_heatmap(df: pd.DataFrame):
    '''Plots a heatmap of feature correlations.'''
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
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

# More to come!