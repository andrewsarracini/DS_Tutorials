import torch 
from torch.utils.data import Dataset
import pandas as pd

class LSTMDataset(Dataset): 
    def __init__(self, df: pd.DataFrame, feature_cols, label_col='label', window_size=10):
        '''
        Converts a df into rolling sequences for LSTM input

        Args: 
            df: pandas df with sequential data 
            feature_cols (list): Which cols to use as feats
            label_col (str): Col name of hte target label
            window_size (int): How many timesteps / sequence 
        '''
        self.features = df[feature_cols].values
        self.labels = df[label_col].values
        self.window_size = window_size

    def __len__(self): 
        # Total number of rolling sequences we can create
        return len(self.features) - self.window_size + 1 
    
    def __getitem__(self, idx): 
        # Return one windowed sequence and its label 
        x_seq = self.features[idx: idx + self.window_size]
        y = self.labels[idx + self.window_size - 1] # timestep label
        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y) 
    

# === WHAT DOES THIS DO? ===
# For a 10-row window, returns: 

# X = tensor([[delta₀, theta₀, ...],  ← epoch 0
#             [delta₁, theta₁, ...],  ← epoch 1
#             ...
#             [delta₉, theta₉, ...]]) ← epoch 9

# y = tensor(label₉)

# This way, each call gives a sequences + label of the final step 