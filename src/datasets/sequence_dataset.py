import torch 
from torch.utils.data import Dataset
import pandas as pd

class LSTMDataset(Dataset): 
    def __init__(self, df: pd.DataFrame, feature_cols, label_col='label', 
                 window_size=10, stride=None):
        '''
        Converts a df into rolling sequences for LSTM input (seq2seq), optional overlap via stride 

        Args: 
            df: pandas df with sequential data 
            feature_cols (list): Which cols to use as feats
            label_col (str): Col name of hte target label
            window_size (int): How many timesteps / sequence 
            stride (int): Step size between window starts (default: no overlap)
        '''
        self.features = df[feature_cols].values
        self.labels = df[label_col].values.astype('int64')
        self.window_size = window_size
        self.stride = stride if stride is not None else window_size

        # Precompute all valid start indices
        self.start_indices = list(range(0, len(self.features) - window_size +1, self.stride))

    def __len__(self): 
        # Total number of rolling sequences we can create
        return len(self.start_indices)
    
    def __getitem__(self, idx): 
        start = self.start_indices[idx]
        end = start + self.window_size

        x_seq = self.features[start:end]
        y_seq = self.labels[start:end] # full label sequence

        x_tensor = torch.tensor(x_seq, dtype=torch.float32)
        y_tensor = torch.tensor(y_seq, dtype=torch.long) 
        
        return x_tensor, y_tensor 



# === WHAT DOES THIS DO? ===
# For a 10-row window, returns: 

# X = tensor([[delta₀, theta₀, ...],  ← epoch 0
#             [delta₁, theta₁, ...],  ← epoch 1
#             ...
#             [delta₉, theta₉, ...]]) ← epoch 9

# y = tensor(label₉)

# This way, each call gives a sequences + label of the final step 