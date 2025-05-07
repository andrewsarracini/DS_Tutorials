import torch 
from torch.utils.data import Dataset
import pandas as pd

class LSTMDataset(Dataset): 
    def __init__(self, df: pd.DataFrame, feature_cols, label_col='label', 
                 window_size=10, stride=None, seq2seq=False):
        '''
        Converts a df into rolling sequences for LSTM input, optional overlap via stride

        Args: 
            df: pandas df with sequential data 
            feature_cols (list): Which cols to use as feats
            label_col (str): Col name of hte target label
            window_size (int): How many timesteps / sequence 
            stride (int): Step size between window starts (default: no overlap)
            seq2seq (book): If True, returns label seq instead of singular final label
        '''
        self.features = df[feature_cols].values
        self.labels = df[label_col].values
        self.window_size = window_size
        self.stride = stride if stride is not None else window_size
        self.seq2seq = seq2seq

        # Precompute all valid start indices
        self.start_indices = list(range(0, len(self.features) - window_size +1, self.stride))

    def __len__(self): 
        # Total number of rolling sequences we can create
        return len(self.start_indices)
    
    def __getitem__(self, idx): 
        start_idx = self.start_indices[idx]
        end_idx = start_idx + self.window_size

        x_seq = self.features[start_idx:end_idx]
        y_seq = self.labels[start_idx:end_idx] if self.seq2seq else self.labels[end_idx - 1]

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