# sleep_wave/stream/live_predictor.py

import torch
import numpy as np
from collections import deque

class LivePredictor:
    def __init__(self, model, seq_len, device='cpu'): 
        self.model = model.to(device) 
        self.model.eval()
        self.seq_len = seq_len
        self.device = device
        self.buffer = deque(maxlen=seq_len) 

    def update(self, features): 
        '''
        Add new feature dict and return pred if buffer is full
        '''
        self.buffer.append([features[f] for f in ['delta', 'theta', 'alpha', 'beta']])

        if len(self.buffer) == self.seq_len:
            return self.predict() 
        
        # This just means the buffer isn't full yet
        # Therefore it can't make a pred yet 
        return None
    
    def predict(self): 
        '''
        Run model on current buffer and return latest predicted class
        '''
        x = torch.tensor([list(self.buffer)], dtype=torch.float32).to(self.device) # (1, seq_len, num_features)
        
        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=-1) # (1, num_classes)
            pred = torch.argmax(probs, dim=-1).item() 
        return pred
    
    