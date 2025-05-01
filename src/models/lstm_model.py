import torch 
import torch.nn as nn 

class SleepLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, num_classes=5, dropout=0.0):
        super(SleepLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, num_classes) 

    def forwar(self, x):
        # x: (batch, seq_len, input_size) 
        output, (hn, cn) = self.lstm(x)
        last_hidden = output[:,-1, :] # Take last time step
        logits = self.fc(last_hidden) # (batch, num_classes) 
        return logits
    
