import torch 
import torch.nn as nn 

# input size: # feats / timestep (bands) + engineered feats
# hidden_size: LSTM units 
# num_layers: stacked LSTM layers
# num_classes: 2 for binary, 5 for multiclass (N1, N2, N3, REM, Wake) 
# dropout: used between LSTM layers (if > 1) 
class SleepLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, num_classes=5, dropout=0.0):
        super(SleepLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True, # makes input shape (batch, seq_len, input_size)
            dropout=dropout if num_layers > 1 else 0.0
        )
        # final layer
        # transforms last hidden state -> logits over class scores
        self.fc = nn.Linear(hidden_size, num_classes) 

    # output sequence of hidden states (discard all but the last!) 
    # hn, cn: hidden cell states 
    # Take last timestep output; pass it through fc() 
    # `logits` produced... feed into CrossEntropyLoss 
    def forward(self, x):
        # x: (batch, seq_len, input_size) 
        output, (hn, cn) = self.lstm(x)
        last_hidden = output[:,-1, :] # Take last time step
        logits = self.fc(last_hidden) # (batch, num_classes) 
        return logits
    
