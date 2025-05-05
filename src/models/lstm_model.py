import torch 
import torch.nn as nn 

# input size: # feats / timestep (bands) + engineered feats
# hidden_size: LSTM units 
# num_layers: stacked LSTM layers
# num_classes: 2 for binary, 5 for multiclass (N1, N2, N3, REM, Wake) 
# dropout: used between LSTM layers (if > 1) 

class SleepLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, 
                 num_classes=5, dropout=0.0, bidirectional=False):
        super(SleepLSTM, self).__init__()

        self.bidirectional = bidirectional
        self.hidden = hidden_size
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True, # makes input shape (batch, seq_len, input_size)
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        # final layer
        # transforms last hidden state -> logits over class scores
        # if needed-- double output size if bidirectional!
        direction_factor = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * direction_factor, num_classes)

    # output sequence of hidden states (discard all but the last!) 
    # hn, cn: hidden cell states 
    # Take last timestep output; pass it through fc() 
    # `logits` produced... feed into CrossEntropyLoss 
    def forward(self, x):
        # x: (batch, seq_len, input_size) 
        output, (hn, cn) = self.lstm(x)

        if self.lstm.bidirectional:
            # Concat last hidden states from both directions
            # hn shape: (num_layers * num_directions, batch, hidden_size)
            # SO: grab [-2] for the last forward layer, [-1] for backward
            last_hidden = torch.cat((hn[-2], hn[-1]), dim=1) # shape: (batch, 2*hidden_size)
        else:
            last_hidden = hn[-1] # shape: (batch, num_classes)

        logits = self.fc(last_hidden) # (batch, num_classes) 
        return logits
    
