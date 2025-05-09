import torch 
import torch.nn as nn 

# input size: # feats / timestep (bands) + engineered feats
# hidden_size: LSTM units 
# num_layers: stacked LSTM layers
# num_classes: 2 for binary, 5 for multiclass (N1, N2, N3, REM, Wake) 
# dropout: used between LSTM layers (if > 1) 

class SleepLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, 
                 num_classes=5, dropout=0.0, bidirectional=False,
                 use_attention=False):
        super(SleepLSTM, self).__init__()

        self.bidirectional = bidirectional
        self.hidden = hidden_size
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True, # makes input shape (batch, seq_len, input_size)
            bidirectional=bidirectional
        )

        lstm_output_size = hidden_size * (2 if bidirectional else 1)

        # Final classification layer maps each timestep to a label
        self.classifier = nn.Linear(lstm_output_size, num_classes) 

    # `logits` produced... feed into CrossEntropyLoss 
    def forward(self, x):
        '''
        One prediction per timestep! 

        x : (batch_size, seq_len, input_size)
        returns: logits of shape (batch_size, seq_len, num_classes) 
        '''
        output, (hn, cn) = self.lstm(x) # output: (batch, seq_len, hidden_size * num_directions)

        logits = self.classifier(output) # (batch, seq_len, num_classes)

        return logits