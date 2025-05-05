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
        self.attn_layer = nn.Linear(self.hidden * self.num_directions, 1)

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
        output, (hn, cn) = self.lstm(x)  # output: (batch, seq_len, hidden_size * num_directions)

        # Attention: score each timestep
        attn_scores = self.attn_layer(output)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (batch, seq_len, 1)

        # Weighted sum across all time steps
        context = torch.sum(attn_weights * output, dim=1)  # (batch, hidden_size * num_directions)

        logits = self.fc(context)  # (batch, num_classes)
        return logits
