from torch import nn
import torch
from torch.nn import functional as F

class LSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_size, num_layers, output_size, device = 'cpu'):
        super(LSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.device = device

        # Defining the LSTM layer
        self.lstm = nn.LSTM(self.embedding_dim, hidden_size, num_layers, batch_first=True, device=device)
        # Defining the Fully Connected output layer - for reshaping the output to the desired output size
        self.fc = nn.Linear(hidden_size, output_size, device=device)

    def __str__(self):
        return "LSTM"

    def __repr__(self) -> str:
        return f"""LSTM(
            input_size={self.input_size},
            hidden_size={self.hidden_size},
            num_layers={self.num_layers},
            output_size={self.output_size}
        )
        """
    

    def forward(self, input, hidden):
        input = input.to(self.device)
        out, hidden = self.lstm(input, hidden)
        out = self.fc(out)
        # log softmax
        out = F.log_softmax(out, dim=1)
        return out, hidden
    
    def initHidden(self):
        # Initialize h0 and c0 with the correct dimensions
        h0 = torch.zeros(self.num_layers, self.hidden_size, device=self.device)
        c0 = torch.zeros(self.num_layers, self.hidden_size, device=self.device)
        return h0, c0
