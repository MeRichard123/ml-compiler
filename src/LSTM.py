from torch import nn
import torch

class LSTM:
    def __init__(self, input_size, hidden_size, num_layers, output_size, device = 'cpu'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.device = device

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def __repr__(self) -> str:
        return f"""LSTM(
            input_size={self.input_size},
            hidden_size={self.hidden_size},
            num_layers={self.num_layers},
            output_size={self.output_size}
        )
        """
    

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out