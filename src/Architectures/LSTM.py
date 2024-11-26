from torch import nn

class LSTM(nn.Module):
    def __init__(self, hidden_size, num_layers, output_size, device = 'cpu'):
        super(LSTM, self).__init__()
        self.input_size = 1
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.device = device

        # Defining the LSTM layer
        self.lstm = nn.LSTM(self.input_size, hidden_size, num_layers, batch_first=True, device=device)
        # Defining the Fully Connected output layer - for reshaping the output to the desired output size
        self.fc = nn.Linear(hidden_size, output_size, device=device)

    def __repr__(self) -> str:
        return f"""LSTM(
            input_size={self.input_size},
            hidden_size={self.hidden_size},
            num_layers={self.num_layers},
            output_size={self.output_size}
        )
        """
    

    def forward(self, x):
        x = x.to(self.device)
        # https://stackabuse.com/how-to-use-gpus-with-pytorch/
        #h0 = torch.zeros(self.num_layers, self.hidden_size, device=self.device)
        #c0 = torch.zeros(self.num_layers, self.hidden_size, device=self.device)

        out, _ = self.lstm(x)
        print(f"Shape of out: {out.size()}")
        x = out[:, -1, :]
        out = self.fc(out)
        return out
