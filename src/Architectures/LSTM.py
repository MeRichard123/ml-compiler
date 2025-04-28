from torch import nn
import torch
from Utils.Moe import MoeLayer
from Utils.Attention import AttentionLayer


class LSTM(nn.Module):
    def __init__(self, embedding_dim: int, batch_size: int, hidden_size: int, 
                 output_size: int, device: str = 'cpu', MOE = None, Attention = False):
        super(LSTM, self).__init__()
        self.embedding_dim: int   = embedding_dim
        self.hidden_size: int     = hidden_size
        self.output_size: int     = output_size
        self.batch_size: int      = batch_size
        self.device: str          = device

        # Defining the LSTM layer
        self.lstm = nn.LSTM(self.embedding_dim, hidden_size, batch_first=True, device=device)

        if MOE is not None:
            self.moe = MoeLayer(MOE['number_of_experts'], embedding_dim)  
        
        if Attention:
            self.attention = AttentionLayer(embedding_dim, 10) 

        # Defining the Fully Connected output layer - for reshaping the output to the desired output size
        self.fc = nn.Linear(hidden_size, output_size, device=device)
        self.dropout = nn.Dropout(0.5)
        # Defining the softmax layer
        self.softmax = nn.LogSoftmax(dim=-1)

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

        if hasattr(self, 'attention'):
            input = self.attention(input)

        if hasattr(self, 'moe'):
            input = self.moe(input)


        out, hidden = self.lstm(input, hidden)
        out = self.fc(out)
        out = self.dropout(out)
        # log softmax
        out = self.softmax(out)
        return out, hidden
    
    def initHidden(self, batch_size=1):
        # Initialize h0 and c0 with the correct dimensions (num_layers, batch_size, hidden_size)
        h0 = torch.zeros(1, batch_size, self.hidden_size, device=self.device)
        c0 = torch.zeros(1, batch_size, self.hidden_size, device=self.device)
        return h0, c0
