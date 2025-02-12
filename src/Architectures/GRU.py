import torch
import torch.nn as nn

class GRU(nn.Module):
    def __init__(self, embedding_dim, batch_size, hidden_size, output_size, device = 'cpu'):
        super(GRU, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.device = device

        self.embedding = nn.Embedding(output_size, embedding_dim)

        # Defining the GRU layer
        self.gru = nn.GRU(self.embedding_dim, hidden_size, batch_first=True, device=device)
        # Defining the Fully Connected output layer - for reshaping the output to the desired output size
        self.fc = nn.Linear(hidden_size, output_size, device=device)

        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=-1)
    
    def __str__(self):
        return "GRU"

    def forward(self, input, hidden):
        input = input.to(self.device)

        # print(f"INPUT SHAPE GRU.forward(): {input.shape}")
        # print(f"HIDDEN SHAPE GRU.forward(): {hidden.shape}")


        out, hidden = self.gru(input, hidden)
            
        out = self.fc(out)
        out = self.dropout(out)

        out = self.softmax(out)
        return out, hidden
    
    def initHidden(self, batch_size=1):
        # Initialize h0 with the correct dimensions
        h0 = torch.zeros(1, batch_size, self.hidden_size, device=self.device)
        return h0