from torch import nn
import torch
from torch.nn import functional as F
from Architectures.minGRU import minGRU

class StackedMinGRU(nn.Module):
    def __init__(self, embedding_dim, batch_size, hidden_size, output_size, device='cpu', num_experts=None, num_layers=3):
        super(StackedMinGRU, self).__init__()
        self.batch_size = batch_size
        self.device = device
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.min_grus = nn.ModuleList([
            minGRU(
                embedding_dim=embedding_dim if i == 0 else hidden_size,
                batch_size=batch_size,
                hidden_size=hidden_size,
                output_size=hidden_size if i < num_layers-1 else output_size,
                device=device,
                MOE={'number_of_experts': num_experts} if num_experts else None
            ) for i in range(num_layers)
        ])
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, h_0):
        if not isinstance(h_0, list):
            h_0 = [h_0.clone() for _ in range(self.num_layers)]

        current_input = x
        new_hiddens = []
        for i, min_gru in enumerate(self.min_grus):
            output, h_new = min_gru(current_input, h_0[i])
            current_input = self.dropout(output) if i < self.num_layers-1 else output
            new_hiddens.append(h_new)

        return output, new_hiddens

    def initHidden(self, batch_size=1):
        return [min_gru.initHidden(batch_size) for min_gru in self.min_grus]