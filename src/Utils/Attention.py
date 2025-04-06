from torch import nn
import torch

class AttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(AttentionLayer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert (
            self.head_dim * num_heads == embed_dim
        ), "Embedding dimension must be divisible by number of heads"

        self.values = nn.Linear(embed_dim, embed_dim, bias=False)
        self.keys = nn.Linear(embed_dim, embed_dim, bias=False)
        self.queries = nn.Linear(embed_dim, embed_dim, bias=False)
        self.fc_out = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        N, seq_len, _ = x.shape
        value_len, key_len, query_len = x.shape[1], x.shape[1], x.shape[1]

        # Split embedding into multiple heads
        values = self.values(x).view(N, value_len, self.num_heads, self.head_dim)
        keys = self.keys(x).view(N, key_len, self.num_heads, self.head_dim)
        queries = self.queries(x).view(N, query_len, self.num_heads, self.head_dim)

        # Transpose to get dimensions for attention calculation
        values = values.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 1, 3)
        queries = queries.permute(0, 2, 1, 3)

        # Calculate attention
        energy = queries @ keys.transpose(-2, -1)

        attention = torch.softmax(energy / (self.head_dim ** (1 / 2)), dim=-1)
        attention = self.dropout(attention)

        # Apply attention to values
        out = attention @ values

        out = out.permute(0, 2, 1, 3).contiguous().view(N, seq_len, self.embed_dim)
        
        return self.fc_out(out)