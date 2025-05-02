from torch import nn
import torch

def generate_causal_mask(seq_len, device):
    # (seq_len, seq_len) upper-triangular mask
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)
    return mask  # shape: (1, 1, seq_len, seq_len)

class AttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_p=0.1, device='cpu'):
        super(AttentionLayer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.device = device

        assert (
            self.head_dim * num_heads == embed_dim
        ), "Embedding dimension must be divisible by number of heads"

        self.values = nn.Linear(embed_dim, embed_dim, bias=False).to(device)
        self.keys = nn.Linear(embed_dim, embed_dim, bias=False).to(device)
        self.queries = nn.Linear(embed_dim, embed_dim, bias=False).to(device)
        self.fc_out = nn.Linear(embed_dim, embed_dim).to(device)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, mask=None):
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

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-inf'))

        attention = torch.softmax(energy / (self.head_dim ** (1 / 2)), dim=-1)
        attention = self.dropout(attention)

        # Apply attention to values
        out = attention @ values

        out = out.permute(0, 2, 1, 3).contiguous().view(N, seq_len, self.embed_dim)

        out = self.dropout(self.fc_out(out))
        return out, attention