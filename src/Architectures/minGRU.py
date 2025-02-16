from torch import nn
import torch
from torch.nn import functional as F
from Utils.scan import log_g, parallel_scan_log
from Utils.Logger import SHAPE_LOG

class minGRU(nn.Module):
    def __init__(self, embedding_dim, batch_size, hidden_size, output_size, device = 'cpu'):
        super(minGRU, self).__init__()
        self.batch_size = batch_size
        self.device = device
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding_dim = embedding_dim

        self.linear_z = nn.Linear(embedding_dim, hidden_size, device=device)
        self.linear_h = nn.Linear(embedding_dim, hidden_size, device=device)
        self.fc = nn.Linear(hidden_size, output_size, device=device)
        self.softmax = nn.LogSoftmax(dim=-1)

    def __str__(self):
        return "minGRU"

    def forward(self, x, h_0):
        # x_t: (batch_size, input_size)
        # h_0: (batch_size, hidden_size)

        batch_size = x.size(0)
        x = x.squeeze(1)
        h_0 = h_0.squeeze(1)

        k = self.linear_z(x) # (batch_size, hidden_size)
        log_z = -F.softplus(-k) # (batch_size, hidden_size)
        log_coeffs = -F.softplus(k) # (batch_size, hidden_size)
        log_h_0 = log_g(h_0) # (batch_size, hidden_size)
        log_tilde_h = log_g(self.linear_h(x)) # (batch_size, hidden_size)

        SHAPE_LOG("minGRU.FORWARD() LOG_COEFFS", log_coeffs)
        SHAPE_LOG("minGRU.FORWARD() LOG_H_0", log_h_0)
        SHAPE_LOG("minGRU.FORWARD() LOG_TILDE_H + LOG_Z", log_tilde_h + log_z)

        log_coeffs = log_coeffs.unsqueeze(1) # (batch_size, 1, hidden_size)
        # (batch_size, 2*hidden_size)
        combined = torch.cat([log_h_0, (log_z + log_tilde_h)], dim=1)

        SHAPE_LOG("minGRU.FORWARD() LOG_COEFFS", log_coeffs)
        SHAPE_LOG("minGRU.FORWARD() LOG_COMBINED", combined)

        h = parallel_scan_log(log_coeffs, combined)
        output = self.fc(h)

        output = self.softmax(output)

        return output, h
    
    def initHidden(self, batch_size=1):
        # Initialize h0 with the correct dimensions
        h0 = torch.zeros(batch_size, self.hidden_size, device=self.device)
        return h0