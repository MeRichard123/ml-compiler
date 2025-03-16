from torch import nn
import torch
from torch.nn import functional as F
from Utils.scan import log_g, parallel_scan_log
from Utils.Logger import SHAPE_LOG
from Utils.Moe import MoeLayer 

class minGRU(nn.Module):
    def __init__(self, embedding_dim, batch_size, hidden_size, output_size, device = 'cpu', MOE = None):
        super(minGRU, self).__init__()
        self.batch_size = batch_size
        self.device = device
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding_dim = embedding_dim

        if MOE is not None:
            self.moe = MoeLayer(MOE['number_of_experts'], embedding_dim, MOE['k'])  
    

        self.linear_z = nn.Linear(embedding_dim, hidden_size, device=device)
        self.linear_h = nn.Linear(embedding_dim, hidden_size, device=device)
        self.fc = nn.Linear(hidden_size, output_size, device=device)
        self.softmax = nn.LogSoftmax(dim=-1)

    def __str__(self):
        return "minGRU"

    def forward(self, x, h_0):
        if hasattr(self, 'moe'):
            x = self.moe(x)
        # x_t: (batch_size, seq_length, input_size)
        # h_0: (batch_size, 1, hidden_size)
        
        # Remove the squeeze operations
        k = self.linear_z(x) 
        log_z = -F.softplus(-k)
        log_coeffs = -F.softplus(k)
        log_h_0 = log_g(h_0)
        log_tilde_h = log_g(self.linear_h(x))

        SHAPE_LOG("minGRU.FORWARD() LOG_COEFFS", log_coeffs)
        SHAPE_LOG("minGRU.FORWARD() LOG_H_0", log_h_0)
        SHAPE_LOG("minGRU.FORWARD() LOG_TILDE_H + LOG_Z", log_tilde_h + log_z)

        #log_coeffs = log_coeffs.unsqueeze(1) # (batch_size, 1, hidden_size)
        # (batch_size, 2*hidden_size)
        combined = torch.cat([log_h_0, (log_z + log_tilde_h)], dim=1)

        SHAPE_LOG("minGRU.FORWARD() LOG_COEFFS", log_coeffs)
        SHAPE_LOG("minGRU.FORWARD() LOG_COMBINED", combined)

        h = parallel_scan_log(log_coeffs, combined)
        output = self.fc(h)

        output = self.softmax(output)

        return output, h
    
    def initHidden(self, batch_size=1):
        # Match the minLSTM hidden state shape
        h0 = torch.zeros(batch_size, 1, self.hidden_size, device=self.device)
        return h0