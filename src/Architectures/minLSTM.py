from torch import nn
import torch
from torch.nn import functional as F
from Utils.Moe import MoeLayer
from Utils.Attention import AttentionLayer
from Utils.scan import log_g, parallel_scan_log
from Utils.Logger import SHAPE_LOG

class minLSTM(nn.Module):
    def __init__(self, embedding_dim, batch_size, hidden_size, output_size, device = 'cpu', MOE = None, Attention = False):
        super(minLSTM, self).__init__()
        self.batch_size = batch_size
        self.device = device
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding_dim = embedding_dim

        if MOE is not None:
            self.moe = MoeLayer(MOE['number_of_experts'], embedding_dim)  
        
        if Attention:
            self.attention = AttentionLayer(embedding_dim, 10) 

        self.linear_f = nn.Linear(embedding_dim, hidden_size, device=device)
        self.linear_i = nn.Linear(embedding_dim, hidden_size, device=device)
        self.linear_h = nn.Linear(embedding_dim, hidden_size, device=device)
        
        self.fc = nn.Linear(hidden_size, output_size, device=device)
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.LogSoftmax(dim=-1)

    def __str__(self):
        return "minLSTM"
    
    def forward(self, x, h_0):
        # x_t: (batch_size, seq_length, input_size)
        # h_0: (batch_size, 1, hidden_size)

        if hasattr(self, 'attention'):
            x = self.attention(x)

        if hasattr(self, 'moe'):
            x = self.moe(x)


        SHAPE_LOG("minLSTM.FORWARD() x", x)

        diff = F.softplus(-self.linear_f(x)) \
                    - F.softplus(-self.linear_i(x))
        
        log_f = -F.softplus(diff)
        log_i = -F.softplus(-diff)

        log_h_0 = log_g(h_0)
        log_tilde_h = log_g(self.linear_h(x))

        SHAPE_LOG("minLSTM.FORWARD() LOG_H_0", log_h_0)
        SHAPE_LOG("minLSTM.FORWARD() LOG_TILDE_H + LOG_Z", log_tilde_h + log_i)

        SHAPE_LOG("minLSTM.FORWARD() LOG_F", log_f)

        h = parallel_scan_log(log_f, torch.cat([log_h_0, log_i + log_tilde_h],dim=1))
        output = self.fc(h)
        output = self.dropout(output)
        output = self.softmax(output)
        SHAPE_LOG("minLSTM.FORWARD() softmax", output)
        return output, h
    
    def initHidden(self, batch_size=1):
        # Initialize h0 with the correct dimensions
        h0 = torch.zeros(batch_size, 1, self.hidden_size, device=self.device)
        return h0