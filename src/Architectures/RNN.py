import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, embedding_dim, batch_size, hidden_size, output_size, device) -> None:
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.device = device
        self.embedding_dim = embedding_dim

        self.i2h = nn.Linear(embedding_dim + hidden_size, hidden_size, device=device)
        self.i2o = nn.Linear(embedding_dim + hidden_size, output_size, device=device)
        self.o2o = nn.Linear(hidden_size + output_size, output_size, device=device)
        self.dropout = nn.Dropout(0.1)

        '''
        - Since I am using nn.NLLLoss I need LogSoftmax
        - If I change to use CrossEntropyLoss i can remove the softmax layer
            - because CrossEntropyLoss() = LogSoftmax() + NLLLoss() 
        '''
        self.softmax = nn.LogSoftmax(dim=-1)

    def __str__(self):
        return "RNN"

    def forward(self, input, hidden):
        # Input shape: (batch_size, seq_len, embedding_dim) or (1, 1, embedding_dim) for single input
        # Hidden shape: (1, batch_size, hidden_size)
        
        # Move to device
        input = input.to(self.device)
        hidden = hidden.to(self.device)
        
        batch_size = input.size(0)
        seq_len = input.size(1)
        
        # Process each step in the sequence
        outputs = []
        for i in range(seq_len):
            # Get current input slice (batch_size, embedding_dim)
            current_input = input[:, i, :]
            
            # Reshape hidden from (1, batch_size, hidden_size) to (batch_size, hidden_size)
            current_hidden = hidden.squeeze(0)
            
            # Handle batch size mismatch
            if current_input.size(0) != current_hidden.size(0):
                if current_input.size(0) == 1:
                    # During sampling: expand input to match hidden state
                    current_input = current_input.expand(current_hidden.size(0), -1)
                else:
                    # During training: use first hidden state for all batch items
                    current_hidden = current_hidden[0:1].expand(current_input.size(0), -1)
            
            # Combine input and hidden
            input_combined = torch.cat((current_input, current_hidden), 1)
            
            # Process through the network
            hidden = self.i2h(input_combined)
            output = self.i2o(input_combined)
            output_combined = torch.cat((hidden, output), 1)
            output = self.o2o(output_combined)
            output = self.dropout(output)
            outputs.append(output)
            
            # Update hidden state
            hidden = hidden.unsqueeze(0)
        
        # Stack outputs along sequence dimension
        output = torch.stack(outputs, dim=1)
        
        # Apply softmax to the output
        output = self.softmax(output)
        
        return output, hidden

    def initHidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        return torch.zeros(1, batch_size, self.hidden_size, device=self.device)