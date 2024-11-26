import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device) -> None:
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.device = device

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size, device=device)
        self.i2o = nn.Linear(input_size + hidden_size, output_size, device=device)
        self.o2o = nn.Linear(hidden_size + output_size, output_size, device=device)
        #self.dropout = nn.Dropout(0.1)

        '''
        - Since I am using nn.NLLLoss I need LogSoftmax
        - If I change to use CrossEntropyLoss i can remove the softmax layer
            - because CrossEntropyLoss() = LogSoftmax() + NLLLoss() 
        '''
        self.softmax = nn.LogSoftmax(dim=1) 

    def forward(self, input, hidden):
        input_combined = torch.cat((input, hidden), 1).to(self.device)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1).to(self.device)
        output = self.o2o(output_combined)
        #output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden
    
    def initHidden(self):
        return torch.zeros(1, self.hidden_size, device=self.device)