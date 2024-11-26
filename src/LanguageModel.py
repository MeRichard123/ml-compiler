import torch
import torch.nn as nn
from Utils.Logger import LOGGER

class LanguageModel:
    def __init__(self, model: nn.Module, device: str):
        self.model = model
        self.learning_rate = 0.0005
        self.criterion = nn.NLLLoss()
        self.loss = torch.Tensor([0]).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.max_sample_length = 10
        self.device = device

    def init_model(self, text, vocab):
        self.char2idx = {char: idx for idx, char in enumerate(vocab)}
        self.idx2char = {idx: char for idx, char in enumerate(vocab)}
        self.text_idx = torch.tensor([self.char2idx[char] for char in text], dtype=torch.long, device=self.device)

    def __train(self, input_tensor, target_tensor):
        hidden = self.model.initHidden()

        self.optimizer.zero_grad()
        self.model.zero_grad()
        self.loss = 0

        for i in range(input_tensor.size(0)):
            # Prepare input as a one-hot vector for each character
            input_char = input_tensor[i].view(1, -1).float().to(self.device)
            output, hidden = self.model(input_char, hidden)
            
            target_idx = (i + 1) % target_tensor.size(0)
            target_char = target_tensor[target_idx].view(-1).to(self.device)

            l = self.criterion(output, target_char)
            self.loss += l
        self.loss.backward()
        self.optimizer.step()

        return output, self.loss.item() / input_tensor.size(0)
    
    def train_loop(self):
        n_iters = 10000
        total_loss = 0
        print_every = 500

        for iter in range(1, n_iters + 1):
            output, loss = self.__train(self.text_idx, self.text_idx) # add tensors
            total_loss += loss

            if iter % print_every == 0:
                print(f"{round(iter / n_iters * 100)}% Training, loss = {loss}%")


    
    def sample(self, start_letter = "h", temperature=0.5):
        with torch.no_grad():
            input = torch.tensor([[self.char2idx[start_letter]]], dtype=torch.float32).to(self.device)
            hidden = self.model.initHidden()

            output_name = start_letter
            LOGGER(f"Starting with: {output_name}")

            for _ in range(self.max_sample_length):
                output, hidden = self.model(input, hidden)

                '''
                # Apply temperature scaling
                output = output.div(temperature)
                LOGGER(output)
                '''

                # Sample from the scaled distribution
                output = nn.functional.softmax(output, dim=1)
                topi = torch.multinomial(output, 1)
                LOGGER(f"topV = {output[0][topi]} topi = {topi}")
                topi = topi[0][0].item()
                LOGGER(f"Generated: {self.idx2char[topi]}")
                if topi == self.char2idx['\n']:
                    break 
                else:
                    letter = self.idx2char[topi]
                    output_name += letter 
                input = torch.tensor([[self.char2idx[letter]]], dtype=torch.float32).to(self.device)
        return output_name
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, weights_only=True))
        self.model.eval()
        self.model.to(self.device)
        LOGGER(f"Model loaded from {path}")
        return self.model