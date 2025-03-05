import torch
import torch.nn as nn
from Utils.Logger import LOGGER, SHAPE_LOG
import matplotlib.pyplot as plt
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import List 
from tqdm import tqdm
from Parser import tokenizer

class LanguageModel:
    def __init__(self, model: nn.Module, vocab_size: int, device: str):
        self.model: nn.Module        = model
        self.learning_rate: float   = 0.0006
        self.criterion: nn.Module   = nn.NLLLoss()  #  negative log likelihood loss
        self.loss: torch.Tensor     = torch.Tensor([0]).to(device)
        self.optimizer: Optimizer   = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.max_sample_length: int = 10
        self.device: str            = device
        
        self.embedding = nn.Embedding(vocab_size, model.embedding_dim).to(device)

    def init_model(self, dataset):
        """Initialize the model with vocabulary from the dataset."""
        self.vocab = dataset.build_vocab()

        self.word2idx = self.vocab["word2idx"]
        self.idx2word = self.vocab["idx2word"]

        # Collect all tokenized text from dataset
        all_indices = []
        for i in range(len(dataset)):
            indexed_data = dataset[i]  # Get the dictionary returned by __getitem__
            all_indices.extend(indexed_data['input'].tolist())  # Access 'input' and convert to list

        # Convert all tokens to a tensor
        self.text_idx = torch.tensor(all_indices, dtype=torch.long, device=self.device)

        print(f"Vocabulary Size: {len(self.word2idx)}")
        print(f"Sample Indexed Text: {self.text_idx[:10]}")  # Display first few indices

    def __train(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor):
        batch_size = input_tensor.size(0)
        hidden = self.model.initHidden(batch_size)

        self.optimizer.zero_grad()
        self.model.zero_grad()
        self.loss = 0

        # Process input sequence
        embedded_input = self.embedding(input_tensor.long()).to(self.device)
        
        # Get final hidden state from processing input
        for i in range(input_tensor.size(1)):
            input_word = embedded_input[:, i, :].unsqueeze(1)
            _, hidden = self.model(input_word, hidden)

        # Now generate output sequence
        output_seq = []
        current_hidden = hidden
        
        for i in range(target_tensor.size(1) - 1):
            output, current_hidden = self.model(embedded_input[:, i, :].unsqueeze(1), current_hidden)
            output_seq.append(output)
            
            target_word = target_tensor[:, i+1].to(self.device)
            l = self.criterion(output[:, -1, :], target_word)
            self.loss += l

        self.loss.backward()
        self.optimizer.step()

        return torch.stack(output_seq, dim=1), self.loss.item() / target_tensor.size(1)
    
    def train_loop(self, dataloader: DataLoader):
        n_iters = 1500
        print_every = 500
        plot_every = 250

        # {'learning_rate': 1.4515934828819934e-05, 'n_iters': 15000, 'input_layers': 50, 'hidden_layers': 300}

        loss_agg = []
        iter_agg = []

        for iter in tqdm(range(1, n_iters + 1), desc="Training Loop", ncols=100):
            total_loss = 0
            num_batches = 0

            for batch in dataloader:
                # Access 'input' and 'output' from the batch dictionary
                self.input = batch['input'].to(self.device)  # Get input tensor
                self.target = batch['output'].to(self.device)  # Get target tensor

                # (batch_size, seq_length, vocab_size), Scalar
                output, loss = self.__train(self.input, self.target) # add tensors
                total_loss += loss
                num_batches += 1

            avg_loss = total_loss / num_batches

            if iter % plot_every == 0:
                loss_agg.append(avg_loss)
                iter_agg.append(iter)

            if iter % print_every == 0:
                print(f"loss = {avg_loss}%")

        plt.title(f"Loss Plot {str(self.model)}")
        plt.plot(iter_agg, loss_agg)
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.show()



    def sample(self, prompt: str = " ", temperature:float = 0.8):
        prompt += " <PROGRAM END>"
        with torch.no_grad():
            # Split prompt into words and convert to indices
            indices, _ = tokenizer(prompt)
            
            input = torch.tensor([indices[0]], dtype=torch.long).to(self.device)
            hidden = self.model.initHidden()

            # Process the entire prompt sequence first
            embedded_input = self.embedding(input).to(self.device)

                    # Process one token at a time to match training
            for i in range(embedded_input.size(1)):
                curr_input = embedded_input[:, i:i+1, :]  # [1, 1, embedding_dim]
                output, hidden = self.model(curr_input, hidden)

            output_text = prompt
            LOGGER(f"Starting with prompt: {output_text}")
            next_word = None
            generated_count = 0

            while next_word != '<eos>' and generated_count < self.max_sample_length: 
                # Use last output for next prediction
                output = nn.functional.softmax(output[:,-1,:] / temperature, dim=1)
                top_probs, top_indices = torch.topk(output, 10)

                topi = torch.multinomial(output, 1)[0][0].item()
                LOGGER(f"topV = {output[0][topi]} topi = {topi}")
                # Sample next token

                next_word = self.idx2word[topi]
                LOGGER(f"Generated: {next_word}")
                if next_word == '<eos>' and generated_count == 0:
                    LOGGER("Warning: Model generated <eos> immediately, check input processing.")
                    return "ERROR: Model terminated generation immediately."
                
                if next_word == '<eos>':
                    break
                
                output_text += (' ' + next_word)
                generated_count += 1

                # Prepare input for next iteration
                input = torch.tensor([[topi]], dtype=torch.long).to(self.device)
                embedded_input = self.embedding(input).to(self.device)
                output, hidden = self.model(embedded_input, hidden)

        return output_text



    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str):
        self.model.load_state_dict(torch.load(path, weights_only=True))
        self.model.eval()
        self.model.to(self.device)
        LOGGER(f"Model loaded from {path}")
        return self.model