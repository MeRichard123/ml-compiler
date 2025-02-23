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
        self.max_sample_length: int = 20
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
            indexed_text = dataset[i].tolist()  # Convert tensor to list
            all_indices.extend(indexed_text)

        # Convert all tokens to a tensor
        self.text_idx = torch.tensor(all_indices, dtype=torch.long, device=self.device)

        print(f"Vocabulary Size: {len(self.word2idx)}")
        print(f"Sample Indexed Text: {self.text_idx[:10]}")  # Display first few indices

    def __train(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor):
        # input_tensor: (batch_size, seq_length)
        # target_tensor:(batch_size, seq_length)
    
        batch_size = input_tensor.size(0)
        hidden = self.model.initHidden(batch_size)

        self.optimizer.zero_grad()
        self.model.zero_grad()
        self.loss = 0

        # [batch_size, seq_length, embedding_dim]
        embedded_input = self.embedding(input_tensor.long()).to(self.device)

        # iterate over seq_length
        for i in range(input_tensor.size(1) - 1):
            # Prepare input as a one-hot vector for each word
            input_word = embedded_input[:, i, :].unsqueeze(1) # [batch_size, 1, embedding_dim]

            SHAPE_LOG("LM.__TRAIN() INPUT SIZE", input_word)
            SHAPE_LOG("LM.__TRAIN() HIDDEN SIZE", hidden)

            # output (batch_size, seq_length, vocab_size)
            # hidden (batch_size, seq_length, hidden_size)
            output, hidden = self.model(input_word, hidden)

            target_idx = (i + 1) 
            #% target_tensor.size(0)
            SHAPE_LOG("LM.__TRAIN() Target Tensor", target_tensor)
            target_word = target_tensor[:, target_idx].to(self.device)
            SHAPE_LOG("LM.__TRAIN() Target Word", target_word)
            SHAPE_LOG("LM.__TRAIN() Output Size", output)

            # (N, vocab_size) (N)
            l = self.criterion(output[:, -1, :], target_word)
            # print("Loss from __train %d", l)
            self.loss += l
        self.loss.backward()
        self.optimizer.step()

        return output, self.loss.item() / input_tensor.size(0)
    
    def train_loop(self, dataloader: DataLoader):
        n_iters = 15000
        print_every = 500
        plot_every = 250

        # {'learning_rate': 1.4515934828819934e-05, 'n_iters': 15000, 'input_layers': 50, 'hidden_layers': 300}

        loss_agg = []
        iter_agg = []

        for iter in tqdm(range(1, n_iters + 1), desc="Training Loop"):
            total_loss = 0
            num_batches = 0

            for batch in dataloader:
                # (N, L-1) 
                self.input = batch[:, :-1].to(self.device)
                self.target = batch[:, 1:].to(self.device)

                # (batch_size, seq_length, vocab_size), Scalar
                output, loss = self.__train(self.input, self.target) # add tensors
                total_loss += loss
                num_batches += 1

            avg_loss = total_loss / num_batches

            if iter % plot_every == 0:
                loss_agg.append(avg_loss)
                iter_agg.append(iter)

            if iter % print_every == 0:
                print(f"Epoch {round(iter / n_iters * 100)}% Training, loss = {avg_loss}%")

        plt.title(f"Loss Plot {str(self.model)}")
        plt.plot(iter_agg, loss_agg)
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.show()



    def sample(self, prompt: str = " ", temperature:float = 1.0):
        prompt += " <PROGRAM END>"
        with torch.no_grad():
            # Split prompt into words and convert to indices
            prompt_words = tokenizer(prompt)[1]

            prompt_words = ["<unk>" if w not in self.vocab["vocab"] else w for w in prompt_words]
            print(prompt_words)
            
            input_indices = [self.word2idx[word] for word in prompt_words]
            input = torch.tensor([input_indices], dtype=torch.long).to(self.device)
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

            while next_word != '<eos>':
                # Use last output for next prediction
                output = nn.functional.softmax(output[:,-1,:] / temperature, dim=1)
                topi = torch.multinomial(output, 1)

                LOGGER(f"topV = {output[0][topi]} topi = {topi}")
                topi = topi[0][0].item()

                next_word = self.idx2word[topi]
                LOGGER(f"Generated: {next_word}")
                if next_word == '<eos>':
                    break
                else:
                    output_text += (' ' + next_word)

                # Prepare input for next iteration
                input = torch.tensor([[topi]], dtype=torch.long).to(self.device)
                embedded_input = self.embedding(input).to(self.device)
                output, hidden = self.model(embedded_input, hidden)

        return output_text
    

    def sample_raw(self, input: torch.Tensor, temperature:float = 1.0):
        with torch.no_grad():
            input = input.to(self.device)
            hidden = self.model.initHidden()

            # convert input tensor to the original text
            input_text = ""
            for i in input[0]:
                input_text += self.idx2word[i.item()] + " "
            LOGGER(f"Starting with prompt: {input_text}")

            # Process the entire prompt sequence first
            embedded_input = self.embedding(input).to(self.device)

                    # Process one token at a time to match training
            for i in range(embedded_input.size(1)):
                curr_input = embedded_input[:, i:i+1, :]  # [1, 1, embedding_dim]
                output, hidden = self.model(curr_input, hidden)

            output_text = ""
            next_word = None

            while next_word != '<eos>':
                # Use last output for next prediction
                output = nn.functional.softmax(output[:,-1,:] / temperature, dim=1)
                topi = torch.multinomial(output, 1)

                topi = topi[0][0].item()

                next_word = self.idx2word[topi]
                if next_word == '<eos>':
                    break
                else:
                    output_text += (' ' + next_word)

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