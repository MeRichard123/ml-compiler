import torch
import torch.nn as nn
from Utils.Logger import LOGGER, SHAPE_LOG
import matplotlib.pyplot as plt
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import List 
from tqdm import tqdm
from Parser import tokenizer
from itertools import chain

class LanguageModel:
    def __init__(self, model: nn.Module, vocab_size: int, device: str):
        self.model: nn.Module        = model
        self.learning_rate: float   = 0.0006
        self.criterion: nn.Module   = nn.NLLLoss()  #  negative log likelihood loss
        self.loss: torch.Tensor     = torch.Tensor([0]).to(device)
        self.optimizer: Optimizer   = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.max_sample_length: int = 2
        self.device: str            = device
        self.vocab_size: int        = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, model.embedding_dim).to(device)

    def init_model(self, dataset):
        """Initialize the model with vocabulary from the dataset."""
        self.vocab = dataset.build_vocab()

        self.word2idx = self.vocab["word2idx"]
        self.idx2word = self.vocab["idx2word"]

        # Collect all tokenized text from dataset
        all_indices = list(chain.from_iterable(data['input'].tolist() for data in dataset))

        # Convert all tokens to a tensor
        self.text_idx = torch.tensor(all_indices, dtype=torch.long, device=self.device)

        print(f"Vocabulary Size: {len(self.word2idx)}")

    def __train(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor, use_teacher_forcing: bool = False):
        batch_size = input_tensor.size(0)
        hidden = self.model.initHidden(batch_size)

        self.optimizer.zero_grad()
        self.loss = 0

        # Process input sequence
        embedded_input = self.embedding(input_tensor.long()).to(self.device)

        for t in range(input_tensor.size(1)):
            curr_input = embedded_input[:, t:t+1, :]
            output, hidden = self.model(curr_input, hidden)

        # Now generate output sequence
        output_seq = []
        current_input = embedded_input[:, 0, :].unsqueeze(1)  # Start with the first token of the input sequence

        # Iterate over the target sequence
        for t in range(target_tensor.size(1) - 1):  # Exclude the last token for teacher forcing
            # Generate the model's output for the current input
            output, hidden = self.model(current_input, hidden)
            output_seq.append(output)

            # Get the target word for the current timestep
            target_word = target_tensor[:, t].to(self.device)

            # decode the target word
            target_tokens = [self.idx2word[idx.item()] for idx in target_word]
            predicted_word = torch.argmax(output[:, -1, :], dim=1)
            predicted_tokens = [self.idx2word[idx.item()] for idx in predicted_word]

            # Compute the loss between model output and target word
            l = self.criterion(output[:, -1, :], target_word)
            self.loss += l

            if use_teacher_forcing:
                # Feed the target as the next input
                current_input = self.embedding(target_word.long()).unsqueeze(1)
            else:
                # Use the predicted word as the next input
                predicted_word = torch.argmax(output[:, -1, :], dim=1)
                current_input = self.embedding(predicted_word.long()).unsqueeze(1)

        # Backpropagate and optimize the model
        self.loss.backward()
        self.optimizer.step()

        return torch.stack(output_seq, dim=1), self.loss.item() / target_tensor.size(1)
        
    def train_loop(self, dataloader: DataLoader, suffix: str = "", use_teacher_forcing: bool = False):
        n_iters = 1250
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
                _, loss = self.__train(self.input, self.target, use_teacher_forcing) # add tensors
                total_loss += loss
                num_batches += 1

            avg_loss = total_loss / num_batches

            if iter % plot_every == 0:
                loss_agg.append(avg_loss)
                iter_agg.append(iter)

            if iter % print_every == 0:
                print(f"loss = {avg_loss}%")

        plt.figure()
        plt.title(f"Loss Plot {str(self.model)}")
        plt.plot(iter_agg, loss_agg)
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.savefig(f"{str(self.model)}_lossPlot_{suffix}.png")



    def sample(self, prompt: str = " ", temperature: float = 1):
        prompt += " <PROGRAM END>"
        print(f"Prompt: {prompt}")
        with torch.no_grad():
            (input_indices, _), (_, _) = tokenizer(prompt, self.word2idx)
            #print(f"Input Indices: {input_indices}")
            #print([i_to_t[idx] for idx in input_indices])

            input = torch.tensor(input_indices, dtype=torch.long).unsqueeze(0).to(self.device)
            hidden = self.model.initHidden()

            # decodce the input
            #input_tokens = [i_to_t[idx] for idx in input_indices]
            #print(f"Input Tokens: {input_tokens}")

            # Process the entire prompt sequence first
            embedded_input = self.embedding(input).to(self.device)

            for i in range(embedded_input.size(1)):
                curr_input = embedded_input[:, i:i+1, :]  # [1, 1, embedding_dim]
                output, hidden = self.model(curr_input, hidden)

            output_text = prompt
            LOGGER(f"Starting with prompt: {output_text}")
            next_word = None
            generated_count = 0

            while next_word != '<eos>' and generated_count < self.max_sample_length: 
                output_probs = nn.functional.softmax(output[:, -1, :] / temperature, dim=1)
                topprobs = torch.topk(output_probs, 10, dim=1)

                # Sample from full probability distribution
                topi = torch.multinomial(output_probs, 1).item()
                LOGGER(f"topV = {output_probs[0][topi]} topi = {topi}")

                next_word = self.idx2word[topi]
                LOGGER(f"Generated: {next_word}")

                if next_word == '<eos>' and generated_count == 0:
                    LOGGER("Warning: Model generated <eos> immediately, check input processing.")
                    return "ERROR: Model terminated generation immediately."
                
                if next_word == '<eos>':
                    break

                output_text += ' ' + next_word
                generated_count += 1

                # Update input and hidden state
                input = torch.tensor([[topi]], dtype=torch.long).to(self.device)
                embedded_input = self.embedding(input).to(self.device)
                output, hidden = self.model(embedded_input, hidden)

        return output_text

    def save_model(self, path: str):
        if not path.endswith(".pth"):
            path += ".pth"
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str):
        self.model.load_state_dict(torch.load(path, weights_only=True))
        self.model.eval()
        self.model.to(self.device)
        LOGGER(f"Model loaded from {path}")
        return self.model