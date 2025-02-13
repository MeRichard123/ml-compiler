import optuna
import torch
import torch.nn as nn
from Utils.Logger import LOGGER
from Architectures.GRU import GRU

import torch
import torch.nn as nn
from Utils.Logger import LOGGER, SHAPE_LOG
import matplotlib.pyplot as plt
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import List 
from Parser import tokenizer
from Data import collate_fn, CodeDataset


optuna.logging.set_verbosity(optuna.logging.DEBUG)


class LanguageModelOptuna:
    def __init__(self, batch_size:int, vocab_size: int, device: str):
        self.criterion: nn.Module   = nn.NLLLoss()  #  negative log likelihood loss
        self.loss: torch.Tensor     = torch.Tensor([0]).to(device)
        self.max_sample_length: int = 20
        self.device: str            = device
        self.vocab_size             = vocab_size
        self.batch_size             = batch_size    
    

    def set_optimiser(self, lr, model):
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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
            l = self.criterion(output.squeeze(1), target_word)
            # print("Loss from __train %d", l)
            self.loss += l
        self.loss.backward()
        self.optimizer.step()

        return output, self.loss.item() / input_tensor.size(0)
    
    def train_loop(self, dataloader: DataLoader, n_iters=10000, learning_rate=0.0005, input_layers=50, hidden_layers=300):
        self.model = GRU(input_layers, batch_size, hidden_layers, self.vocab_size, self.device).to(self.device)
        self.embedding = nn.Embedding(self.vocab_size, self.model.embedding_dim).to(self.device)
        self.set_optimiser(learning_rate, self.model)
        print_every = 500

        for iter in range(1, n_iters + 1):
            total_loss = 0
            num_batches = 0

            for batch in dataloader:
                # (N, L-1) 
                batch = batch.unsqueeze(0) if batch.dim() == 1 else batch
                self.input = batch[:, :-1].to(self.device)
                self.target = batch[:, 1:].to(self.device)

                # (batch_size, seq_length, vocab_size), Scalar
                output, loss = self.__train(self.input, self.target) # add tensors
                total_loss += loss
                num_batches += 1

            avg_loss = total_loss / num_batches

            if iter % print_every == 0:
                print(f"Epoch {round(iter / n_iters * 100)}% Training, loss = {avg_loss}%")
        return avg_loss


    def sample(self, prompt: str = " ", temperature:float = 0.5):
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
            output, hidden = self.model(embedded_input, hidden)

            output_text = prompt
            LOGGER(f"Starting with prompt: {output_text}")

            for _ in range(self.max_sample_length):
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

    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str):
        self.model.load_state_dict(torch.load(path, weights_only=True))
        self.model.eval()
        self.model.to(self.device)
        LOGGER(f"Model loaded from {path}")
        return self.model
    

def objective(trial, device, dataset, vocab):
    # Create model instance
    lm = LanguageModelOptuna(batch_size, len(vocab), device)
    lm.init_model(dataset)

    # Define hyperparameters to tune
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    n_iters = trial.suggest_int('n_iters', 1000, 20000, step=1000)
    input_layers = trial.suggest_int('input_layers', 50, 350, step=50)
    hidden_layers = trial.suggest_int('hidden_layers', 200, 400, step=50)

    # Train with suggested hyperparameters
    try:
        avg_loss = lm.train_loop(dataset, n_iters=n_iters, learning_rate=learning_rate, input_layers=input_layers, hidden_layers=hidden_layers)
        return avg_loss
    except Exception as e:
        LOGGER(f"Trial failed with error: {str(e)}")
        raise optuna.TrialPruned() 


def tune_hyperparameters(device, dataset, vocab, n_trials=10):
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, device, dataset, vocab), 
                n_trials=n_trials)

    LOGGER('Best trial:')
    trial = study.best_trial

    LOGGER(f'  Value: {trial.value}')
    LOGGER('  Params: ')
    for key, value in trial.params.items():
        LOGGER(f'    {key}: {value}')

    return trial.params

# Load the Data
code_dataset = CodeDataset()

batch_size = len(code_dataset) // 9
trainset, testset = code_dataset.train_test_split()

train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab = code_dataset.build_vocab()["vocab"]


LM = LanguageModelOptuna(batch_size, len(vocab), device)
LM.init_model(code_dataset)

best_params = tune_hyperparameters(device, code_dataset, vocab, n_trials=100)
print(best_params)

