import optuna
import torch
import torch.nn as nn
from Utils.Logger import LOGGER
from Architectures.minGRU import minGRU

import torch
import torch.nn as nn
from Utils.Logger import LOGGER, SHAPE_LOG
from torch.utils.data import DataLoader
from Parser import tokenizer
from Data import collate_fn, CodeDataset
from itertools import chain
from tqdm import tqdm


optuna.logging.set_verbosity(optuna.logging.DEBUG)


class LanguageModelOptuna:
    def __init__(self, batch_size:int, vocab_size: int, device: str):
        self.criterion: nn.Module   = nn.CrossEntropyLoss()  #  negative log likelihood loss
        self.loss: torch.Tensor     = torch.Tensor([0]).to(device)
        self.max_sample_length: int = 10
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
        # Now generate output sequence
        output_seq = []

        current_input = embedded_input[:, 0, :].unsqueeze(1)  # Start with the first token of the input sequence

        for t in range(input_tensor.size(1)):
            curr_input = embedded_input[:, t:t+1, :]
            _, hidden = self.model(curr_input, hidden)

        # Iterate over the target sequence
        for t in range(target_tensor.size(1)):
            # Generate the model's output for the current input
            output, hidden = self.model(current_input, hidden)
            output_seq.append(output)

            # Get the target word for the current timestep
            target_word = target_tensor[:, t].to(self.device)

            # Compute the loss between model output and target word
            l = self.criterion(output[:, -1, :], target_word)
            self.loss += l

            #print(f"Input: {[self.idx2word[i.item()] for i in input_tensor[0]]}")
            #print(f"Target: {[self.idx2word[i.item()] for i in target_tensor[0]]}")

            if t < target_tensor.size(1) - 1:
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

        avg_loss = self.loss.item() / target_tensor.size(1)
        return torch.stack(output_seq, dim=1), avg_loss
        
    
    def train_loop(self, dataloader: DataLoader, n_iters=10000, learning_rate=0.0005, input_layers=50, hidden_layers=300):
        self.model = minGRU(input_layers, self.batch_size, hidden_layers, self.vocab_size, self.device).to(self.device)
        self.embedding = nn.Embedding(self.vocab_size, self.model.embedding_dim).to(self.device)
        self.set_optimiser(learning_rate, self.model)
        print_every = 500

        for iter in tqdm(range(1, n_iters + 1), desc="Training Loop", ncols=100):
            total_loss = 0
            num_batches = 0

            for batch in dataloader:
                # Access 'input' and 'output' from the batch dictionary
                self.input = batch['input'].to(self.device)  # Get input tensor
                self.target = batch['output'].to(self.device)  # Get target tensor

                # (batch_size, seq_length, vocab_size), Scalar
                _, loss = self.__train(self.input, self.target, use_teacher_forcing = False) # add tensors
                total_loss += loss
                num_batches += 1

            avg_loss = total_loss / num_batches

        return avg_loss


    def sample(self, prompt: str = " ", temperature: float = 1):
        prompt += " <PROGRAM END>"
        # print(f"Prompt: {prompt}")
        with torch.no_grad():
            (input_indices, _), (_, _) = tokenizer(prompt, self.word2idx)

            # log unknown tokens
            unk_idx = self.word2idx.get('<unk>', -1)
            if unk_idx == -1:
                print("[Error]: <unk> token not in Vocabulary")
                return "ERROR: <unk> token not in Vocabulary"
            
            if input_indices.count(unk_idx) > 0:
                # log the unknown tokens
                print(f"Error: <unk> tokens found in prompt: {input_indices}")
                return "ERROR: Prompt contains unknown tokens."

            input = torch.tensor(input_indices, dtype=torch.long).unsqueeze(0).to(self.device)
            hidden = self.model.initHidden()

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

                # Sample from full probability distribution
                topi = torch.multinomial(output_probs, 1).item()
                LOGGER(f"topV = {output_probs[0][topi]} topi = {topi}")

                next_word = self.idx2word.get(topi, '<unk>')
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
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str):
        self.model.load_state_dict(torch.load(path, weights_only=True))
        self.model.eval()
        self.model.to(self.device)
        LOGGER(f"Model loaded from {path}")
        return self.model
    

def objective(trial, device, train_dataloader, vocab):  # Change dataset to train_dataloader
    # Create model instance
    lm = LanguageModelOptuna(batch_size, len(vocab), device)
    lm.init_model(train_dataloader.dataset)  # Use dataset from DataLoader

    # Define hyperparameters to tune
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    n_iters = trial.suggest_int('n_iters', 1000, 4000, step=1000)
    input_layers = trial.suggest_int('input_layers', 50, 500, step=50)
    hidden_layers = trial.suggest_int('hidden_layers', 200, 600, step=50)

    # Train with suggested hyperparameters
    try:
        avg_loss = lm.train_loop(train_dataloader, n_iters=n_iters, learning_rate=learning_rate, input_layers=input_layers, hidden_layers=hidden_layers)
        return avg_loss
    except Exception as e:
        print(f"Trial failed with error: {str(e)}")
        raise e
        raise optuna.TrialPruned()

def tune_hyperparameters(device, train_dataloader, vocab, n_trials=10):  # Update to use train_dataloader
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, device, train_dataloader, vocab), n_trials=n_trials)

    LOGGER('Best trial:')
    trial = study.best_trial

    LOGGER(f'  Value: {trial.value}')
    LOGGER('  Params: ')
    for key, value in trial.params.items():
        LOGGER(f'    {key}: {value}')

    return trial.params

# Load the Data
code_dataset = CodeDataset()
vocab = code_dataset.build_vocab()["vocab"]

batch_size = min(64, len(code_dataset) // 20)
trainset, testset = code_dataset.train_test_split()

train_dataloader = DataLoader(
    trainset, 
    batch_size=batch_size,  
    collate_fn=collate_fn,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LM = LanguageModelOptuna(batch_size, len(vocab), device)
LM.init_model(code_dataset)

best_params = tune_hyperparameters(device, train_dataloader, vocab, n_trials=100)
print(best_params)
