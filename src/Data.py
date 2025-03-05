import torch 
from torch.utils.data import Dataset, random_split
from torch.nn.utils.rnn import pad_sequence
import os
from Parser import tokenizer, Tokeniser

def collate_fn(batch):
    # Sort by input length
    batch = sorted(batch, key=lambda x: len(x['input']), reverse=True)
    
    # Pad sequences
    input_sequences = [x['input'] for x in batch]
    output_sequences = [x['output'] for x in batch]
    
    padded_input = pad_sequence(input_sequences, batch_first=True)
    padded_output = pad_sequence(output_sequences, batch_first=True)
    
    return {
        'input': padded_input,
        'output': padded_output
    }

class CodeDataset(Dataset):
    def __init__(self, curricum_num=1):
        self.data_dir = f"./training_examples/Curriculum{curricum_num}"
        self.tokenizer = tokenizer
        self.data = os.listdir(self.data_dir)

    def build_vocab(self):
        vocab = set()  # Use a set to accumulate unique tokens
        # Add literals from training data
        for data_path in self.data:
            data_path = os.path.join(self.data_dir, data_path)
            with open(data_path, "r") as f:
                text = f.read()
                current_vocab = self.tokenizer(text)[1].keys()
                vocab.update(current_vocab)  # Update instead of overwrite

        vocab = sorted(list(vocab))  # Convert to sorted list for consistent indexing
        
        word2idx = {word: idx for idx, word in enumerate(vocab)}
        idx2word = {idx: word for idx, word in enumerate(vocab)}

        return {
            "vocab": vocab,
            "word2idx": word2idx,
            "idx2word": idx2word,
            "size": len(vocab)
        }
    
    def get_prompts(self):
        tokeniser = Tokeniser()
        
        prompts = []
        for data_path in self.data:
            data_path = os.path.join(self.data_dir, data_path)
            with open(data_path, "r") as f:
                text = f.read()
                if len(text) > 0:
                    prompts.append(text)
        return prompts

    def __len__(self):
        return len(self.data)

    def train_test_split(self, test_size=0.2):
        train_size = int(len(self) * (1 - test_size))
        test_size = len(self) - train_size
        return random_split(self, [train_size, test_size])

    def __getitem__(self, index):
        data_path = os.path.join(self.data_dir, self.data[index])
        with open(data_path, "r") as f:
            text = f.read()
            text += "\n <eos>"

        (input_indices, output_indices), _ = self.tokenizer(text)
        return {
            'input': torch.tensor(input_indices, dtype=torch.long),
            'output': torch.tensor(output_indices, dtype=torch.long)
        }

