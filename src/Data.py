import torch 
from torch.utils.data import random_split, IterableDataset
from torch.nn.utils.rnn import pad_sequence
import os
from Parser import tokenizer, Tokeniser


def collate_fn(batch):
    batch = sorted(batch, key=lambda x: len(x['input']), reverse=True)
    
    input_sequences = [x['input'] for x in batch]  # Already tensors from your dataset
    output_sequences = [x['output'] for x in batch]
    
    # Add length checks
    max_len = max(len(seq) for seq in input_sequences)
    if max_len > 10000:  # Arbitrary large number
        raise ValueError(f"Sequence too long: {max_len} tokens")
    
    padded_input = pad_sequence(input_sequences, batch_first=True, padding_value=0)
    padded_output = pad_sequence(output_sequences, batch_first=True, padding_value=0)
    
    return {'input': padded_input, 'output': padded_output}


class CodeDatasetSubset(IterableDataset):
    def __init__(self, data_dir, file_list, tokenizer):
        self.data_dir = data_dir
        self.data = file_list
        self.tokenizer = tokenizer
    
    def __iter__(self):
        for filename in self.data:
            data_path = os.path.join(self.data_dir, filename)
            with open(data_path, "r") as f:
                text = f.read().strip() + "\n <eos>"
            
            (input_indices, output_indices), _ = self.tokenizer(text)
            
            yield {
                'input': torch.tensor(input_indices, dtype=torch.long),
                'output': torch.tensor(output_indices, dtype=torch.long)
            }

class CodeDataset(IterableDataset):
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
                current_vocab = filter(lambda x: not (x in ['"', "'",'", "']), current_vocab)
                vocab.update(current_vocab)  # Update

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
        import random
        random.shuffle(self.data)
        split_idx = int(len(self.data) * (1 - test_size))
        train_files = self.data[:split_idx]
        test_files = self.data[split_idx:]
        
        train_dataset = CodeDatasetSubset(self.data_dir, train_files, self.tokenizer)
        test_dataset = CodeDatasetSubset(self.data_dir, test_files, self.tokenizer)
        
        return train_dataset, test_dataset
    
    def load_data(self):
        """ Generator function that lazily reads files and tokenizes data. """
        for filename in self.data:
            data_path = os.path.join(self.data_dir, filename)
            with open(data_path, "r") as f:
                text = f.read().strip() + "\n <eos>"

            (input_indices, output_indices), _ = self.tokenizer(text)
            
                    # Add validation
            if len(output_indices) > 1000:  # Adjust threshold as needed
                print(f"ERROR: Corrupted file '{filename}'")
                print(f"Input len: {len(input_indices)}, Output len: {len(output_indices)}")
                print(f"First 100 chars: {text[:100]}...")
                continue  # Skip this file or raise an error
            
            yield {
                'input': torch.tensor(input_indices, dtype=torch.long),
                'output': torch.tensor(output_indices, dtype=torch.long)
            }

    def __iter__(self):
        return self.load_data()

