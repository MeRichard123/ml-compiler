import torch 
from torch.utils.data import random_split, IterableDataset
from torch.nn.utils.rnn import pad_sequence
import os
from Parser import tokenizer, Tokeniser
from itertools import chain
from Parser import PROMPT_TOKENS

def collate_fn(batch):
    batch = sorted(batch, key=lambda x: len(x['input']), reverse=True)
    
    input_sequences = [x['input'] for x in batch]  # Already tensors from your dataset
    output_sequences = [x['output'] for x in batch]
    
    padded_input = pad_sequence(input_sequences, batch_first=True, padding_value=0)
    padded_output = pad_sequence(output_sequences, batch_first=True, padding_value=0)
    
    return {'input': padded_input, 'output': padded_output}


class CodeDatasetSubset(IterableDataset):
    def __init__(self, data_dir, file_list, tokenizer, word2idx):
        self.data_dir = data_dir
        self.data = file_list
        self.tokenizer = tokenizer
        self.word2idx = word2idx
    
    def __iter__(self):
        for filename in self.data:
            data_path = os.path.join(self.data_dir, filename)
            with open(data_path, "r") as f:
                text = f.read().strip() + "\n <eos>"
            
            (input_indices, output_indices), _, = self.tokenizer(text, self.word2idx)
            
            yield {
                'input': torch.tensor(input_indices, dtype=torch.long),
                'output': torch.tensor(output_indices, dtype=torch.long),
            }

    def get_prompts(self):
        prompts = []
        for data_path in self.data:
            data_path = os.path.join(self.data_dir, data_path)
            with open(data_path, "r") as f:
                text = f.read()
                if len(text) > 0:
                    prompts.append(text)
        return prompts
    
    def build_vocab(self):
        t_to_i = {}
        i_to_t = {}
        all_tokens = set()
        for data_path in self.data:
            with open(os.path.join(self.data_dir, data_path), "r") as f:
                text = f.read()
                input_tokens, output_tokens, _ = Tokeniser().tokenise_code(text)
                all_tokens.update(input_tokens + output_tokens)
        all_tokens.update(PROMPT_TOKENS.get_list())
        vocab = sorted(list(all_tokens))  # Convert to sorted list for consistent indexing
        word2idx = {token: idx for idx, token in enumerate(vocab)}
        idx2word = {idx: token for token, idx in word2idx.items()}
        return {
            "vocab": vocab,
            "word2idx": word2idx,
            "idx2word": idx2word,
            "size": len(vocab)
        }

class CodeDataset(IterableDataset):
    def __init__(self, curricum_num=1):
        #self.data_dir = f"./training_examples/Curriculum{curricum_num}"
        self.data_dir = f"./training_examples/Testing"
        self.tokenizer = tokenizer
        self.data = os.listdir(self.data_dir)
        self.vocab = self.build_vocab()
        self.word2idx = self.vocab["word2idx"]

    def build_vocab(self):
        all_tokens = set()
        for data_path in self.data:
            with open(os.path.join(self.data_dir, data_path), "r") as f:
                text = f.read()
                input_tokens, output_tokens, _ = Tokeniser().tokenise_code(text)
                all_tokens.update(input_tokens + output_tokens)
        all_tokens.update(PROMPT_TOKENS.get_list())
        vocab = sorted(list(all_tokens))  # Convert to sorted list for consistent indexing
        word2idx = {token: idx for idx, token in enumerate(vocab)}
        idx2word = {idx: token for token, idx in word2idx.items()}
        return {
            "vocab": vocab,
            "word2idx": word2idx,
            "idx2word": idx2word,
            "size": len(vocab)
        }

    def __len__(self):
        return len(self.data)

    def train_test_split(self, test_size=0.2):
        import random
        random.shuffle(self.data)
        split_idx = int(len(self.data) * (1 - test_size))
        train_files = self.data[:split_idx]
        test_files = self.data[split_idx:]
        
        train_dataset = CodeDatasetSubset(
            self.data_dir, 
            train_files,
            self.tokenizer,
            self.word2idx
            )
        test_dataset = CodeDatasetSubset(
            self.data_dir,
            test_files,
            self.tokenizer,
            self.word2idx
            )
        
        return train_dataset, test_dataset
    
    def lpocv_split(self, p = 10):
        splits = []
        for i in range(len(self.data) - p + 1):
            test_file = [self.data[i:i+p]]
            train_files = self.data[:i] + self.data[i+p:]

            train_dataset = CodeDatasetSubset(
                self.data_dir, 
                train_files, 
                self.tokenizer,
                self.word2idx
                )
            test_dataset = CodeDatasetSubset(
                self.data_dir, 
                test_file, 
                self.tokenizer,
                self.word2idx
                )

            splits.append((train_dataset, test_dataset))
        return splits
    
    def load_data(self):
        """ Generator function that lazily reads files and tokenizes data. """
        for filename in self.data:
            data_path = os.path.join(self.data_dir, filename)
            with open(data_path, "r") as f:
                text = f.read().strip() + "\n <eos>"

            (input_indices, output_indices), (_,_),  = self.tokenizer(text, self.word2idx)
            
                    # Add validation
            if len(output_indices) > 1000:  # Adjust threshold as needed
                print(f"ERROR: Corrupted file '{filename}'")
                print(f"Input len: {len(input_indices)}, Output len: {len(output_indices)}")
                print(f"First 100 chars: {text[:100]}...")
                continue  # Skip this file or raise an error

            yield {
                'input': torch.tensor(input_indices, dtype=torch.long),
                'output': torch.tensor(output_indices, dtype=torch.long),
            }

    def __iter__(self):
        return self.load_data()

