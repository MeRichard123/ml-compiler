import torch 
from torch.utils.data import Dataset, random_split
from torch.nn.utils.rnn import pad_sequence
import os
from Parser import tokenizer, Tokeniser

def collate_fn(batch):
    batch = sorted(batch, key=lambda x: len(x), reverse=True)
    padded_batch = pad_sequence(batch, batch_first=True)
    return padded_batch

class CodeDataset(Dataset):
    def __init__(self, curricum_num=1):
        self.data_dir = f"./training_examples/Curriculum{curricum_num}"
        self.tokenizer = tokenizer
        self.data = os.listdir(self.data_dir)

    def build_vocab(self):
        vocab = set()
        for data_path in self.data:
            data_path = os.path.join(self.data_dir, data_path)
            with open(data_path, "r") as f:
                text = f.read()
                tokens = self.tokenizer(text)[1].keys()
                vocab.update(tokens)
        vocab = ["<eos>", "<PROGRAM END>", "<unk>"] + list(vocab)

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
                tokens = tokeniser.tokenise_code(text)
                if len(tokens) > 0:
                    prompts.append(tokens)
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

        indices, _ = self.tokenizer(text)
        return torch.tensor(indices, dtype=torch.long)

