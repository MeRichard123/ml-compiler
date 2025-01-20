import torch
import torch.nn as nn
from Utils.Logger import LOGGER
import matplotlib.pyplot as plt

class LanguageModel:
    def __init__(self, model: nn.Module, vocab_size: int, device: str):
        self.model = model
        self.learning_rate = 0.0006
        self.criterion = nn.NLLLoss()  #  negative log likelihood loss
        self.loss = torch.Tensor([0]).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.max_sample_length = 20
        self.device = device

        self.embedding = nn.Embedding(vocab_size, model.embedding_dim).to(device)

    def init_model(self, text, vocab):
        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2word = {idx: word for idx, word in enumerate(vocab)}
        self.text_idx = torch.tensor([self.word2idx[word] for word in text], dtype=torch.long, device=self.device)

    def tokenize(self, text):
        return text.split(" ") + ["<eos>"]
    
    def __train(self, input_tensor, target_tensor):
        hidden = self.model.initHidden()

        self.optimizer.zero_grad()
        self.model.zero_grad()
        self.loss = 0

        embedded_input = self.embedding(input_tensor.long()).to(self.device)

        for i in range(input_tensor.size(0)):
            # Prepare input as a one-hot vector for each word
            input_word = embedded_input[i].view(1, -1).float().to(self.device)
            output, hidden = self.model(input_word, hidden)

            target_idx = (i + 1) % target_tensor.size(0)
            target_word = target_tensor[target_idx].view(-1).to(self.device)

            l = self.criterion(output, target_word)
            self.loss += l
        self.loss.backward()
        #torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        #nn.utils.clip_grad_norm_(self.model.parameters(), 1)
        self.optimizer.step()

        return output, self.loss.item() / input_tensor.size(0)
    
    def train_loop(self):
        n_iters = 10000
        total_loss = 0
        print_every = 500
        plot_every = 250

        loss_agg = []
        iter_agg = []

        for iter in range(1, n_iters + 1):
            output, loss = self.__train(self.text_idx, self.text_idx) # add tensors
            total_loss += loss

            if iter % plot_every == 0:
                loss_agg.append(loss)
                iter_agg.append(iter)

            if iter % print_every == 0:
                print(f"{round(iter / n_iters * 100)}% Training, loss = {loss}%")

        plt.title(f"Loss Plot {str(self.model)}")
        plt.plot(iter_agg, loss_agg)
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.show()



    def sample(self, start_word = "hello", temperature=0.5):
        with torch.no_grad():
            input = torch.tensor([[self.word2idx[start_word]]], dtype=torch.long).to(self.device)
            hidden = self.model.initHidden()

            output_text = start_word
            LOGGER(f"Starting with: {output_text}")

            for _ in range(self.max_sample_length):
                embedded_input = self.embedding(input).view(1, -1).to(self.device)
                output, hidden = self.model(embedded_input, hidden)

                # Sample from the scaled distribution
                output = nn.functional.softmax(output, dim=1)
                output = output.div(temperature)
                topi = torch.multinomial(output, 1)
                LOGGER(f"topV = {output[0][topi]} topi = {topi}")
                topi = topi[0][0].item()

                next_word = self.idx2word[topi]
                LOGGER(f"Generated: {next_word}")
                if next_word == '<eos>':
                    break
                else:
                    output_text += (' ' + next_word)
                input = torch.tensor([[topi]], dtype=torch.long).to(self.device)
        return output_text

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, weights_only=True))
        self.model.eval()
        self.model.to(self.device)
        LOGGER(f"Model loaded from {path}")
        return self.model