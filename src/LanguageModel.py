import torch
import torch.nn as nn
from Utils.Logger import LOGGER, SHAPE_LOG
import matplotlib.pyplot as plt
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from Parser import tokenizer
from itertools import chain
import datetime
import time 
from Utils.Attention import AttentionLayer, generate_causal_mask

class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1, vocab_size=None):
        super().__init__()
        self.smoothing = smoothing
        self.vocab_size = vocab_size
    def forward(self, output, target):
        confidence = 1.0 - self.smoothing
        low_confidence = self.smoothing / (self.vocab_size - 1)
        true_dist = torch.zeros_like(output).fill_(low_confidence).scatter_(1, target.unsqueeze(1), confidence)
        return nn.KLDivLoss(reduction='batchmean')(nn.functional.log_softmax(output, dim=1), true_dist)

class LanguageModel:
    def __init__(self, model: nn.Module, vocab_size: int, device: str):
        self.model: nn.Module        = model
        self.learning_rate: float   = 9.912115401164314e-05
        self.criterion: nn.Module   = nn.CrossEntropyLoss()  #  negative log likelihood loss
        self.loss: torch.Tensor     = torch.Tensor([0]).to(device)
        self.optimizer: Optimizer   = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-3)
        self.max_sample_length: int = 2
        self.device: str            = device
        self.vocab_size: int        = vocab_size
        
        
        self.embedding = nn.Embedding(vocab_size, model.embedding_dim).to(device)
        self.attention = AttentionLayer(embed_dim=951, num_heads=3, device=device).to(device)
        self.context_projection = nn.Linear(951, model.hidden_size).to(device)  # Project 951 to 900

    def init_model(self, dataset):
        """Initialize the model with vocabulary from the dataset."""
        start = time.time()
        new_vocab = dataset.build_vocab()
        end = time.time()
        print(f"Vocabulary building took {end - start:.2f} seconds")
        self.word2idx = new_vocab["word2idx"]
        self.idx2word = new_vocab["idx2word"]
        self.criterion = LabelSmoothingLoss(smoothing=0.2, vocab_size=len(self.word2idx))


        # Collect all tokenized text from dataset
        all_indices = list(chain.from_iterable(data['input'].tolist() for data in dataset))

        # Convert all tokens to a tensor
        self.text_idx = torch.tensor(all_indices, dtype=torch.long, device=self.device)

        print(f"Vocabulary Size: {len(self.word2idx)}")


    def __train(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor, use_teacher_forcing: bool = False):
        self.model.train()
        batch_size = input_tensor.size(0)
        hidden = self.model.initHidden(batch_size)

        self.optimizer.zero_grad()
        self.loss = 0
        scheduled_teaching_prob = 1.0

        # Process input sequence
        embedded_input = self.embedding(input_tensor.long()).to(self.device)

        # Now generate output sequence
        output_seq = []
        hidden_states = []
        current_input = embedded_input[:, 0, :].unsqueeze(1)  # Start with the first token of the input sequence

        for t in range(input_tensor.size(1)):
            curr_input = embedded_input[:, t:t+1, :]
            output, hidden = self.model(curr_input, hidden)
            hidden_states.append(output[:, -1, :])

        hidden_states = torch.stack(hidden_states, dim=1)  # Shape: (batch_size, seq_length, hidden_size)
        # Apply attention mechanism
        context, attention_scores = self.attention(hidden_states)

        # Iterate over the target sequence
        for t in range(target_tensor.size(1)):
            # Generate the model's output for the current input
            output, hidden = self.model(current_input, hidden)
            context_t = context[:, t, :].unsqueeze(1)
            context_t = self.context_projection(context_t)  # Project context to match hidden size
            output = self.model.fc(context_t)
            output = self.model.softmax(output)
            output_seq.append(output)

            # Get the target word for the current timestep
            target_word = target_tensor[:, t].to(self.device)

            # Compute the loss between model output and target word
            l = self.criterion(output[:, -1, :], target_word)
            self.loss += l

            #print(f"Input: {[self.idx2word[i.item()] for i in input_tensor[0]]}")
            #print(f"Target: {[self.idx2word[i.item()] for i in target_tensor[0]]}")

            if t < target_tensor.size(1) - 1:
                if use_teacher_forcing and torch.rand(1).item() < scheduled_teaching_prob:
                    # Feed the target as the next input
                    current_input = self.embedding(target_word.long()).unsqueeze(1)
                    # decay the teaching probability
                    scheduled_teaching_prob *= 0.99
                else:
                    # Use the predicted word as the next input
                    predicted_word = torch.argmax(output[:, -1, :], dim=1)
                    current_input = self.embedding(predicted_word.long()).unsqueeze(1)

        # Backpropagate and optimize the model
        self.loss.backward()
        #nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        lr = self.optimizer.param_groups[0]['lr']
        

        return torch.stack(output_seq, dim=1), self.loss.item() / target_tensor.size(1), lr

    def __validate(self, input, target):
        self.model.eval()
        with torch.no_grad():
            batch_size = input.size(0)
            hidden = self.model.initHidden(batch_size)
            loss = 0

            # Process input sequence
            embedded_input = self.embedding(input.long()).to(self.device)
            hidden_states = []
            for t in range(input.size(1)):
                curr_input = embedded_input[:, t:t+1, :]  # [batch_size, 1, n_embed]
                output, hidden = self.model(curr_input, hidden)
                hidden_states.append(output[:, -1, :])

            hidden_states = torch.stack(hidden_states, dim=1)  # Shape: (batch_size, seq_length, hidden_size)
            # Apply attention mechanism
            context, attention_scores = self.attention(hidden_states)

            # Generate output sequence
            current_input = embedded_input[:, 0, :].unsqueeze(1)  # Start with first token
            for t in range(target.size(1)):
                output, hidden = self.model(current_input, hidden)
                context_t = context[:, t, :].unsqueeze(1)
                context_t = self.context_projection(context_t)  # Project context to match hidden size
                output = self.model.fc(context_t)
                output = self.model.softmax(output)
                target_word = target[:, t].to(self.device)
                loss += self.criterion(output[:, -1, :], target_word)

                # Use target as next input (no teacher forcing needed, but align with __train)
                if t < target.size(1) - 1:
                    predicted_word = torch.argmax(output[:, -1, :], dim=1)
                    current_input = self.embedding(predicted_word.long()).unsqueeze(1)

            return loss.item() / target.size(1)
        
    def train_loop(self, dataloader: DataLoader, suffix: str = "", validation : DataLoader = None, use_teacher_forcing: bool = False):
        n_iters = 3000
        print_every = 250
        plot_every = 250
        self.scheduler: StepLR  = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10, min_lr=1.45e-05)
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
                _, loss, lr = self.__train(self.input, self.target, use_teacher_forcing) # add tensors
                total_loss += loss
                num_batches += 1

            avg_loss = total_loss / num_batches
            perplexity = torch.exp(torch.tensor(avg_loss)).item()

            if iter % plot_every == 0:
                loss_agg.append(avg_loss)
                iter_agg.append(iter)

            self.model.eval()
            with torch.no_grad():
                if validation is not None:
                    val_loss = 0
                    for batch in validation:
                        input = batch['input'].to(self.device)
                        target = batch['output'].to(self.device)
                        loss = self.__validate(input, target)
                        val_loss += loss
                    val_loss /= len(validation)
                    val_perplexity = torch.exp(torch.tensor(val_loss)).item()

            if iter % print_every == 0:
                print(f" loss = {avg_loss:.4}%, Per = {perplexity:.4}, lr = {lr:.4}, val = {val_loss:.4}, val_per = {val_perplexity:.4}")

            self.scheduler.step(val_loss)

        plt.figure()
        plt.title(f"Loss Plot {str(self.model)}")
        plt.plot(iter_agg, loss_agg, c="blue", label="Training Loss")
        if validation is not None:
            plt.plot(iter_agg, [val_loss] * len(iter_agg), c="red", label="Validation Loss")
            plt.legend()
        else:
            plt.legend(["Training Loss"])
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        date = datetime.datetime.now().strftime("%m-%d-%Y")
        plt.savefig(f"lossPlot_{suffix}_{date}.png")

        return perplexity, val_perplexity

    def sample(self, prompt: str = " ", temperature: float = .8):
        prompt += " <PROGRAM END>"
        # print(f"Prompt: {prompt}")
        with torch.no_grad():
            (input_indices, _), (_, _) = tokenizer(prompt, self.word2idx)
            self.max_sample_length = len(input_indices)

            # log unknown tokens
            unk_idx = self.word2idx.get('<unk>', -1)
            if unk_idx == -1:
                print("[Error]: <unk> token not in Vocabulary")
                return "ERROR: <unk> token not in Vocabulary"
            
                # log the unknown tokens
                print(f"Error: <unk> tokens found in prompt: {input_indices}")
                return "ERROR: Prompt contains unknown tokens."

            input = torch.tensor(input_indices, dtype=torch.long).unsqueeze(0).to(self.device)
            hidden = self.model.initHidden()
            hidden_states = []

            print(f"Input: {[self.idx2word[i.item()] for i in input[0]]}")

            # Process the entire prompt sequence first
            embedded_input = self.embedding(input).to(self.device)

            for i in range(embedded_input.size(1)):
                curr_input = embedded_input[:, i:i+1, :]  # [1, 1, embedding_dim]
                output, hidden = self.model(curr_input, hidden)
                hidden_states.append(output[:, -1, :])

            hidden_states = torch.stack(hidden_states, dim=1)  # Shape: (batch_size, seq_length, hidden_size)
            # Apply attention mechanism
            context, attention_scores = self.attention(hidden_states)

            output_text = prompt
            LOGGER(f"Starting with prompt: {output_text}")
            next_word = None
            generated_count = 0

            while next_word != '<eos>' and generated_count < self.max_sample_length: 
                output_probs = nn.functional.softmax(output[:, -1, :] / temperature, dim=1)
                attention_weights = attention_scores[:, :, -1, -1].mean(dim=1)
                attention_weights = torch.clamp(attention_weights, min=1e-10) + 1e-10
                context_t = context[:, -1, :].unsqueeze(1)
                context_t = self.context_projection(context_t)  # Project context to match hidden size
                output = self.model.fc(context_t)
                output_probs = output_probs * attention_weights.unsqueeze(1)  # Apply attention weights
                output_probs = output_probs / torch.sum(output_probs, dim=1, keepdim=True)

                # Sample from full probability distribution
                topi = torch.multinomial(output_probs, 1).item()
                LOGGER(f"topV = {output_probs[0][topi]} topi = {topi}")

                # Sample from top-k
                _, top_indices = torch.topk(output_probs, k=5)
                top_indices = top_indices[0].tolist()
                #print([self.idx2word.get(idx, '<unk>') for idx in top_indices])
                #print("Raw logits for top 10:", torch.topk(output[:, -1, :], k=10)[0])
                #print("Logit for <eos>:", output[:, -1, self.word2idx['<eos>']].item())

                next_word = self.idx2word.get(topi, '<unk>')
                LOGGER(f"Generated: {next_word}")

                if next_word == '<eos>' and generated_count == 0:
                    LOGGER("Warning: Model generated <eos> immediately, check input processing.")
                    continue
                    #return "ERROR: Model terminated generation immediately."
                
                elif next_word == '<eos>' and generated_count > 0:
                    break

                output_text += ' ' + next_word
                generated_count += 1

                # Update input and hidden state
                input = torch.tensor([[topi]], dtype=torch.long).to(self.device)
                embedded_input = self.embedding(input).to(self.device)
                output, hidden = self.model(embedded_input, hidden)
                hidden_states = torch.cat((hidden_states, output[:, -1, :].unsqueeze(1)), dim=1)
                context, attention_scores = self.attention(hidden_states)

        return output_text
    
    def samplek(self, input_indices: torch.Tensor, num_samples: int = 1,  temperature: float = .8):
        self.model.eval()
        with torch.no_grad():
            input_indices = input_indices.to(self.device)

            hidden = self.model.initHidden()

            embedded_input = self.embedding(input_indices).to(self.device)
            for i in range(embedded_input.size(1)):
                curr_input = embedded_input[:, i:i+1, :]
                output, hidden = self.model(curr_input, hidden)
            
            candiates = []
            for _ in range(num_samples):
                generated_indices = []
                curr_hidden = hidden
                curr_output = output
                generated_count = 0
                next_idx = -1
                output_text = ""
                
                while generated_count < self.max_sample_length and next_idx != 0: 
                    output_probs = nn.functional.softmax(curr_output[:, -1, :] / temperature, dim=1)
                    # Sample from full probability distribution
                    topi = torch.multinomial(output_probs, 1).item()
                    generated_indices.append(topi)
                    LOGGER(f"topV = {output_probs[0][topi]} topi = {topi}")

                    next_idx = topi
                    next_word = self.idx2word.get(topi, '<unk>')

                    if next_word == '<eos>' and generated_count == 0:
                        LOGGER("Warning: Model generated <eos> immediately, check input processing.")
                        #return "ERROR: Model terminated generation immediately."
                    
                    if next_word == '<eos>':
                        break

                    output_text += ' ' + next_word
                    # Update input and hidden state
                    input = torch.tensor([[topi]], dtype=torch.long).to(self.device)
                    embedded_input = self.embedding(input).to(self.device)
                    curr_output, curr_hidden = self.model(embedded_input, curr_hidden)
                    generated_count += 1

                candiates.append(torch.tensor(generated_indices, dtype=torch.long).to(self.device))
        return candiates

        

    def save_model(self, path: str):
        import datetime
        if path.endswith(".pth"):
            path = path[:-4]
            
        path += datetime.datetime.now().strftime("%Y-%m-%d")
        if not path.endswith(".pth"):
            path += ".pth"

        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss,
            'vocab_size': self.vocab_size,
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'embedding': self.embedding.state_dict(),
        }
        torch.save(state, path)

    def load_model(self, path: str):
        state = torch.load(path, weights_only=True)
        self.model.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.loss = state['loss']
        self.vocab_size = state['vocab_size']
        self.word2idx = state['word2idx']
        self.idx2word = state['idx2word']
        self.embedding.load_state_dict(state['embedding'])
        self.embedding.to(self.device)
        self.model.to(self.device)
        self.model.eval()

        LOGGER(f"Model loaded from {path}")
        return self.model