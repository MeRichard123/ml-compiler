import torch
from LSTM import LSTM
from torch import nn
from RNN import RNN
import config

LOGGING = config.LOGGING

def LOGGER(string: str):
    if LOGGING:
        print(string)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    LOGGER(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
    LOGGER(f"CUDA version: {torch.version.cuda}")
        
    # Storing ID of current CUDA device
    cuda_id = torch.cuda.current_device()
    LOGGER(f"ID of current CUDA device: {torch.cuda.current_device()}")
            
    LOGGER(f"Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}")

    # Defining the model
    vocab = "heloap\n"  # Include all lowercase letters

    char2idx = {char: idx for idx, char in enumerate(vocab)}
    idx2char = {idx: char for idx, char in enumerate(vocab)}
    
    text =  "hello\nhello\nhelp\nhell\nheal\n"
    # play with the hidden size and the learning rate
        # maybe an LR scheduler
        # batch processing
        # temperature
        # Validation set
    model = RNN(1, 128, len(vocab), device).to(device)

    text_idx = torch.tensor([char2idx[char] for char in text], dtype=torch.long).to(device)

    learning_rate = 0.0005
    criterion = nn.NLLLoss()

    loss = torch.Tensor([0])

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def train(input_tensor, target_tensor):
        hidden = model.initHidden()

        optimizer.zero_grad()
        model.zero_grad()
        loss = 0

        for i in range(input_tensor.size(0)):
            # Prepare input as a one-hot vector for each character
            input_char = input_tensor[i].view(1, -1).float().to(device)
            output, hidden = model(input_char, hidden)
            
            target_idx = (i + 1) % target_tensor.size(0)
            target_char = target_tensor[target_idx].view(-1).to(device)

            l = criterion(output, target_char)
            loss += l
        loss.backward()
        optimizer.step()

        return output, loss.item() / input_tensor.size(0)

    n_iters = 10000
    total_loss = 0
    print_every = 500

    for iter in range(1, n_iters + 1):
        output, loss = train(text_idx, text_idx) # add tensors
        total_loss += loss

        if iter % print_every == 0:
            print(f"{round(iter / n_iters * 100)}% Training, loss = {loss}%")

    max_length = 100
    
    def sample(start_letter = "h", temperature=0.5):
        with torch.no_grad():
            input = torch.tensor([[char2idx[start_letter]]], dtype=torch.float32).to(device)
            hidden = model.initHidden()

            output_name = start_letter
            LOGGER(f"Starting with: {output_name}")

            for _ in range(max_length):
                output, hidden = model(input, hidden)
                # Apply temperature scaling
                output = output.div(temperature)
                LOGGER(output)
                # Sample from the scaled distribution
                output = torch.nn.functional.softmax(output, dim=1)
                topi = torch.multinomial(output, 1)
                LOGGER(f"topV = {output[0][topi]} topi = {topi}")
                topi = topi[0][0].item()
                LOGGER(f"Generated: {idx2char[topi]}")
                if topi == char2idx['\n']:
                    break 
                else:
                    letter = idx2char[topi]
                    output_name += letter 
                input = torch.tensor([[char2idx[letter]]], dtype=torch.float32).to(device)
        return output_name

    print(sample("h"))


    # Loading the model


if __name__ == "__main__":
    main()
