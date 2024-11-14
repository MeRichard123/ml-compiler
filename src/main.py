import torch
from LSTM import LSTM
from torch import nn
from RNN import RNN

LOGGING = True

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
    vocab = "helo\n"
    char2idx = {char: idx for idx, char in enumerate(vocab)}
    idx2char = {idx: char for idx, char in enumerate(vocab)}
    
    text =  "hello\n"
    model = RNN(1, 128, len(vocab), device).to(device)

    text_idx = torch.tensor([char2idx[char] for char in text], dtype=torch.long)
    text_idx.to(device)

    learning_rate = 0.0005
    criterion = nn.NLLLoss()

    loss = torch.Tensor([0])

    def train(input_tensor, target_tensor):
        hidden = model.initHidden()

        model.zero_grad()
        loss = 0

        for i in range(input_tensor.size(0)):
            # Prepare input as a one-hot vector for each character
            input_char = input_tensor[i].view(1, -1).float().to(device)
            
            output, hidden = model(input_char, hidden)
            if i == target_tensor.size(0) - 1:
                i-=1
            target_char = target_tensor[i].view(-1).to(device)
            l = criterion(output, target_char)
            loss += l
        loss.backward()

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
    
    def sample(start_letter = "h"):
        with torch.no_grad():
            input = torch.tensor([[char2idx[start_letter]]], dtype=torch.float32).to(device)
            hidden = model.initHidden()

            output_name = start_letter
            LOGGER(f"Starting with: {output_name}")

            for _ in range(max_length):
                output, hidden = model(input, hidden)
                topv, topi = output.topk(1)
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
