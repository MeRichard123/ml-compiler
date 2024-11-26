import torch
from Architectures.LSTM import LSTM
from torch import nn
from Architectures.RNN import RNN
from Utils.Logger import LOGGER
from LanguageModel import LanguageModel

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

    
    text =  "hello\nhelp\nhell\nheal\n"
    # play with the hidden size and the learning rate
        # maybe an LR scheduler
        # batch processing
        # temperature
        # Validation set

    model = RNN(1, 16, len(vocab), device).to(device)

    LM = LanguageModel(model, device)
    #LM.init_model(text, vocab)
    #LM.train_loop()

    #print(LM.sample("h"))
    # Saving the model
    #LM.save_model("RNN_model.pth")

    # Loading the model

    LM.load_model("./trained_models/RNN_model.pth")
    LM.init_model(text, vocab)
    test = LM.sample("h")
    print(test)


if __name__ == "__main__":
    main()
