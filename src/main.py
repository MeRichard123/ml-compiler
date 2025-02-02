import torch
from torch.utils.data import DataLoader

from Architectures.LSTM import LSTM
from Architectures.RNN import RNN
from Architectures.GRU import GRU

from Utils.Logger import LOGGER
from LanguageModel import LanguageModel
from Data import collate_fn, CodeDataset

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    LOGGER(f"Is CUDA supported by this system? {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        LOGGER(f"CUDA version: {torch.version.cuda}")

        # Storing ID of current CUDA device
        cuda_id = torch.cuda.current_device()
        LOGGER(f"ID of current CUDA device: {torch.cuda.current_device()}")

        LOGGER(f"Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}")


    # Load the Data
    code_dataset = CodeDataset()
    batch_size = len(code_dataset) // 9
    trainset, testset = code_dataset.train_test_split()

    train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    text = """
    print("Hello World")

    <PROGRAM END>

    Hello World
    """


    text = text.split(" ") + ["<eos>"]
    vocab = list(set(text))  

    print(vocab)
    # play with the hidden size and the learning rate
        # maybe an LR scheduler
        # batch processing
        # temperature
        # Validation set
        # BPE

    model = RNN(10, 100, len(vocab), device).to(device)
    # The of of then and The Rabbit having she twice to and it of the for Oh and into or nor
    model_gru = GRU(10, 100, len(vocab), device).to(device)
    # The dear I shall be worth getting a was and look that she to was had once of day after and

    model_lstm = LSTM(10, 100, 10, len(vocab), device).to(device)
    # The was beginning to get very tired of sitting by her sister the on to and having having nothing do once


    LM = LanguageModel(model_gru, len(vocab), device)
    LM.init_model(text, vocab)
    LM.train_loop()

    print(LM.sample())
    # Saving the model
    #LM.save_model("LSTM_model.pth")

    # Loading the model

    #LM.load_model("./trained_models/LSTM_model.pth")
    #LM.init_model(text, vocab)
    #test = LM.sample("So")
    #print(test)


if __name__ == "__main__":
    main()
