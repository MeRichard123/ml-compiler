import torch
from torch.utils.data import DataLoader

from Architectures.LSTM import LSTM
from Architectures.RNN import RNN
from Architectures.GRU import GRU
from Architectures.minGRU import minGRU
from Architectures.minLSTM import minLSTM

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

    batch_size = len(code_dataset) // 3
    trainset, testset = code_dataset.train_test_split()

    train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(testset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # print the shape of each batch
    for i, batch in enumerate(train_dataloader):
        print(f"Batch {i} shape: {batch.shape}")

    vocab = code_dataset.build_vocab()["vocab"]

    # play with the hidden size and the learning rate
        # maybe an LR scheduler
        # batch processing
        # temperature
        # Validation set
        # BPE

    model = RNN(0, 100, len(vocab), device).to(device)
    # The of of then and The Rabbit having she twice to and it of the for Oh and into or nor
    model_gru = GRU(50, batch_size, 300, len(vocab), device).to(device)
    # The dear I shall be worth getting a was and look that she to was had once of day after and
    model_minGRU = minGRU(50, batch_size, 300, len(vocab), device).to(device)
    model_minLSTM = minLSTM(50, batch_size, 300, len(vocab), device).to(device)

    model_lstm = LSTM(10, 100, 10, len(vocab), device).to(device)
    # The was beginning to get very tired of sitting by her sister the on to and having having nothing do once


    LM = LanguageModel(model_minGRU, len(vocab), device)
    LM.init_model(code_dataset)
    LM.train_loop(train_dataloader)

    # print(LM.sample("print('Oliver')"))
    # Saving the model
    
    LM.save_model("minGRU_model.pth")

    # Loading the model

    #LM.load_model("./trained_models/minLSTM_model.pth")
    #LM.init_model(code_dataset)
    #test = LM.sample("So")
    #print(LM.sample('print("Hello World")'))
    #print(test)

    for prompt in testset.dataset.get_prompts():
        print(prompt)
        idx = prompt.index("<PROGRAM END>")
        prompt = ' '.join(prompt[0: idx])
        print(LM.sample(prompt))
        print("\n\n\n")

if __name__ == "__main__":
    main()