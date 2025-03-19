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
from Evaluation import Evaluator

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

    train_dataloader = DataLoader(
        trainset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
    )
    test_dataloader = DataLoader(testset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # print the shape of each batch
    for i, batch in enumerate(train_dataloader):
        print(f"Batch {i} shape: {batch['input'].shape}")

    vocab = code_dataset.build_vocab()["vocab"]

    # play with the hidden size and the learning rate
        # maybe an LR scheduler
        # Validation set
        # BPE
    
    MOE = {
        "number_of_experts": 10,
        "k": 3
    }

    model = RNN(50, batch_size, 300, len(vocab), device)\
        .to(device)
    model_gru = GRU(50, batch_size, 300, len(vocab), device)\
        .to(device)
    model_minGRU = minGRU(50, batch_size, 300, len(vocab), device, MOE)\
        .to(device)
    model_minLSTM = minLSTM(50, batch_size, 300, len(vocab), device)\
        .to(device)
    model_lstm = LSTM(50, batch_size, 300, len(vocab), device).\
        to(device)

    model_file_names = [
        "rnn_model_code_ast",
        "gru_model_code_ast",
        "minGRU_model_code_ast",
        "lstm_model_code_ast",
        "minLSTM_model_code_ast",
        "minGRU_code_ast_moe"
    ]



    LM = LanguageModel(model_minGRU, len(vocab), device)
    LM.init_model(code_dataset)
    LM.train_loop(train_dataloader, model_file_names[5])



    """"
    # Min LSTM - Code, AST
    print("Min LSTM - Code, AST")
    LM = LanguageModel(model_minLSTM, len(vocab), device)
    LM.load_model("./trained_models/minLSTM_model_code.pth")
    LM.init_model(code_dataset)
    print(LM.sample("print('Hello World!')"))
    """


    eval = Evaluator(testset.dataset, LM)
    eval.evaluate()


if __name__ == "__main__":
    main()