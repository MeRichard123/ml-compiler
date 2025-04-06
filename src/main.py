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

def calculate_average_metrics(metrics):
    avg_metrics = {
        'perp'  : 0,
        'pass@k': 0,
        'EM'    : 0,
        'ss'    : 0
    }

    N = len(metrics)

    for metric in metrics:
        for k in avg_metrics.keys():
            avg_metrics[k] += metrics[k]
    
    for k in avg_metrics:
        avg_metrics[k] /= N
    return avg_metrics


def main_LPOCV():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    LOGGER(f"Is CUDA supported by this system? {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        LOGGER(f"CUDA version: {torch.version.cuda}")

        # Storing ID of current CUDA device
        cuda_id = torch.cuda.current_device()
        LOGGER(f"ID of current CUDA device: {torch.cuda.current_device()}")

        LOGGER(f"Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}")

    dataset = CodeDataset()
    vocab = dataset.build_vocab()["vocab"]
    batch_size = min(64, len(dataset) // 10)
    MOE = {
        "number_of_experts": 10,
        "k": 3
    }

    model = minGRU(50, batch_size, 500, len(vocab), device, MOE).to(device)

    splits = dataset.lpocv_split(p = 10)
    all_metrics = []

    for fold_idx, (train, test) in enumerate(splits):
        LOGGER(f"Training Fold {fold_idx + 1}/{len(splits)}")

        train_dataloader = DataLoader(
            train, 
            batch_size=batch_size,
            collate_fn=collate_fn
        )

        LM = LanguageModel(model, len(vocab), device)
        LM.init_model(dataset)
        LM.train_loop(train_dataloader, f"LPOCV_fold_{fold_idx + 1}")

        eval = Evaluator(test, LM)
        metrics = eval.evaluate(log=False)
        all_metrics.append(metrics)
    
    avg_metrics = calculate_average_metrics(all_metrics)
    LOGGER("Average metrics across all folds:")
    LOGGER(f"Average Perplexity: {avg_metrics['perp']:.4f}")
    LOGGER(f"Average Pass@k: {avg_metrics['p@k']:.4f}")
    LOGGER(f"Average Exact Match: {avg_metrics['EM']:.4f}")
    LOGGER(f"Average Sentence Similarity: {avg_metrics['ss']:.4f}")



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
    vocab = code_dataset.build_vocab()["vocab"]

    batch_size = min(64, len(code_dataset) // 20)
    trainset, testset = code_dataset.train_test_split()

    train_dataloader = DataLoader(
        trainset, 
        batch_size=batch_size,  
        collate_fn=collate_fn,
    )

    #print(vocab)

    # play with the hidden size and the learning rate
        # maybe an LR scheduler
        # Validation set
        # BPE
    
    MOE = {
        "number_of_experts": 12
    }

    model = RNN(50, batch_size, 300, len(vocab), device)\
        .to(device)
    model_gru = GRU(50, batch_size, 300, len(vocab), device)\
        .to(device)
    model_minGRU = minGRU(50, batch_size, 200, len(vocab), device, MOE, True)\
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
        "minGRU_code_ast_moe_attn",
        "minGRU_code_ast_moe_tf",
        "Testing_Model_Fast",
        "Attention_Model",
    ]



    LM = LanguageModel(model_minGRU, len(vocab), device) 
    
    LM.init_model(code_dataset)
    
    LM.train_loop(train_dataloader, model_file_names[6])
    LM.save_model(model_file_names[8])



    """"
    # Min LSTM - Code, AST
    print("Min LSTM - Code, AST")
    LM = LanguageModel(model_minLSTM, len(vocab), device)
    LM.load_model("./trained_models/minLSTM_model_code.pth")
    LM.init_model(code_dataset)
    print(LM.sample("print('Hello World!')"))

    LM.load_model("./trained_models/minGRU_attention.pth")
    LM.init_model(code_dataset)
    """
    #LM.load_model("./Testing_Model_Fast.pth")
    #LM.init_model(code_dataset)
    
    #eval = Evaluator(testset, LM)
    #eval.evaluate()

    print(LM.sample("print('Xenophile')"))
    print(LM.sample("print('Fandango')"))

if __name__ == "__main__":
    main()