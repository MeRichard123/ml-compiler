import torch
from Architectures.LSTM import LSTM
from Architectures.RNN import RNN
from Architectures.GRU import GRU

from Utils.Logger import LOGGER
from LanguageModel import LanguageModel

text = """
Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into 
the book her sister was reading, but it had no pictures or conversations in it, “and what is the use of a book,” thought Alice “without 
pictures or conversations?”
So she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid), whether the pleasure 
of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran 
close by her.
There was nothing so very remarkable in that; nor did Alice think it so very much out of the way to hear the Rabbit say to itself, “Oh dear! 
Oh dear! I shall be late!” (when she thought it over afterwards, it occurred to her that she ought to have wondered at this, but at the time
it all seemed quite natural); but when the Rabbit actually took a watch out of its waistcoat-pocket, and looked at it, and then hurried on, 
Alice started to her feet, for it flashed across her mind that she had never before seen a rabbit with either a waistcoat-pocket, or a watch 
to take out of it, and burning with curiosity, she ran across the field after it, and fortunately was just in time to see it pop down a large 
rabbit-hole under the hedge.
In another moment down went Alice after it, never once considering how in the world she was to get out again.
The rabbit-hole went straight on like a tunnel for some way, and then dipped suddenly down, so suddenly that Alice had not a moment to think 
about stopping herself before she found herself falling down a very deep well.
Either the well was very deep, or she fell very slowly, for she had plenty of time as she went down to look about her and to wonder what was 
going to happen next. First, she tried to look down and make out what she was coming to, but it was too dark to see anything; then she looked 
at the sides of the well, and noticed that they were filled with cupboards and book-shelves; here and there she saw maps and pictures hung 
upon pegs. She took down a jar from one of the shelves as she passed; it was labelled “ORANGE MARMALADE”, but to her great disappointment it 
was empty: she did not like to drop the jar for fear of killing somebody underneath, so managed to put it into one of the cupboards as she 
fell past it.
"""
import re
text = text.replace("\n", " ").replace("\r", " ")
text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
text = re.sub(r"\s+", " ", text).strip()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    LOGGER(f"Is CUDA supported by this system? {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        LOGGER(f"CUDA version: {torch.version.cuda}")

        # Storing ID of current CUDA device
        cuda_id = torch.cuda.current_device()
        LOGGER(f"ID of current CUDA device: {torch.cuda.current_device()}")

        LOGGER(f"Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}")

    vocab = text.split(" ") + ["<eos>"]
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


    LM = LanguageModel(model_lstm, len(vocab), device)
    LM.init_model(text.split(), vocab)
    LM.train_loop()

    print(LM.sample("The"))
    # Saving the model
    LM.save_model("LSTM_model.pth")

    # Loading the model

    #LM.load_model("./trained_models/RNN_model.pth")
    #LM.init_model(text, vocab)
    #test = LM.sample("h")
    #print(test)


if __name__ == "__main__":
    main()
