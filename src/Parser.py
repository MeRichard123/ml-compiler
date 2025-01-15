import torch.nn as nn
import torch

class Tokeniser:
    def __init__(self):
        self.current = 0

    def tokenise_word(self, text):
        quotes = ['"', "'"]
        for quote in quotes:
            if quote in text:
                text = text.replace(quote, "")
        return text.split(" ")
    
    def tokenise_code(self, text):
        symbols = '()[]{}.,'
        token = ""
        tokens = []
        stringClosed = False

        while self.current < len(text):
            char = text[self.current]
            self.current += 1

            if char in ['"', "'"] and not stringClosed:
                token = ""
                endSymbol = char
                token += char

                for c in self.adance_until(text, endSymbol):
                    token += c
                token += endSymbol
                tokens.append(self.tokenise_word(token))
                token = ""
                stringClosed = True

            elif char in symbols:
                if token != "''":
                    tokens.append(token)
                tokens.append(char)
                token = ""
            
            token += char
        return tokens

    def adance_until(self, text, endSymbol):
        while self.current < len(text) and text[self.current] != endSymbol:
            yield text[self.current]
            self.current += 1

    def display_tokens(self, tokens, prefix=" "):
        print(prefix + "[")
        for token in tokens:
            if isinstance(token, list):
                self.display_tokens(token, prefix + "   ")
            else:
                print((prefix*2) + "<" + token + ">")
        print(prefix + "]")

    def tokens_to_index(self, tokens):
        indices = {}
        for token in tokens:
            if isinstance(token, list):
                indices.update(self.tokens_to_index(token))
            else:
                if token not in indices:
                    indices[token] = len(indices)
        return indices
    

if __name__ == "__main__":
    text = "print('Hello, World!')"
    tokeniser = Tokeniser()
    tokens = tokeniser.tokenise_code(text)
    t_to_i = tokeniser.tokens_to_index(tokens)

    indices = []
    for token in tokens:
        if isinstance(token, list):
            indices.extend([t_to_i[t] for t in token])
        else:
            indices.append(t_to_i[token])

    idx_tensor = torch.tensor(indices)
    vocab_size = len(t_to_i)
    embedding = nn.Embedding(vocab_size, 128)
    embedded = embedding(idx_tensor)
    print(embedded)