import torch.nn as nn
import torch
import string

Keywords = ["function", "for", "do", "if", "then", "elseif", "else", "end", "return", "while", "repeat", "until", "break", "local", "nil", "true", "false", "and", "or", "not", "in"]
Operators = ["+", "-", "*", "/", "%", "^", "#", "==", "~=", "<=", ">=", "<", ">", "="]
Symbols = ["(", ")", "[", "]", "{", "}", ",", ".", "..", ";", ":", "..."]
prompt_tokens = ['<PROGRAM END>', '<eos>']


class Tokeniser:
    def __init__(self):
        self.current = 0

    def tokenise_string(self, text):
        quotes = ['"', "'"]
        for quote in quotes:
            if quote in text:
                text = text.replace(quote, "")
        return ' '.join(text.split(" "))
    
    def tokenise_code(self, text):
        token = ""
        tokens = []
        stringClosed = False
        text = ' '.join(text.lstrip().rstrip() for text in text.split("\n"))

        while self.current < len(text):
            char = text[self.current]
            self.current += 1

            if char == "<":
                token_end = char
                for c in self.advance_until(text, ">"):
                    token_end += c
                token_end += ">"
                if token_end in prompt_tokens:
                    tokens.append(token_end)
            
            elif char in string.ascii_letters:
                token_variable = char
                for c in self.advance_until(text, ['(', ')', " "]):
                    token_variable += c
                tokens.append(token_variable)
                
            elif char in ['"', "'"] and not stringClosed:
                token = ""
                endSymbol = char
                token += char

                for c in self.advance_until(text, endSymbol):
                    token += c
                token += endSymbol
                tokens.append(self.tokenise_string(token))
                token = ""
                stringClosed = True

            elif char in list(map(lambda op: op[0], Operators)):
                pass
            
            elif char in Symbols:
                pass
            
            token += char
        return tokens

    def advance_until(self, text, endSymbol):
        while self.current < len(text) and text[self.current] != endSymbol:
            yield text[self.current]
            self.current += 1
        else:
            return None
    def advance_until(self, text, endSymbols):
        while self.current < len(text) and text[self.current] not in endSymbols:
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

def tokenizer(text):
    tokeniser = Tokeniser()
    tokens = tokeniser.tokenise_code(text)
    t_to_i = tokeniser.tokens_to_index(tokens)

    indices = []
    for token in tokens:
        if isinstance(token, list):
            indices.extend([t_to_i[t] for t in token])
        else:
            indices.append(t_to_i[token])

    return indices, t_to_i

if __name__ == "__main__":
    text = """
function fizzbuzz(n)
    for i = 1, n do
        if i % 3 == 0 and i % 5 == 0 then
            print("FizzBuzz") 
        elseif i % 3 == 0 then
            print("Fizz")     
        elseif i % 5 == 0 then
            print("Buzz")     
        else
            print(i)        
        end
    end
end
fizzbuzz(100)
<PROGRAM END>
45
<eos>
    """
    tokeniser = Tokeniser()
    tokens = tokeniser.tokenise_code(text)
    t_to_i = tokeniser.tokens_to_index(tokens)

    print(tokens)

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