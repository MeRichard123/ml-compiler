import torch.nn as nn
import tree_sitter_lua as lua_grammar
from tree_sitter import Language, Parser
from enum import Enum

LANGUAGE = Language(lua_grammar.language())

node_types = LANGUAGE.node_kind_count
AST_NODES = {
    LANGUAGE.node_kind_for_id(i): i for i in range(node_types)
}

class PROMPT_TOKENS(Enum):
    PROGRAM_END = "<PROGRAM END>"
    EOS         = "<eos>"
    UNK         = "<unk>"

    @staticmethod
    def get_list():
        return [token.value for token in PROMPT_TOKENS]


class Tokeniser:
    def __init__(self):
        self.parser = Parser(LANGUAGE)
        self.literals = []

    def generate_ast(self, code):
        tree = self.parser.parse(code.encode("utf8"))
        return tree.root_node

    def tokenise_code(self, text):
        # Split into input and output parts
        if PROMPT_TOKENS.PROGRAM_END.value in text:
            program, output = text.split(PROMPT_TOKENS.PROGRAM_END.value)
            output = output.strip()
        else:
            program = text
            output = ""

        # Parse the program
        tree = self.generate_ast(program)

        input_tokens = self.traverse_ast(tree)
        input_tokens.append(PROMPT_TOKENS.PROGRAM_END.value)
        
        # Tokenize the output directly (no AST needed for output)
        output_tokens = output.split()
        
        # Combine for vocabulary building but keep track of what's output
        all_tokens = input_tokens + output_tokens
        
        return input_tokens, output_tokens, all_tokens

    def traverse_ast(self, node):
        tokens = []
        
        # Only include node type for program structure
        tokens.append(node.type)
        
        # For nodes containing literal values, add the actual value
        if node.type in ["string", "number", "identifier"]:
            literal = node.text.decode('utf8')
            tokens.append(literal)
            self.literals.append(literal)
            
        # Recursively process children
        for child in node.children:
            tokens.extend(self.traverse_ast(child))
            
        return tokens

def tokenizer(text):
    tokeniser = Tokeniser()
    input_tokens, output_tokens, all_tokens = tokeniser.tokenise_code(text)
    
    # Create vocabulary from all tokens
    vocab = set(all_tokens)
    vocab.update(PROMPT_TOKENS.get_list())
    
    # Convert to index mapping
    t_to_i = {token: idx for idx, token in enumerate(sorted(vocab))}
    
    # Convert tokens to indices
    input_indices = [t_to_i[token] for token in input_tokens]
    output_indices = [t_to_i[token] for token in output_tokens]
    
    return (input_indices, output_indices), t_to_i

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
1
2
3
"""
    (input, output), t_to_i = tokenizer(text)
    print("INPUT: \n")
    print(input)
    print("\n\nOUTPUT: \n")
    print(output)

    program = """
    print("Hello Sheep!")
    <PROGRAM END>
    Hello Sheep!
    """

    (input, output), t_to_i = tokenizer(program)
    print("INPUT: \n")
    print(input)
    print("\n\nOUTPUT: \n")
    print(output)


    print(t_to_i)