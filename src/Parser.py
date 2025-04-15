import torch.nn as nn
from tree_sitter import Language, Parser
from enum import Enum
import re

LANGUAGE = Language("./src/Utils/lua.dll", name="lua")

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
        self.parser = Parser()
        self.parser.set_language(LANGUAGE)

    def generate_ast(self, code):
        tree = self.parser.parse(code.encode("utf8"))
        return tree.root_node

    def tokenise_code(self, text):
        # Split into input and output parts
        if PROMPT_TOKENS.PROGRAM_END.value in text:
            program, output = text.split(PROMPT_TOKENS.PROGRAM_END.value)
            output = self.traverse_output(output)
        else:
            program = text
            output = []
        self.program = program

        # Parse the program
        tree = self.generate_ast(program)

        input_tokens = self.traverse_ast(tree)
        input_tokens.append(PROMPT_TOKENS.PROGRAM_END.value)
        output_tokens = output
        

        # Combine for vocabulary building but keep track of what's output
        all_tokens = input_tokens + output_tokens
        
        return input_tokens, output_tokens, all_tokens
    
    def traverse_output(self, text):
        # Tokenize the output directly (no AST needed for output)
        text = text.strip().split("\n")

        tokens = []
        for line in text:
            line = line.strip()
            line = re.sub('[^A-Za-z0-9 ]+', '', line) 

            if line.isnumeric():
                tokens.append(f'NUMBER({line})')
            elif line == 'eos':
                tokens.append(PROMPT_TOKENS.EOS.value)
            else:
                for word in line.split():
                    if word.isnumeric():
                        tokens.append(f'NUMBER({word})')
                    else:
                        tokens.append(f'STRING({word})')
        return tokens

    def traverse_ast(self, node):
        tokens = []

        match node.type:
            case 'end':
                tokens.append('END')
            case 'identifier':
                tokens.append(f"IDENTIFIER({self.program[node.start_byte:node.end_byte]})")
            case 'return':
                tokens.append('RETURN')
            case ';':
                tokens.append('SEMICOLON')
            case '=':
                tokens.append('EQUALS')
            case ',':
                tokens.append('COMMA')
            case '::':
                tokens.append('LABEL')
            case 'break_statement':
                tokens.append('BREAK')
            case 'goto':
                tokens.append('GOTO')
            case 'do':
                tokens.append('DO')
            case 'while':
                tokens.append('WHILE')
            case 'repeat':
                tokens.append('REPEAT')
            case 'until':
                tokens.append('UNTIL')
            case 'if':
                tokens.append('IF')
            case 'then':
                tokens.append('THEN')
            case 'elseif_statement':
                tokens.append('ELSEIF')
            case 'else':
                tokens.append('ELSE')
            case 'for_statement':
                tokens.append('FOR')
            case 'in':
                tokens.append('IN')
            case 'function':
                tokens.append('FUNCTION')
            case 'local':
                tokens.append('LOCAL')
            case 'nil':
                tokens.append('NIL')
            case 'false':
                tokens.append('FALSE')
            case 'true':
                tokens.append('TRUE')
            case 'number':
                tokens.append(f'NUMBER({self.program[node.start_byte:node.end_byte]})')
            case 'string_content':
                string = self.program[node.start_byte:node.end_byte]
                string = re.sub('[^A-Za-z0-9 ]+', '', string)
                for word in string.split():
                    if word.isnumeric():
                        tokens.append(f'NUMBER({word})')
                    else:
                        tokens.append(f'STRING({word})')
                #tokens.append(f'STRING({string})')
            case '+':
                tokens.append('PLUS')
            case '-':
                tokens.append('MINUS')
            case '*':
                tokens.append('MULTIPLY')
            case '/':
                tokens.append('DIVIDE')
            case '..':
                tokens.append('CONCAT')
            case '==':
                tokens.append('EQUALITY')
            case '~=':
                tokens.append('NOT_EQUAL')
            case '<':
                tokens.append('LESS_THAN')
            case '<=':
                tokens.append('LESS_THAN_EQUAL')
            case '>':
                tokens.append('GREATER_THAN')
            case '>=':
                tokens.append('GREATER_THAN_EQUAL')
            case '%':
                tokens.append('MODULUS')
            case 'function_call':
                tokens.extend(self.process_function_call(node))
                return tokens
            case _:
                pass
                #tokens.append(f'UNKNOWN({node.type})')
    
            
        # Recursively process children
        for child in node.children:
            tokens.extend(self.traverse_ast(child))
            
        return tokens
    
    def process_function_call(self, node):
        """Process a function_call node and return its tokens."""
        tokens = []
        if node.child_count > 0:
            func_node = node.children[0]  # First child is the function expression
            if func_node.type == 'identifier':
                func_name = self.program[func_node.start_byte:func_node.end_byte]
                tokens.append(f'FUNCTION_CALL({func_name})')
            elif func_node.type == 'field_expression':
                func_name = self._extract_field_expression(func_node)
                tokens.append(f'FUNCTION_CALL({func_name})')
            elif func_node.type == 'variable' and node.child_count > 1 and node.children[1].type == ':':
                obj_node = func_node.children[0]
                method_node = node.children[2]
                if obj_node.type == 'identifier' and method_node.type == 'identifier':
                    obj_name = self.program[obj_node.start_byte:obj_node.end_byte]
                    method_name = self.program[method_node.start_byte:method_node.end_byte]
                    tokens.append(f'FUNCTION_CALL({obj_name}:{method_name})')
            # Process arguments (skip the function name child)
            for child in node.children[1:]:
                tokens.extend(self.traverse_ast(child))
        return tokens
    
    def _extract_field_expression(self, node):
        """Helper to extract full name from a field_expression like table.insert"""
        parts = []
        current = node
        while current.child_count > 0:
            if current.type == 'field_expression':
                # Left part is the base (e.g., table), right is the field (e.g., insert)
                if current.child_count >= 3:  # base, dot, field
                    base = current.children[0]
                    field = current.children[2]
                    if base.type == 'identifier':
                        parts.insert(0, self.program[base.start_byte:base.end_byte])
                    if field.type == 'identifier':
                        parts.append(self.program[field.start_byte:field.end_byte])
                    current = base  # Continue with nested base if any
                else:
                    break
            else:
                break
        return '.'.join(parts[::-1])  # Reverse to get table.insert order

def tokenizer(text, word2idx):
    tokeniser = Tokeniser()
    input_tokens, output_tokens, all_tokens = tokeniser.tokenise_code(text)
    
    # Create vocabulary from all tokens
    vocab = set(all_tokens)
    vocab.update(PROMPT_TOKENS.get_list())
    
    # Convert to index mapping
    t_to_i = word2idx.copy()
    next_index = max(t_to_i.values()) + 1 if t_to_i else 0
    for token in vocab:
        if token not in t_to_i:
            t_to_i[token] = next_index
            next_index += 1

    unk_idx = t_to_i.get(PROMPT_TOKENS.UNK.value, -1)  # Ensure <unk> exists
    if unk_idx == -1:
        t_to_i[PROMPT_TOKENS.UNK.value] = next_index
        unk_idx = next_index
        next_index += 1
    
    i_to_t = {idx: token for token, idx in t_to_i.items()}
    # Convert tokens to indices
    input_indices = [t_to_i.get(token, unk_idx) for token in input_tokens]
    output_indices = [t_to_i.get(token, unk_idx) for token in output_tokens]
    
    return (input_indices, output_indices), (t_to_i, i_to_t)

if __name__ == "__main__":
    text = """
str = "LUA123"
print(string.lower(str))

<PROGRAM END>
lua123
 <eos>
"""
    input_tokens, output_tokens, all_tokens = Tokeniser().tokenise_code(text)
    print("INPUT: \n")
    print(input_tokens)
    
    print("\n\nOUTPUT: \n")
    print(output_tokens)