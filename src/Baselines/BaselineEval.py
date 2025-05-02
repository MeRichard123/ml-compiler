from torch import nn
import torch
from Parser import Tokeniser
import os 
from enum import Enum


"""
    Metrics
    - Pass@k
    - Sentence Similarity
    - EM
    - Perplexity
"""
# D:\richa\Documents\Dev\Dissertation\ml-compiler\training_examples



class PROMPT_TOKENS(Enum):
    PROGRAM_END = "<PROGRAM END>"
    EOS         = "<eos>"
    UNK         = "<unk>"

    @staticmethod
    def get_list():
        return [token.value for token in PROMPT_TOKENS]

class Evaluator:
    def __init__(self, k = 5, word2idx = None, dir = None, device = "cpu") -> None:
        self.k = k
        self.word2idx = word2idx
        self.files = os.listdir(dir)
        self.dir = dir
        BASEPATH = os.path.dirname(os.path.abspath(__file__)).replace("Baselines", "").replace("src\\", "")
        self.training_dir = os.path.join(BASEPATH, "training_examples", "All")
        self.device = device

    def pad_vectors(self, vec1, vec2):
        max_len = max(vec1.size(1), vec2.size(1))

        if vec1.size(1) < max_len:
            padding = torch.zeros(1, max_len - vec1.size(1), dtype=torch.long, device=self.device)
            vec1 = torch.cat([vec1, padding], dim=1)
        if vec2.size(1) < max_len:
            padding = torch.zeros(1, max_len - vec2.size(1), dtype=torch.long, device=self.device)
            vec2 = torch.cat([vec2, padding], dim=1)

        return vec1, vec2
    
    def get_vector(self, tokens, word2idx):
        unk_idx = word2idx.get(PROMPT_TOKENS.UNK.value, -1)  # Ensure <unk> exists

        indices = [word2idx.get(token, unk_idx) for token in tokens]
        indices_tensor = torch.tensor(indices, dtype=torch.long, device=self.device)
        return indices_tensor.unsqueeze(0) 

    def evaluate(self, perplexity, log = True):
        metrics_sum = {
            'pass_at_k': 0,
            'exact_match': 0,
            'similarity': 0,
            'f1': 0,
            'precision': 0,
            'recall': 0,
        }
        n_samples = 0

        for file in self.files:
            training_file = os.path.join(self.training_dir, file)

            if not os.path.exists(training_file):
                print(f"File {training_file} does not exist")
                continue

            lines = list(map(lambda x: x.strip(), open(training_file).readlines()))
            end_idx = lines.index("<PROGRAM END>")
            prompt_program = "\n".join(lines[:end_idx])
            promp_output = "\n".join(lines[end_idx + 1:])

            tokeniser = Tokeniser()
            file_content = open(os.path.join(self.dir, file),encoding="utf8").read()
            generated_output = file_content.split("<PROGRAM END>")[1].strip()
            gen_tokens = tokeniser.traverse_output(generated_output)

            out_tokens = tokeniser.traverse_output(promp_output)
            expected_output = out_tokens
  
            exp_vec = self.get_vector(expected_output, self.word2idx)
            out_vec = self.get_vector(gen_tokens, self.word2idx)

            out_vec, exp_vec = self.pad_vectors(out_vec, exp_vec)


            similarity = self.sentence_similarity(out_vec, exp_vec)

            exact_match = self.exact_match(out_vec, exp_vec)
            precision = self.precision(expected_output, gen_tokens)
            recall = self.recall(expected_output, gen_tokens)
            f1 = self.f1_score(expected_output, gen_tokens)

            if log:
                print("-------------------- Data -----------------")
                print(f"Input: {prompt_program}")
                print(f"Output: {gen_tokens}")
                print(f"Expected: {expected_output}")
                print("----------------- Metrics ------------------")
                print(f"Sentence Similarity: {similarity}")
                print(f"Exact Match: {exact_match}")
                print(f"F1 Score: {f1}")
                print(f"Precision: {precision}")
                print(f"Recall: {recall}")
                print("\n\n\n\n\n")
            
            metrics_sum['exact_match'] += exact_match
            metrics_sum['similarity'] += similarity
            metrics_sum['f1'] += f1
            metrics_sum['precision'] += precision
            metrics_sum['recall'] += recall
            
            n_samples += 1
            print(f"Sampled {n_samples} samples")
            
        metrics_avg = {k : v/n_samples for k,v in metrics_sum.items()}
        metrics_avg['perplexity'] = perplexity


        return metrics_avg

    def sentence_similarity(self, vec1, vec2):
        similarity = nn.functional.cosine_similarity(
            vec1.float(),
            vec2.float()
        )

        return similarity.item()
    
    def exact_match(self, vec1, vec2):
        if vec1.size(1) != vec2.size(1):
            return 0

        return int(torch.all(vec1 == vec2).item())

    
    def precision(self, expected, generated):
        words_in_generated = len(generated)
        overlap = len(set(expected).intersection(generated))
        if words_in_generated == 0:
            return 0
        return overlap / words_in_generated
    
    def recall(self, expected, generated):
        words_in_expected = len(expected)
        overlap = len(set(expected).intersection(generated))
        if words_in_expected == 0:
            return 0
        return overlap / words_in_expected
    
    def f1_score(self, expected, generated):
        p = self.precision(expected, generated)
        r = self.recall(expected, generated)
        if p + r == 0:
            return 0
        return 2 * ((p * r) / (p + r))
    

