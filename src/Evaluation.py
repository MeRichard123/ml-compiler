from LanguageModel import LanguageModel
from Data import CodeDataset
from torch import nn
import torch
from Parser import tokenizer
from nltk.translate.bleu_score import sentence_bleu
from torcheval.metrics import Perplexity

"""
    Metrics
    - Pass@k
    - Sentence Similarity
    - BLUE
    - Perplexity
"""

class Evaluator:
    def __init__(self, test_dataset: CodeDataset, lm: LanguageModel, k = 5) -> None:
        self.test = test_dataset
        self.lm = lm
        self.k = k

    def pad_vectors(self, vec1, vec2):
        max_len = max(vec1.size(1), vec2.size(1))

        if vec1.size(1) < max_len:
            padding = torch.zeros(1, max_len - vec1.size(1), dtype=torch.long, device=self.lm.device)
            vec1 = torch.cat([vec1, padding], dim=1)
        if vec2.size(1) < max_len:
            padding = torch.zeros(1, max_len - vec2.size(1), dtype=torch.long, device=self.lm.device)
            vec2 = torch.cat([vec2, padding], dim=1)

        return vec1, vec2

    def evaluate(self):
        for prompt in self.test.get_prompts():
            idx = prompt.index("<PROGRAM END>")
            input_prompt = ''.join(prompt[0: idx])  # Extract the input prompt

            generated_output = self.lm.sample(input_prompt).split("<PROGRAM END>")[1] # Generate output based on the input prompt
        
            expected_output = prompt.split("<PROGRAM END>")[1]

            (exp_int, _), _ = tokenizer(expected_output)
            exp_vec = torch.tensor([exp_int], dtype=torch.long, device=self.lm.device)
            (out_int, _), _ = tokenizer(generated_output)
            out_vec = torch.tensor([out_int], dtype=torch.long, device=self.lm.device)

            out_vec, exp_vec = self.pad_vectors(out_vec, exp_vec)

            similarity = self.sentence_similarity(out_vec, exp_vec)

            bleu = sentence_bleu(
                generated_output.split(" "),
                expected_output.split("\n")[0:len(expected_output.split("\n"))-1]
            )

            perplexity = self.perplexity(out_vec, exp_vec)
            pass_at_k = self.pass_at_k(out_vec, exp_vec, self.k)

            print(f"Input: {input_prompt}")
            print(f"Output: {generated_output}")
            print(f"Expected: {expected_output}")
            print(f"Sentence Similarity: {similarity}")
            print(f"BLEU: {bleu}")
            print(f"Perplexity: {perplexity}")
            print(f"Pass@{self.k}: {pass_at_k}")
            print("\n\n\n\n\n")

    def sentence_similarity(self, vec1, vec2):
        similarity = nn.functional.cosine_similarity(
            vec1.float(),
            vec2.float()
        )

        return similarity.item()
    
    def perplexity(self, vec1, vec2):
        return 0
    
    def pass_at_k(self, vec1, vec2, k):
        return 0
