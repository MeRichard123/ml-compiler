from LanguageModel import LanguageModel
from Data import CodeDataset
from torch import nn
import torch
from Parser import tokenizer
from torcheval.metrics import Perplexity

"""
    Metrics
    - Pass@k
    - Sentence Similarity
    - EM
    - Perplexity
"""

class Evaluator:
    def __init__(self, test_dataset: CodeDataset, lm: LanguageModel, k = 5) -> None:
        self.test = test_dataset
        self.lm = lm
        self.k = k
        self.word2idx = self.lm.word2idx

    def pad_vectors(self, vec1, vec2):
        max_len = max(vec1.size(1), vec2.size(1))

        if vec1.size(1) < max_len:
            padding = torch.zeros(1, max_len - vec1.size(1), dtype=torch.long, device=self.lm.device)
            vec1 = torch.cat([vec1, padding], dim=1)
        if vec2.size(1) < max_len:
            padding = torch.zeros(1, max_len - vec2.size(1), dtype=torch.long, device=self.lm.device)
            vec2 = torch.cat([vec2, padding], dim=1)

        return vec1, vec2

    def evaluate(self, log = True):
        metrics_sum = {
            'perplexity': 0,
            'pass_at_k': 0,
            'exact_match': 0,
            'similarity': 0
        }
        n_samples = 0

        for prompt in self.test.get_prompts():
            idx = prompt.index("<PROGRAM END>")
            input_prompt = ''.join(prompt[0: idx])  # Extract the input prompt

            generated_output = self.lm.sample(input_prompt)  # Generate output based on the input prompt
            print(f"Generated Output: {generated_output}")
            #generated_output = self.lm.sample(input_prompt).split("<PROGRAM END>")[1] # Generate output based on the input prompt
        
            expected_output = prompt.split("<PROGRAM END>")[1]

            (exp_int, _), _ = tokenizer(expected_output, self.word2idx)
            exp_vec = torch.tensor([exp_int], dtype=torch.long, device=self.lm.device)
            (out_int, _), _ = tokenizer(generated_output, self.word2idx)
            out_vec = torch.tensor([out_int], dtype=torch.long, device=self.lm.device)

            out_vec, exp_vec = self.pad_vectors(out_vec, exp_vec)

            similarity = self.sentence_similarity(out_vec, exp_vec)

            perplexity = self.perplexity(out_vec, exp_vec)
            pass_at_k = self.pass_at_k(out_vec, exp_vec, self.k)
            exact_match = self.exact_match(out_vec, exp_vec)
            if log:
                print("-------------------- Data -----------------")
                print(f"Input: {input_prompt}")
                print(f"Output: {generated_output}")
                print(f"Expected: {expected_output}")
                print("----------------- Metrics ------------------")
                print(f"Sentence Similarity: {similarity}")
                print(f"Perplexity: {perplexity}")
                print(f"Pass@{self.k}: {pass_at_k}")
                print(f"Exact Match: {exact_match}")
                print("\n\n\n\n\n")
            
            metrics_sum['perplexity'] += perplexity
            metrics_sum['pass_at_k'] += pass_at_k
            metrics_sum['exact_match'] += exact_match
            metrics_sum['similarity'] += similarity
            n_samples += 1
            
        metrics_avg = {k : v/n_samples for k,v in metrics_sum.items()}
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
    
    def perplexity(self, vec1, vec2):
        # Initialize the perplexity metric
        metric = Perplexity().to(self.lm.device)
        vocab_size = self.lm.vocab_size
        
        # Reshape tensors to correct dimensions
        # vec1 should be [batch_size, sequence_length]
        if len(vec1.shape) > 2:
            vec1 = vec1.squeeze()
        
        # Create a simple prediction distribution
        # [batch_size, sequence_length, vocab_size]
        logits = torch.zeros((1, vec1.size(1), vocab_size), device=vec1.device)
        logits.scatter_(2, vec1.unsqueeze(-1), 1)  # One-hot encoding
        probs = torch.softmax(logits, dim=-1)
        vec2 = vec2.to(self.lm.device)
        # Update and compute perplexity
        metric.update(probs, vec2)
        return metric.compute().item()
    
    def pass_at_k(self, vec1, vec2, k):
        # Check if the vectors are of the same size
        if vec1.size(1) != vec2.size(1):
            return 0
        
        # Calculate the number of matching positions
        matches = (vec1 == vec2).sum(dim=1).item()
        
        # Calculate the total length of the sequence
        total_length = vec1.size(1)
        
        # If matches/total_length >= 1/k, consider it a pass
        return int((matches / total_length) >= (1 / k))
