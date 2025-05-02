from LanguageModel import LanguageModel
from Data import CodeDataset, collate_fn, CodeDatasetSubset
from torch import nn
import torch
from Parser import Tokeniser
from torch.utils.data import DataLoader
from Utils.Logger import LOGGER

"""
    Metrics
    - Pass@k
    - Sentence Similarity
    - EM
    - Perplexity
"""

class Evaluator:
    def __init__(self, test_dataset: CodeDatasetSubset, lm: LanguageModel, k = 5) -> None:
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
    
    def get_vector(self, tokens, word2idx):
        indices = [word2idx[token] for token in tokens]
        indices_tensor = torch.tensor(indices, dtype=torch.long, device=self.lm.device)
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

        for prompt in self.test.get_prompts():
            lines = list(map(lambda s: s.strip(), filter(lambda x : x != '', prompt.split("\n"))))
            end_idx = lines.index("<PROGRAM END>")
            prompt_program = "\n".join(lines[:end_idx])
            promp_output = "\n".join(lines[end_idx + 1:])


            generated_output = self.lm.sample(prompt_program)  # Generate output based on the input prompt
            if generated_output == "ERROR: Model terminated generation immediately.":
                print("Model terminated generation immediately.")
                continue
            elif generated_output == "ERROR: Prompt contains unknown tokens.":
                print("Prompt contains unknown tokens.")
                continue


            tokeniser = Tokeniser()
            out_tokens = tokeniser.traverse_output(promp_output)
            expected_output = out_tokens
            generated_output = list(filter(lambda x: x != '', generated_output.split("<PROGRAM END>")[1].split(" ")))

            exp_vec = self.get_vector(expected_output, self.word2idx)
            out_vec = self.get_vector(generated_output, self.word2idx)

            out_vec, exp_vec = self.pad_vectors(out_vec, exp_vec)


            similarity = self.sentence_similarity(out_vec, exp_vec)

            exact_match = self.exact_match(out_vec, exp_vec)
            precision = self.precision(expected_output, generated_output)
            recall = self.recall(expected_output, generated_output)
            f1 = self.f1_score(expected_output, generated_output)

            if log:
                print("-------------------- Data -----------------")
                print(f"Input: {prompt_program}")
                print(f"Output: {generated_output}")
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

        passk = self.pass_at_k_multiple(k=self.k, num_samples=10)
        metrics_avg['pass@k'] = passk

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

    
    def pass_at_k_single(self, ground_truth, candiates, k):
        if len(candiates) > k:
            candiates = candiates[:k]

        for candidate in candiates:
            if torch.equal(ground_truth, candidate):
                return 1
        return 0
    
    def pass_at_k_multiple(self, k = 1, num_samples = 20):
        pass_k_total = 0
        num_examples = 0

        test_dataloader = DataLoader(
            self.test,
            batch_size=1,
            collate_fn=collate_fn
        )

        for batch in test_dataloader:
            input_seq = batch['input'].to(self.lm.device)
            ground_truth = batch['output'].to(self.lm.device)

            candiates = self.lm.samplek(input_seq, num_samples=num_samples)

            pass_k = self.pass_at_k_single(ground_truth.squeeze(0), candiates, k)
            pass_k_total += pass_k
            num_examples += 1

            #if pass_k == 0:
                #print(f"Pass@{k} failed for GT: {ground_truth.squeeze(0).tolist()}")
                #print(f"Candidates: {[c.tolist() for c in candiates]}")
        
        pass_at_k = pass_k_total / num_examples if num_examples > 0 else 0
        return pass_at_k

    
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