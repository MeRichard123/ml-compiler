import torch.nn as nn
import torch

class MoeLayer(nn.Module):
    # Dense layer with a mixture of experts
    def __init__(self, num_experts, n_embed):
        super(MoeLayer, self).__init__()
        self.experts = nn.ModuleList(
            [nn.Linear(n_embed, n_embed) for _ in range(num_experts)]
            )
        assert len(self.experts) > 0
        self.gate = nn.Linear(n_embed, num_experts, bias=False)

    def forward(self, inputs: torch.Tensor):
        batch_size, seq_len, n_embed = inputs.shape 

        input_squashed = inputs.view(-1, n_embed) # (batch_size * seq_len, n_embed)

        # Get the gate scores and apply softmax
        gate_logits = self.gate(input_squashed)
        gate_probs = nn.functional.softmax(gate_logits, dim=-1)

        # Get the expert outputs
        expert_outputs = torch.stack(
            [expert(input_squashed) for expert in self.experts], dim=-1
        )

        # weight expert outputs by the gate scores and sum them
        weighted_expert_outputs = gate_probs.unsqueeze(1) * expert_outputs
        output = weighted_expert_outputs.sum(dim=-1) # (batch_size * seq_len, n_embed)

        return output.reshape(batch_size, seq_len, n_embed)