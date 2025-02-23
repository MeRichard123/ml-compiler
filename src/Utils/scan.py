import torch 
import torch.nn.functional as F
from .Logger import SHAPE_LOG

def log_g(x: torch.Tensor):
    return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))

def parallel_scan_log(log_coeffs: torch.Tensor, log_values: torch.Tensor):
    """Parallel scan operation for log domain values."""
    # log_coeffs: (batch_size, seq_length, input_size)

    # log_values: (batch_size, seq_length + 1, input_size)

    hidden_size = log_coeffs.size(-1)
    log_values = log_values.split(hidden_size, dim=-1)[0]  # Take first half

    SHAPE_LOG("After Split, log_values shapes", log_values)
    SHAPE_LOG("Log Coeffs", log_coeffs)

    a_star = F.pad(
        torch.cumsum(log_coeffs, dim=1),
        (0, 0, 1, 0)
    )

    SHAPE_LOG("a_star", a_star)

    log_h0_plus_b_star = torch.logcumsumexp(log_values - a_star, dim=1)
    log_h = a_star + log_h0_plus_b_star
    return torch.exp(log_h)[:, 1:]