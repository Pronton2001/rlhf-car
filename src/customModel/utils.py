import torch
import numpy as np
import math

def kl_divergence(dist_p, dist_q, obs):
    """
    Compute the KL divergence between two distributions.

    Args:
        dist_p (torch.distributions.Distribution): First distribution.
        dist_q (torch.distributions.Distribution): Second distribution.
        obs (TensorType): Input tensor.

    Returns:
        kl (TensorType): KL divergence tensor.
    """
    p_log_prob = dist_p.logp(action)
    q_log_prob = dist_q.log_prob(obs)
    kl = torch.sum(p_log_prob - q_log_prob, dim=-1)
    return kl
from torch.distributions import Distribution

class PretrainedDistribution(Distribution):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def sample(self):
        return torch.normal(self.mean, self.std)

    def log_prob(self, value):
        variance = self.std.pow(2)
        log_scale = torch.log(self.std)
        log_density = -(value - self.mean).pow(2) / (2 * variance) - log_scale - math.log(math.sqrt(2 * math.pi))
        return log_density.sum(dim=-1)