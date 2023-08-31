from typing import Optional, Union
import torch
import numpy as np
import math

from ray.rllib.models.torch.torch_action_dist import TorchSquashedGaussian
from ray.rllib.models.torch.torch_action_dist import TorchDiagGaussian 
# from ray.rllib.models.torch.torch_distributions import TorchDiagGaussian
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

def kl_mc(sg1: TorchSquashedGaussian, sg2: TorchSquashedGaussian, num_samples: int = 10000000) -> torch.TensorType:
    z1 = sg1.dist.rsample((num_samples,))
    # print('sample d1', z1)
    logp1 = sg1.logp(z1)
    # print('sample logp1', logp1)
    z2 = sg2.dist.rsample((num_samples,))
    # print('sample d2', z2)
    logp2 = sg2.logp(z2)
    # print('sample logp2', logp2)

    kl = (logp1 - logp2).mean()
    return kl

def kl_approx(sg1: Union[TorchDiagGaussian, TorchSquashedGaussian], sg2: Union[TorchDiagGaussian, TorchSquashedGaussian]) -> torch.TensorType:
    sigma1 = sg1.dist.scale
    mu1 = sg1.dist.loc
    sigma2 = sg2.dist.scale.to(sigma1.device)
    mu2 = sg2.dist.loc.to(sigma1.device)

    kl = torch.log(sigma2 / sigma1) + (sigma1 ** 2 + (mu1 - mu2) ** 2) / (2 * sigma2 ** 2) - 0.5
    return kl.sum()

if __name__ == '__main__':
    import time
    s = time.time()
    d1 = TorchSquashedGaussian(np.array([2.0,5.0,3.0,-1,-1,-1]),None)
    print('time:',time.time() - s)
    s = time.time()
    d2 = TorchSquashedGaussian(np.array([1.0,2.0,0.0,-1,-1,-1]),None)
    print('time:',time.time() - s)
    s = time.time()
    print('kl_mc', abs(kl_mc(d1, d2)))
    print('time:',time.time() - s)
    # print(kl_mc())
    s = time.time()
    print('kl approx', kl_approx(d1, d2))
    print('time:',time.time() - s)
    # import torch.distributions as dis
    # truekl = dis.kl_divergence(d1.dist, d2.dist)
    # print('true KL', truekl)
    # p,q = d1.dist, d2.dist
    # logr = p.log_prob(x) - q.log_prob(x)
    # k1 = -logr
    # k2 = logr ** 2 / 2
    # k3 = (logr.exp() - 1) - logr
    # for k in (k1, k2, k3):
    # print((k.mean() - truekl) / truekl, k.std() / truekl)
    d3 = TorchDiagGaussian(torch.tensor([[2.0,5.0,3.0,-1,-1,-1]]), None)
    d4 = TorchDiagGaussian(torch.tensor([[1.0,2.0,0.0,-1,-1,-1]]), None)
    # d3 = TorchDiagGaussian(loc = torch.tensor([1.0,2.0,3.0]), scale = torch.tensor([1,1,1]))
    # d4 = TorchDiagGaussian(loc = torch.tensor([1.0,2.0,2.0]), scale = torch.tensor([1,1,1]))
    print('kl', d3.kl(d4))
    print('inverse kl', d4.kl(d3))
    import torch.distributions as dis
    p = d1.dist
    q = d2.dist
    x = q.sample(sample_shape=(10_000_000,))
    truekl = dis.kl_divergence(p, q)
    print("true", truekl)
    logr = p.log_prob(x) - q.log_prob(x)
    k1 = -logr
    k2 = logr ** 2 / 2
    k3 = (logr.exp() - 1) - logr
    for k in (k1, k2, k3):
        print((k.mean() - truekl) / truekl, k.std() / truekl)
