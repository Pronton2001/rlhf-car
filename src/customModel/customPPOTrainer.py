"""
Proximal Policy Optimization (PPO)
==================================

This file defines the distributed Algorithm class for proximal policy
optimization.
See `ppo_[tf|torch]_policy.py` for the definition of the policy loss.

Detailed documentation: https://docs.ray.io/en/master/rllib-algorithms.html#ppo
"""

import logging
from typing import List, Optional, Type, Union

from ray.util.debug import log_once
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.algorithms.pg import PGConfig
from ray.rllib.execution.rollout_ops import (
    standardize_fields,
)
from ray.rllib.execution.train_ops import (
    train_one_step,
    multi_gpu_train_one_step,
)
from ray.rllib.utils.annotations import ExperimentalAPI
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import (
    Deprecated,
    DEPRECATED_VALUE,
    deprecation_warning,
)
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.typing import ResultDict
from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
    SYNCH_WORKER_WEIGHTS_TIMER,
)

logger = logging.getLogger(__name__)
from ray.rllib.algorithms.ppo.ppo import PPOConfig, PPO




class KLPPO(PPO):
    # @classmethod
    # @override(PPO)
    # def get_default_config(cls) -> AlgorithmConfig:
    #     return PPOConfig()

    @classmethod
    @override(PPO)
    def get_default_policy_class(
        cls, config: AlgorithmConfig
    ) -> Optional[Type[Policy]]:
        from src.customModel.customPPOTorchPolicy import KLPPOTorchPolicy
        return KLPPOTorchPolicy