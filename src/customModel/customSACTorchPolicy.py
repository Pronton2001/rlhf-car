"""
PyTorch policy class used for SAC.
"""

import gym
from gym.spaces import Box, Discrete
import logging
import tree  # pip install dm_tree
from typing import Dict, List, Optional, Tuple, Type, Union

import ray
import ray.experimental.tf_utils
from ray.rllib.algorithms.sac.sac_tf_policy import (
    build_sac_model,
    postprocess_trajectory,
    validate_spaces,
)
from ray.rllib.algorithms.dqn.dqn_tf_policy import PRIO_WEIGHTS
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import (
    TorchCategorical,
    TorchDistributionWrapper,
    TorchDirichlet,
    TorchSquashedGaussian,
    TorchDiagGaussian,
    TorchBeta,
)
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.policy_template import build_policy_class
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.spaces.simplex import Simplex
from ray.rllib.policy.torch_mixins import TargetNetworkMixin
from ray.rllib.utils.torch_utils import (
    apply_grad_clipping,
    concat_multi_gpu_td_errors,
    huber_loss,
)
from ray.rllib.utils.typing import (
    LocalOptimizer,
    ModelInputDict,
    TensorType,
    AlgorithmConfigDict,
)

from src.constant import SRC_PATH
torch, nn = try_import_torch()
F = nn.functional

logger = logging.getLogger(__name__)


logging.basicConfig(filename=SRC_PATH + 'src/log/info.log', level=logging.DEBUG, filemode='w')

def _get_dist_class(
    policy: Policy, config: AlgorithmConfigDict, action_space: gym.spaces.Space
) -> Type[TorchDistributionWrapper]:
    """Helper function to return a dist class based on config and action space.

    Args:
        policy: The policy for which to return the action
            dist class.
        config: The Algorithm's config dict.
        action_space (gym.spaces.Space): The action space used.

    Returns:
        Type[TFActionDistribution]: A TF distribution class.
    """
    if hasattr(policy, "dist_class") and policy.dist_class is not None:
        return policy.dist_class
    elif config["model"].get("custom_action_dist"):
        action_dist_class, _ = ModelCatalog.get_action_dist(
            action_space, config["model"], framework="torch"
        )
        return action_dist_class
    elif isinstance(action_space, Discrete):
        return TorchCategorical
    elif isinstance(action_space, Simplex):
        return TorchDirichlet
    else:
        assert isinstance(action_space, Box)
        if config["normalize_actions"]:
            return (
                TorchSquashedGaussian
                if not config["_use_beta_distribution"]
                else TorchBeta
            )
        else:
            return TorchDiagGaussian


def build_sac_model_and_action_dist(
    policy: Policy,
    obs_space: gym.spaces.Space,
    action_space: gym.spaces.Space,
    config: AlgorithmConfigDict,
) -> Tuple[ModelV2, Type[TorchDistributionWrapper]]:
    """Constructs the necessary ModelV2 and action dist class for the Policy.

    Args:
        policy: The TFPolicy that will use the models.
        obs_space (gym.spaces.Space): The observation space.
        action_space (gym.spaces.Space): The action space.
        config: The SAC trainer's config dict.

    Returns:
        ModelV2: The ModelV2 to be used by the Policy. Note: An additional
            target model will be created in this function and assigned to
            `policy.target_model`.
    """
    model = build_sac_model(policy, obs_space, action_space, config)
    action_dist_class = _get_dist_class(policy, config, action_space)
    return model, action_dist_class


def action_distribution_fn(
    policy: Policy,
    model: ModelV2,
    input_dict: ModelInputDict,
    *,
    state_batches: Optional[List[TensorType]] = None,
    seq_lens: Optional[TensorType] = None,
    prev_action_batch: Optional[TensorType] = None,
    prev_reward_batch=None,
    explore: Optional[bool] = None,
    timestep: Optional[int] = None,
    is_training: Optional[bool] = None
) -> Tuple[TensorType, Type[TorchDistributionWrapper], List[TensorType]]:
    """The action distribution function to be used the algorithm.

    An action distribution function is used to customize the choice of action
    distribution class and the resulting action distribution inputs (to
    parameterize the distribution object).
    After parameterizing the distribution, a `sample()` call
    will be made on it to generate actions.

    Args:
        policy: The Policy being queried for actions and calling this
            function.
        model (TorchModelV2): The SAC specific model to use to generate the
            distribution inputs (see sac_tf|torch_model.py). Must support the
            `get_action_model_outputs` method.
        input_dict: The input-dict to be used for the model
            call.
        state_batches (Optional[List[TensorType]]): The list of internal state
            tensor batches.
        seq_lens (Optional[TensorType]): The tensor of sequence lengths used
            in RNNs.
        prev_action_batch (Optional[TensorType]): Optional batch of prev
            actions used by the model.
        prev_reward_batch (Optional[TensorType]): Optional batch of prev
            rewards used by the model.
        explore (Optional[bool]): Whether to activate exploration or not. If
            None, use value of `config.explore`.
        timestep (Optional[int]): An optional timestep.
        is_training (Optional[bool]): An optional is-training flag.

    Returns:
        Tuple[TensorType, Type[TorchDistributionWrapper], List[TensorType]]:
            The dist inputs, dist class, and a list of internal state outputs
            (in the RNN case).
    """
    # Get base-model output (w/o the SAC specific parts of the network).
    model_out, _ = model(input_dict, [], None)
    # Use the base output to get the policy outputs from the SAC model's
    # policy components.
    action_dist_inputs, _ = model.get_action_model_outputs(model_out)
    # Get a distribution class to be used with the just calculated dist-inputs.
    action_dist_class = _get_dist_class(policy, policy.config, policy.action_space)

    return action_dist_inputs, action_dist_class, []


def actor_critic_loss(
    policy: Policy,
    model: ModelV2,
    dist_class: Type[TorchDistributionWrapper],
    train_batch: SampleBatch,
) -> Union[TensorType, List[TensorType]]:
    """Constructs the loss for the Soft Actor Critic.

    Args:
        policy: The Policy to calculate the loss for.
        model (ModelV2): The Model to calculate the loss for.
        dist_class (Type[TorchDistributionWrapper]: The action distr. class.
        train_batch: The training data.

    Returns:
        Union[TensorType, List[TensorType]]: A single loss tensor or a list
            of loss tensors.
    """
    # Look up the target model (tower) using the model tower.
    target_model = policy.target_models[model]

    # Should be True only for debugging purposes (e.g. test cases)!
    deterministic = policy.config["_deterministic_loss"]

    model_out_t, _ = model(
        SampleBatch(obs=train_batch[SampleBatch.CUR_OBS], _is_training=True), [], None
    )

    model_out_tp1, _ = model(
        SampleBatch(obs=train_batch[SampleBatch.NEXT_OBS], _is_training=True), [], None
    )

    target_model_out_tp1, _ = target_model(
        SampleBatch(obs=train_batch[SampleBatch.NEXT_OBS], _is_training=True), [], None
    )

    alpha = torch.exp(model.log_alpha)

    # Discrete case.
    if model.discrete:
        # Get all action probs directly from pi and form their logp.
        action_dist_inputs_t, _ = model.get_action_model_outputs(model_out_t)
        log_pis_t = F.log_softmax(action_dist_inputs_t, dim=-1)
        policy_t = torch.exp(log_pis_t)
        action_dist_inputs_tp1, _ = model.get_action_model_outputs(model_out_tp1)
        log_pis_tp1 = F.log_softmax(action_dist_inputs_tp1, -1)
        policy_tp1 = torch.exp(log_pis_tp1)
        # Q-values.
        q_t, _ = model.get_q_values(model_out_t)
        # Target Q-values.
        q_tp1, _ = target_model.get_q_values(target_model_out_tp1)
        if policy.config["twin_q"]:
            twin_q_t, _ = model.get_twin_q_values(model_out_t)
            twin_q_tp1, _ = target_model.get_twin_q_values(target_model_out_tp1)
            q_tp1 = torch.min(q_tp1, twin_q_tp1)
        q_tp1 -= alpha * log_pis_tp1

        # Actually selected Q-values (from the actions batch).
        one_hot = F.one_hot(
            train_batch[SampleBatch.ACTIONS].long(), num_classes=q_t.size()[-1]
        )
        q_t_selected = torch.sum(q_t * one_hot, dim=-1)
        if policy.config["twin_q"]:
            twin_q_t_selected = torch.sum(twin_q_t * one_hot, dim=-1)
        # Discrete case: "Best" means weighted by the policy (prob) outputs.
        q_tp1_best = torch.sum(torch.mul(policy_tp1, q_tp1), dim=-1)
        q_tp1_best_masked = (1.0 - train_batch[SampleBatch.DONES].float()) * q_tp1_best
    # Continuous actions case.
    else:
        # Sample single actions from distribution.
        action_dist_class = _get_dist_class(policy, policy.config, policy.action_space)
        action_dist_inputs_t, _ = model.get_action_model_outputs(model_out_t)
        action_dist_t = action_dist_class(action_dist_inputs_t, model)
        policy_t = (
            action_dist_t.sample()
            if not deterministic
            else action_dist_t.deterministic_sample()
        )
        log_pis_t = torch.unsqueeze(action_dist_t.logp(policy_t), -1)
        action_dist_inputs_tp1, _ = model.get_action_model_outputs(model_out_tp1)
        action_dist_tp1 = action_dist_class(action_dist_inputs_tp1, model)
        policy_tp1 = (
            action_dist_tp1.sample()
            if not deterministic
            else action_dist_tp1.deterministic_sample()
        )
        log_pis_tp1 = torch.unsqueeze(action_dist_tp1.logp(policy_tp1), -1)

        # Q-values for the actually selected actions.
        q_t, _ = model.get_q_values(model_out_t, train_batch[SampleBatch.ACTIONS])
        if policy.config["twin_q"]:
            twin_q_t, _ = model.get_twin_q_values(
                model_out_t, train_batch[SampleBatch.ACTIONS]
            )

        # Q-values for current policy in given current state.
        q_t_det_policy, _ = model.get_q_values(model_out_t, policy_t)
        if policy.config["twin_q"]:
            twin_q_t_det_policy, _ = model.get_twin_q_values(model_out_t, policy_t)
            q_t_det_policy = torch.min(q_t_det_policy, twin_q_t_det_policy)

        # Target q network evaluation.
        q_tp1, _ = target_model.get_q_values(target_model_out_tp1, policy_tp1)
        if policy.config["twin_q"]:
            twin_q_tp1, _ = target_model.get_twin_q_values(
                target_model_out_tp1, policy_tp1
            )
            # Take min over both twin-NNs.
            q_tp1 = torch.min(q_tp1, twin_q_tp1)

        q_t_selected = torch.squeeze(q_t, dim=-1)
        if policy.config["twin_q"]:
            twin_q_t_selected = torch.squeeze(twin_q_t, dim=-1)
        q_tp1 -= alpha * log_pis_tp1

        q_tp1_best = torch.squeeze(input=q_tp1, dim=-1)
        q_tp1_best_masked = (1.0 - train_batch[SampleBatch.DONES].float()) * q_tp1_best

    # compute RHS of bellman equation
    logging.debug(f'loss reward: {train_batch[SampleBatch.REWARDS]}')
    q_t_selected_target = (
        train_batch[SampleBatch.REWARDS]
        + (policy.config["gamma"] ** policy.config["n_step"]) * q_tp1_best_masked
    ).detach()

    # Compute the TD-error (potentially clipped).
    base_td_error = torch.abs(q_t_selected - q_t_selected_target)
    if policy.config["twin_q"]:
        twin_td_error = torch.abs(twin_q_t_selected - q_t_selected_target)
        td_error = 0.5 * (base_td_error + twin_td_error)
    else:
        td_error = base_td_error

    critic_loss = [torch.mean(train_batch[PRIO_WEIGHTS] * huber_loss(base_td_error))]
    if policy.config["twin_q"]:
        critic_loss.append(
            torch.mean(train_batch[PRIO_WEIGHTS] * huber_loss(twin_td_error))
        )

    # Alpha- and actor losses.
    # Note: In the papers, alpha is used directly, here we take the log.
    # Discrete case: Multiply the action probs as weights with the original
    # loss terms (no expectations needed).
    if model.discrete:
        weighted_log_alpha_loss = policy_t.detach() * (
            -model.log_alpha * (log_pis_t + model.target_entropy).detach()
        )
        # Sum up weighted terms and mean over all batch items.
        alpha_loss = torch.mean(torch.sum(weighted_log_alpha_loss, dim=-1))
        # Actor loss.
        actor_loss = torch.mean(
            torch.sum(
                torch.mul(
                    # NOTE: No stop_grad around policy output here
                    # (compare with q_t_det_policy for continuous case).
                    policy_t,
                    alpha.detach() * log_pis_t - q_t.detach(),
                ),
                dim=-1,
            )
        )
    else:
        alpha_loss = -torch.mean(
            model.log_alpha * (log_pis_t + model.target_entropy).detach()
        )
        # Note: Do not detach q_t_det_policy here b/c is depends partly
        # on the policy vars (policy sample pushed through Q-net).
        # However, we must make sure `actor_loss` is not used to update
        # the Q-net(s)' variables.
        actor_loss = torch.mean(alpha.detach() * log_pis_t - q_t_det_policy)

    # Store values for stats function in model (tower), such that for
    # multi-GPU, we do not override them during the parallel loss phase.
    model.tower_stats["q_t"] = q_t
    model.tower_stats["policy_t"] = policy_t
    model.tower_stats["log_pis_t"] = log_pis_t
    model.tower_stats["actor_loss"] = actor_loss
    model.tower_stats["critic_loss"] = critic_loss
    model.tower_stats["alpha_loss"] = alpha_loss

    # TD-error tensor in final stats
    # will be concatenated and retrieved for each individual batch item.
    model.tower_stats["td_error"] = td_error

    # Return all loss terms corresponding to our optimizers.
    return tuple([actor_loss] + critic_loss + [alpha_loss])


def stats(policy: Policy, train_batch: SampleBatch) -> Dict[str, TensorType]:
    """Stats function for SAC. Returns a dict with important loss stats.

    Args:
        policy: The Policy to generate stats for.
        train_batch: The SampleBatch (already) used for training.

    Returns:
        Dict[str, TensorType]: The stats dict.
    """
    q_t = torch.stack(policy.get_tower_stats("q_t"))

    return {
        "actor_loss": torch.mean(torch.stack(policy.get_tower_stats("actor_loss"))),
        "critic_loss": torch.mean(
            torch.stack(tree.flatten(policy.get_tower_stats("critic_loss")))
        ),
        "alpha_loss": torch.mean(torch.stack(policy.get_tower_stats("alpha_loss"))),
        "alpha_value": torch.exp(policy.model.log_alpha),
        "log_alpha_value": policy.model.log_alpha,
        "target_entropy": policy.model.target_entropy,
        "policy_t": torch.mean(torch.stack(policy.get_tower_stats("policy_t"))),
        "mean_q": torch.mean(q_t),
        "max_q": torch.max(q_t),
        "min_q": torch.min(q_t),
    }


def optimizer_fn(policy: Policy, config: AlgorithmConfigDict) -> Tuple[LocalOptimizer]:
    """Creates all necessary optimizers for SAC learning.

    The 3 or 4 (twin_q=True) optimizers returned here correspond to the
    number of loss terms returned by the loss function.

    Args:
        policy: The policy object to be trained.
        config: The Algorithm's config dict.

    Returns:
        Tuple[LocalOptimizer]: The local optimizers to use for policy training.
    """
    policy.actor_optim = torch.optim.Adam(
        params=policy.model.policy_variables(),
        lr=config["optimization"]["actor_learning_rate"],
        eps=1e-7,  # to match tf.keras.optimizers.Adam's epsilon default
    )

    critic_split = len(policy.model.q_variables())
    if config["twin_q"]:
        critic_split //= 2

    policy.critic_optims = [
        torch.optim.Adam(
            params=policy.model.q_variables()[:critic_split],
            lr=config["optimization"]["critic_learning_rate"],
            eps=1e-7,  # to match tf.keras.optimizers.Adam's epsilon default
        )
    ]
    if config["twin_q"]:
        policy.critic_optims.append(
            torch.optim.Adam(
                params=policy.model.q_variables()[critic_split:],
                lr=config["optimization"]["critic_learning_rate"],
                eps=1e-7,  # to match tf.keras.optimizers.Adam's eps default
            )
        )
    policy.alpha_optim = torch.optim.Adam(
        params=[policy.model.log_alpha],
        lr=config["optimization"]["entropy_learning_rate"],
        eps=1e-7,  # to match tf.keras.optimizers.Adam's epsilon default
    )

    return tuple([policy.actor_optim] + policy.critic_optims + [policy.alpha_optim])


# TODO: Unify with DDPG's ComputeTDErrorMixin when SAC policy subclasses PolicyV2
class ComputeTDErrorMixin:
    """Mixin class calculating TD-error (part of critic loss) per batch item.

    - Adds `policy.compute_td_error()` method for TD-error calculation from a
      batch of observations/actions/rewards/etc..
    """

    def __init__(self):
        def compute_td_error(
            obs_t, act_t, rew_t, obs_tp1, done_mask, importance_weights
        ):
            input_dict = self._lazy_tensor_dict(
                {
                    SampleBatch.CUR_OBS: obs_t,
                    SampleBatch.ACTIONS: act_t,
                    SampleBatch.REWARDS: rew_t,
                    SampleBatch.NEXT_OBS: obs_tp1,
                    SampleBatch.DONES: done_mask,
                    PRIO_WEIGHTS: importance_weights,
                }
            )
            # Do forward pass on loss to update td errors attribute
            # (one TD-error value per item in batch to update PR weights).
            actor_critic_loss(self, self.model, None, input_dict)

            # `self.model.td_error` is set within actor_critic_loss call.
            # Return its updated value here.
            return self.model.tower_stats["td_error"]

        # Assign the method to policy (self) for later usage.
        self.compute_td_error = compute_td_error


def setup_late_mixins(
    policy: Policy,
    obs_space: gym.spaces.Space,
    action_space: gym.spaces.Space,
    config: AlgorithmConfigDict,
) -> None:
    """Call mixin classes' constructors after Policy initialization.

    - Moves the target model(s) to the GPU, if necessary.
    - Adds the `compute_td_error` method to the given policy.
    Calling `compute_td_error` with batch data will re-calculate the loss
    on that batch AND return the per-batch-item TD-error for prioritized
    replay buffer record weight updating (in case a prioritized replay buffer
    is used).
    - Also adds the `update_target` method to the given policy.
    Calling `update_target` updates all target Q-networks' weights from their
    respective "main" Q-metworks, based on tau (smooth, partial updating).

    Args:
        policy: The Policy object.
        obs_space (gym.spaces.Space): The Policy's observation space.
        action_space (gym.spaces.Space): The Policy's action space.
        config: The Policy's config.
    """
    ComputeTDErrorMixin.__init__(policy)
    TargetNetworkMixin.__init__(policy)

from ray.rllib.evaluation.episode import Episode
from ray.rllib.utils.typing import (
    AgentID,
    LocalOptimizer,
    ModelGradients,
    TensorType,
    AlgorithmConfigDict,
)
# pretrained_policy.compute_single_action()
# pretrained_sac.compute()
# TODO: Test this

from l5kit.environment.envs.l5_env import L5Env, SimulationConfigGym
from src.customEnv.wrapper import L5EnvWrapper
from l5kit.configs import load_config_data
env_config_path = 'src/configs/gym_config84.yaml'
cfg = load_config_data(env_config_path)
def rllib_model():
    train_envs = 4
    lr = 3e-3
    lr_start = 3e-4
    lr_end = 3e-5
    config_param_space = {
        "env": "L5-CLE-V1",
        "framework": "torch",
        "num_gpus": 0,
        # "num_workers": 63,
        "num_envs_per_worker": train_envs,
        'q_model_config' : {
                # "dim": 112,
                # "conv_filters" : [[64, [7,7], 3], [32, [11,11], 3], [32, [11,11], 3]],
                # "conv_activation": "relu",
                "post_fcnet_hiddens": [256],
                "post_fcnet_activation": "relu",
            },
        'policy_model_config' : {
                # "dim": 112,
                # "conv_filters" : [[64, [7,7], 3], [32, [11,11], 3], [32, [11,11], 3]],
                # "conv_activation": "relu",
                "post_fcnet_hiddens": [256],
                "post_fcnet_activation": "relu",
            },
        'tau': 0.005,
        'target_network_update_freq': 1,
        'replay_buffer_config':{
            'type': 'MultiAgentPrioritizedReplayBuffer',
            'capacity': int(1e5),
            "worker_side_prioritization": True,
        },
        'num_steps_sampled_before_learning_starts': 8000,
        
        'target_entropy': 'auto',
    #     "model": {
    #         "custom_model": "GN_CNN_torch_model",
    #         "custom_model_config": {'feature_dim':128},
    #     },
        '_disable_preprocessor_api': True,
        "eager_tracing": True,
        "restart_failed_sub_environments": True,
    
        # 'train_batch_size': 4000,
        # 'sgd_minibatch_size': 256,
        # 'num_sgd_iter': 16,
        # 'store_buffer_in_checkpoints' : False,
        'seed': 42,
        'batch_mode': 'truncate_episodes',
        "rollout_fragment_length": 1,
        'train_batch_size': 2048,
        'training_intensity' : 32, # (4x 'natural' value = 8)
        'gamma': 0.8,
        'twin_q' : True,
        "lr": 3e-4,
        "min_sample_timesteps_per_iteration": 8000,
    }
    from ray import tune
    rollout_sim_cfg = SimulationConfigGym()
    rollout_sim_cfg.num_simulation_steps = None

    env_kwargs = {'env_config_path': env_config_path, 
                'use_kinematic': True, 
                'sim_cfg': rollout_sim_cfg,  
                'train': True, 
                'return_info': True}

    rollout_env = L5EnvWrapper(env = L5Env(**env_kwargs), \
                            raster_size= cfg['raster_params']['raster_size'][0], \
                            n_channels = 7,)
    tune.register_env("L5-CLE-V2", 
                    lambda config: L5EnvWrapper(env = L5Env(**env_kwargs), \
                                                raster_size= cfg['raster_params']['raster_size'][0], \
                                                n_channels = 7))
    from ray.rllib.algorithms.sac import SAC
    # checkpoint_path = 'l5kit/ray_results/01-01-2023_15-53-49/SAC/SAC_L5-CLE-V1_cf7bb_00000_0_2023-01-01_08-53-50/checkpoint_000170'
    # checkpoint_path = '/home/pronton/ray_results/31-12-2022_07-53-04/SAC/SAC_L5-CLE-V1_7bae1_00000_0_2022-12-31_00-53-04/checkpoint_000360'
    checkpoint_path = '/home/pronton/ray_results/31-12-2022_07-53-04(SAC ~-30)/SAC/SAC_L5-CLE-V1_7bae1_00000_0_2022-12-31_00-53-04/checkpoint_000360'
    algo = SAC(config=config_param_space, env='L5-CLE-V2')
    algo.restore(checkpoint_path)
    return rollout_env, algo

_, pretrained_sac_model = rllib_model()

pretrained_policy = pretrained_sac_model.get_policy()
# device = next(pretrained_policy.model.parameters()).device
# logging.debug(f'>>>>>>>>>>>>>>>>>>>>..pretrained sac policy device:{device}')

import numpy as np
from ray.rllib.models.torch.torch_distributions import TorchDiagGaussian
kl_div_weight = 0.1
raster_size = 84
n_channels = 7

def custom_postprocess_trajectory(policy: Policy,
    sample_batch: SampleBatch,
    other_agent_batches: Optional[Dict[AgentID, SampleBatch]] = None,
    episode: Optional[Episode] = None,
) -> SampleBatch:
    # device = pretrained_sac_model()
    logging.debug('somessssssssssssssssss')
    # change reward here
    # device = next(pretrained_policy.model.parameters()).device
    # logging.debug(f'>>>>>>>>>>>>>>>>>>>>..pretrained sac policy device:{device}')
    # device = next(policy.model.parameters()).device
    # logging.debug(f'>>>>>>>>>>>>>>>>>>>>.training sac policy device: {device}')
    # device = 'cuda'
    obs = sample_batch[SampleBatch.CUR_OBS]
    if type(obs) == Dict:
        obs = {k: torch.as_tensor(v) for k, v in sample_batch[SampleBatch.CUR_OBS].items()}
    elif type(obs) == np.ndarray:
        obs = torch.as_tensor(obs)
    assert obs.shape[1:] == (84,84,7), f'{obs.shape[1:]} != (B, 84,84,7)'
    pretrained_logits = pretrained_policy.compute_actions_from_input_dict({'obs': obs.view(-1, raster_size, raster_size, n_channels)})[2]['action_dist_inputs']
    assert pretrained_logits.shape[1] == 6, 'Not (B, 6)'

    # compute 2 action dist
    logging.debug(f'pretrain action logits: {pretrained_logits}, device: {torch.as_tensor(pretrained_logits).device}')
    logging.debug(f'training action logits: {sample_batch[SampleBatch.ACTION_DIST_INPUTS]}, device: {torch.as_tensor(sample_batch[SampleBatch.ACTION_DIST_INPUTS]).device}')
    logging.debug(f'-----------------------')
    
    pretrained_action_dist = pretrained_policy.dist_class(torch.as_tensor(pretrained_logits), pretrained_policy.model)
    sac_action_dist =policy.dist_class(torch.as_tensor(sample_batch[SampleBatch.ACTION_DIST_INPUTS]), policy.model)

    # compute kl div
    # kl_div = sac_action_dist.kl(pretrained_action_dist)
    pretrained_action_sample = pretrained_action_dist.deterministic_sample()
    pretrained_logp =pretrained_action_dist.logp(pretrained_action_sample)
    sac_action_sample = sac_action_dist.deterministic_sample()
    sac_logp =sac_action_dist.logp(sac_action_sample)

    logging.debug(f'pretrain action deterministic sample: {pretrained_action_sample}')
    logging.debug(f'training action deterministic sample: {sac_action_sample}')
    logging.debug(f'-----------------------')

    logging.debug(f'pretrain action logp sample: {pretrained_logp}')
    logging.debug(f'training action logp sample: {sac_logp}')
    logging.debug(f'-----------------------')


    kl_div =(sac_logp- pretrained_logp)
    # print('kl_div', kl_div)

    # print(f'reward before: {sample_batch[SampleBatch.REWARDS]},\
    #       shape: {sample_batch[SampleBatch.REWARDS].shape}')
    # logging.debug(f'reward shape{sample_batch[SampleBatch.REWARDS].shape}')
    logging.debug(f'kl_div: {kl_div}')
    kl_div = kl_div.cpu().numpy()
    # self.kl_il_rl = kl_div.mean()
    #logging.debug('kl div:', kl_div* self.kl_div_weight)
    # self.rs_after = kl_div.cpu().numpy().mean()
    sample_batch[SampleBatch.REWARDS] -=  kl_div* kl_div_weight
    # print( sample_batch[SampleBatch.REWARDS].device)
    # self.regularized_rewards= sample_batch[SampleBatch.REWARDS]

    # print(f'reward after: {sample_batch[SampleBatch.REWARDS]},\
    #       shape: {sample_batch[SampleBatch.REWARDS].shape}')

    # if type(logits) == Dict: 
    #     pred_x = logits['positions'][:,0, 0].view(-1,1)# take the first action 
    #     pred_y = logits['positions'][:,0, 1].view(-1,1)# take the first action
    #     pred_yaw = logits['yaws'][:,0,:].view(-1,1)# take the first action
    # else: # np.ndarray type
    #     batch_size = len(obs)
    #     predicted = logits.view(batch_size, -1, 3) # B, N, 3 (X,Y,yaw)
    #     pred_x = predicted[:, 0, 0].view(-1,1) # take the first action 
    #     pred_y = predicted[:, 0, 1].view(-1,1) # take the first action
    #     pred_yaw = predicted[:, 0, 2].view(-1,1)# take the first action
    # ones = torch.ones_like(pred_x) 

    # # lx, ly, lyaw= policy.model.log_std_x, policy.model.log_std_y, policy.model.log_std_yaw
    # output_logits = torch.cat((pred_x, pred_y, pred_yaw), dim = -1)
    # model.get_policy().compute_actions_from_input_dict({'obs':obs.reshape(-1,84,84,7)})
    # lx, ly, lyaw = -5, -5, -5
    # output_logits_std = torch.cat((ones*lx, ones * ly, ones * lyaw), dim = -1)
    
    # # 
    # pretrained_action_dist = TorchDiagGaussian(loc=output_logits, scale=torch.exp(output_logits_std))
    # pretrained_action_dist = TorchSquashedGaussian(inputs= output_logits, scale=torch.exp(output_logits_std))

    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print('device:', device)
    # ppo_logits, _ = self.model.forward({'obs': obs.to(device)} ,None,None)
    # print(f'ppo logits: {ppo_logits}')
    # print('----------------')
    # # ppo_logits = ppo_logits.to(self.device)
    # assert ppo_logits.shape[1] == 6, f'{ppo_logits.shape} != torch.Size([x,6])'
    # ppo_action_dist =policy.dist_class(torch.as_tensor(sample_batch[SampleBatch.ACTION_DIST_INPUTS]).to(device), policy.model)

    
    
    # Create a distribution from the pretrained model
    # pretrained_dist = pretrained_action_dist.sample()
    
    # Calculate the KL divergence between the PPO and pretrained distributions
    # kl_div = torch.distributions.kl_divergence(
    #     self.action_dist, pretrained_dist).mean()

    # print(f'ppo action dist sample {ppo_action_dist.dist.sample()}')
    # print(f'pretrain action dist sample {pretrained_action_dist.dist.sample()}')
    # print('----------------')
    # logging.debug(f'ppo action dist {ppo_action_dist.dist}')
    # logging.debug(f'pretrain action dist {pretrained_action_dist.dist}')
    # kl_div = ppo_action_dist.kl(pretrained_action_dist)
    # # print(f'kl_div: {kl_div}, shape: {kl_div.shape}')
    # # kl_div = pretrained_action_dist.kl(ppo_action_dist)
    # # kl_div = kl_divergence(pretrained_dist)
    # # reversed_kl_div = pretrained_action_dist.kl(ppo_action_dist)
    # # print(f'reversed kl_div: {reversed_kl_div}, shape: {reversed_kl_div.shape}')
    # # print('----------------')
    
    # # Add the KL penalty to the rewards
    # # self.my_cur_rewards = sample_batch[SampleBatch.REWARDS]
    # print(f'reward before: {sample_batch[SampleBatch.REWARDS]},\
    #       shape: {sample_batch[SampleBatch.REWARDS].shape}')
    # # logging.debug(f'reward shape{sample_batch[SampleBatch.REWARDS].shape}')
    # # logging.debug(f'kl shape{kl_div.shape}, kl_div: {kl_div}')
    # kl_div = kl_div.cpu().numpy()
    # # self.kl_il_rl = kl_div.mean()
    # #logging.debug('kl div:', kl_div* self.kl_div_weight)
    # # self.rs_after = kl_div.cpu().numpy().mean()
    # sample_batch[SampleBatch.REWARDS] -=  kl_div* kl_div_weight
    # # print( sample_batch[SampleBatch.REWARDS].device)
    # self.regularized_rewards= sample_batch[SampleBatch.REWARDS]

    # print(f'reward after: {sample_batch[SampleBatch.REWARDS]},\
    #       shape: {sample_batch[SampleBatch.REWARDS].shape}')

        # print('----------------')
    # sample_batch[SampleBatch.REWARDS] = 

    logging.debug('somessssssssssssssssss')
    return postprocess_trajectory(policy, sample_batch, other_agent_batches, episode = None,)

# Build a child class of `TorchPolicy`, given the custom functions defined
# above.

KLSACTorchPolicy = build_policy_class(
    name="KLSACTorchPolicy",
    framework="torch",
    loss_fn=actor_critic_loss,
    get_default_config=lambda: ray.rllib.algorithms.sac.sac.DEFAULT_CONFIG,
    stats_fn=stats,
    postprocess_fn=custom_postprocess_trajectory,
    extra_grad_process_fn=apply_grad_clipping,
    optimizer_fn=optimizer_fn,
    validate_spaces=validate_spaces,
    before_loss_init=setup_late_mixins,
    make_model_and_action_dist=build_sac_model_and_action_dist,
    extra_learn_fetches_fn=concat_multi_gpu_td_errors,
    mixins=[TargetNetworkMixin, ComputeTDErrorMixin],
    action_distribution_fn=action_distribution_fn,
)