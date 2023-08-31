"""
PyTorch policy class used for SAC.
"""
import time

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
from src.customEnv.action_utils import inverseUnicycle, standard_normalizer_kin, standard_normalizer_nonKin
from src.customModel.utils import kl_mc, kl_approx
torch, nn = try_import_torch()
F = nn.functional

logger = logging.getLogger(__name__)


# logging.basicConfig(filename=SRC_PATH + 'src/log/info.log', level=logging.DEBUG, filemode='w')

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

from l5kit.environment.envs.l5_env import L5Env, SimulationConfigGym
from src.customEnv.wrapper import L5EnvWrapper
from l5kit.configs import load_config_data

def getBCPretrained():
    model_path = f"{SRC_PATH}src/model/OL_HS.pt"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    model = torch.load(model_path).to(device)
    # model = SAC.load("/home/pronton/rl/l5kit/examples/RL/gg colabs/logs/SAC_640000_steps.zip")
    model = model.eval()
    # torch.set_grad_enabled(False)
    print('-----------------------------------> device', device)
    for  name, param in model.named_parameters():
        param.requires_grad = False
    return model
pretrained_policy = getBCPretrained()


import numpy as np

device = next(pretrained_policy.parameters()).device
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

    ############## CHANGE HERE ##############
    # alpha = torch.exp(model.log_alpha)
    m_entropy, m_kl, use_entropy_kl_params = policy.model.action_model.m_entropy, policy.model.action_model.m_kl, policy.model.action_model.use_entropy_kl_params
    m_l0 = policy.model.action_model.m_l0
    sac_entropy_equal_m_entropy = policy.model.action_model.sac_entropy_equal_m_entropy
    if use_entropy_kl_params:
        m_tau = m_entropy + m_kl
        m_alpha = m_kl/m_tau
    else:
        m_tau, m_alpha = policy.model.action_model.m_tau, policy.model.action_model.m_alpha
        m_entropy = m_tau * (1-m_alpha)
    if sac_entropy_equal_m_entropy:
        alpha = m_entropy
    else:
        alpha = m_tau
    logging.debug(f'alpha: {alpha}, m_tau = {m_tau}, m_alpha: {m_alpha}')

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
        # log_pis_t: log of RL policy
        # log_pi0s_t: log of Behavior Policy
        # s = time.time()
        ##############################################################
        obs = train_batch[SampleBatch.CUR_OBS]
        if type(obs) == dict:
            obs = {k: torch.as_tensor(v).to(device) for k, v in train_batch[SampleBatch.CUR_OBS].items()}
        elif type(obs) == np.ndarray:
            obs = torch.as_tensor(obs).to(device)

        ##############################################################

        logits = pretrained_policy(obs) # check this
        if type(logits) == dict: # TODO: Change Vectorized output from dict -> numpy.ndarray
            pred_x = logits['positions'][:,0, 0].view(-1,1)# take the first action 
            pred_y = logits['positions'][:,0, 1].view(-1,1)# take the first action
            pred_yaw = logits['yaws'][:,0,:].view(-1,1)# take the first action
        else: # np.ndarray type
            batch_size = len(obs)
            predicted = logits.view(batch_size, -1, 3) # B, N, 3 (X,Y,yaw)
            pred_x = predicted[:, 0, 0].view(-1,1) # take the first action 
            pred_y = predicted[:, 0, 1].view(-1,1) # take the first action
            pred_yaw = predicted[:, 0, 2].view(-1,1)# take the first action
            
        ##############################################################

        # convert to acc, steer
        output_logits = inverseUnicycle(pred_x, pred_y, pred_yaw, obs['old_speed']) # B, 1
        # logging.debug(f'inverse Unicycle: {output_logits}')
        # steer, acc = actions['steer'], actions['acc']
        
        # print('-----------------------------> before', output_logits)
        output_logits = standard_normalizer_kin(output_logits) # scale actions
        # logging.debug(f'normalize actions: {output_logits}')
        # print('-----------------------------> after normalize', output_logits)

        ones = torch.ones_like(pred_x) 

        std_acc, std_steer= policy.model.action_model.log_std_acc, policy.model.action_model.log_std_steer
        output_logits = torch.cat((output_logits, ones * std_steer, ones * std_acc), dim = -1).to('cpu')
        # output_logits_std = torch.cat((ones*lx, ones * ly, ones * lyaw), dim = -1)
        
        # pretrained_action_dist = TorchDiagGaussian(output_logits.to(policy_t.device), None)
        pretrained_action_dist2 = TorchSquashedGaussian(output_logits.to(policy_t.device), None)
        # log_pi0s_t = torch.unsqueeze(pretrained_action_dist.logp(policy_t), -1)
        # logging.debug(f'time:{time.time() - s}')
        ##############################################################
        
        log_pis_t = torch.unsqueeze(action_dist_t.logp(policy_t), -1) 
        # log_pis_t = log_pis_t + log_pis_t - log_pi0s_t # convert from entroy -> entropy + KL
        action_dist_inputs_tp1, _ = model.get_action_model_outputs(model_out_tp1)
        action_dist_tp1 = action_dist_class(action_dist_inputs_tp1, model)
        policy_tp1 = (
            action_dist_tp1.sample()
            if not deterministic
            else action_dist_tp1.deterministic_sample()
        )
        # log_pi0s_tp1 = torch.unsqueeze(pretrained_action_dist.logp(policy_tp1), -1)
        log_pis_tp1 = torch.unsqueeze(action_dist_tp1.logp(policy_tp1), -1)
        # log_pis_tp1 = log_pis_tp1 + log_pis_tp1- log_pi0s_tp1 # convert from entroy -> entropy + KL

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
    # logging.debug(f'loss reward: {train_batch[SampleBatch.REWARDS]}')
    q_t_selected_target = (
        train_batch[SampleBatch.REWARDS]
        + (policy.config["gamma"] ** policy.config["n_step"]) * q_tp1_best_masked
    ).detach()
    with torch.no_grad(): # MUNCHAUSEN with behavior policy
        # log_pi0s_t = pretrained_action_dist.logp(policy_t)
        log_pi0s_t2 = pretrained_action_dist2.logp(policy_t)
        # logging.debug(f'log_pis_t: {log_pis_t}')
        # logging.debug(f'log_pi0s_t: {log_pi0s_t}')
        # logging.debug(f'log_pi0s_t2: {log_pi0s_t2}')
        # logging.debug(f'q1: {M_ALPHA * torch.clamp(M_TAU * log_pi0s_t, M_L0, 0)}')
        # logging.debug(f'q2: {M_ALPHA * torch.clamp(M_TAU * log_pi0s_t2, M_L0, 0)}')
        # q_t_selected_target += M_ALPHA * torch.clamp(M_TAU * log_pi0s_t2, M_L0, 0)
        q_t_selected_target += m_alpha * torch.clamp(m_tau * log_pi0s_t2, m_l0, 0) # detach => no grad ?

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
        ############## CHANGE HERE ##############
        # alpha_loss = -torch.mean(
        #     model.log_alpha * (log_pis_t + model.target_entropy).detach()
        # )
        alpha_loss = torch.tensor(0.0).to(log_pis_t.device)
        # Note: Do not detach q_t_det_policy here b/c is depends partly
        # on the policy vars (policy sample pushed through Q-net).
        # However, we must make sure `actor_loss` is not used to update
        # the Q-net(s)' variables.
        ############## CHANGE HERE ##############
        # actor_loss = torch.mean(alpha.detach() * log_pis_t - q_t_det_policy)
        actor_loss = torch.mean(alpha * log_pis_t - q_t_det_policy)

    # Store values for stats function in model (tower), such that for
    # multi-GPU, we do not override them during the parallel loss phase.
    model.tower_stats["q_t"] = q_t
    model.tower_stats["policy_t"] = policy_t
    model.tower_stats["log_pis_t(entropy + KL)"] = log_pis_t
    model.tower_stats["actor_loss"] = actor_loss
    model.tower_stats["critic_loss"] = critic_loss
    ############## CHANGE HERE ##############
    # model.tower_stats["alpha_loss"] = alpha_loss

    # TD-error tensor in final stats
    # will be concatenated and retrieved for each individual batch item.
    model.tower_stats["td_error"] = td_error

    # Return all loss terms corresponding to our optimizers.
    ############## CHANGE HERE ##############
    # return tuple([actor_loss] + critic_loss + [alpha_loss])
    return tuple([actor_loss] + critic_loss)


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
        # "alpha_loss": torch.mean(torch.stack(policy.get_tower_stats("alpha_loss"))),
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
    ############## CHANGE HERE ##############
    # policy.alpha_optim = torch.optim.Adam(
    #     params=[policy.model.log_alpha],
    #     lr=config["optimization"]["entropy_learning_rate"],
    #     eps=1e-7,  # to match tf.keras.optimizers.Adam's epsilon default
    # )

    ############## CHANGE HERE ##############
    # return tuple([policy.actor_optim] + policy.critic_optims + [policy.alpha_optim])
    return tuple([policy.actor_optim] + policy.critic_optims)


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




# def custom_postprocess_trajectory(policy: Policy,
#     sample_batch: SampleBatch,
#     other_agent_batches: Optional[Dict[AgentID, SampleBatch]] = None,
#     episode: Optional[Episode] = None,
# ) -> SampleBatch:

#     return postprocess_trajectory(policy, sample_batch, other_agent_batches, episode = None,)

# Build a child class of `TorchPolicy`, given the custom functions defined
# above.

KLSACPolicy = build_policy_class(
    name="KLSACPolicy",
    framework="torch",
    loss_fn=actor_critic_loss,
    get_default_config=lambda: ray.rllib.algorithms.sac.sac.DEFAULT_CONFIG,
    stats_fn=stats,
    postprocess_fn=postprocess_trajectory,
    extra_grad_process_fn=apply_grad_clipping,
    optimizer_fn=optimizer_fn,
    validate_spaces=validate_spaces,
    before_loss_init=setup_late_mixins,
    make_model_and_action_dist=build_sac_model_and_action_dist,
    extra_learn_fetches_fn=concat_multi_gpu_td_errors,
    mixins=[TargetNetworkMixin, ComputeTDErrorMixin],
    action_distribution_fn=action_distribution_fn,
)
