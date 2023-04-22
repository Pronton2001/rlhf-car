import logging
from typing import Dict, List, Type, Union

import ray
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.algorithms.ppo.ppo_tf_policy import validate_config
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.models.torch.torch_distributions import TorchDiagGaussian
from ray.rllib.models.torch.torch_action_dist import TorchDiagGaussian as TorchActionDiagGaussian
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TensorType
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.numpy import convert_to_numpy

import numpy as np
torch, nn = try_import_torch()

logger = logging.getLogger(__name__)


class KLPPOTorchPolicy(
    # ValueNetworkMixin,
    # LearningRateSchedule,
    # EntropyCoeffSchedule,
    # KLCoeffMixin,
    # TorchPolicyV2,
    PPOTorchPolicy,
):
    """PyTorch policy class used with PPO."""

    def __init__(self, observation_space, action_space, config):
        self.kl_div_weight = 0.1
        config = dict(ray.rllib.algorithms.ppo.ppo.PPOConfig().to_dict(), **config)
        # TODO: Move into Policy API, if needed at all here. Why not move this into
        #  `PPOConfig`?.
        validate_config(config)
        self.pretrained_policy = None
        self.kl_il_rl = 0
        self.regularized_rewards = 0
        self.my_cur_rewards= 0

        config["model"]["max_seq_len"] = 20
        # print('init pretrained:', device)
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.pretrained_policy.to(device)
        # print('init device:', device)
        PPOTorchPolicy.__init__(
            self,
            observation_space,
            action_space,
            config,
        )
        self.pretrained_policy = self.model.pretrained_policy # NOTE: cannot move to def __init__() (some bugs)
        self.device = next(self.pretrained_policy.parameters()).device
        print('>>>>>>>>>>> pretrained KL PPO Torch:', self.device)
        self.kl_il_rl = 0
        self.regularized_rewards = 0
        self.my_cur_rewards= 0
        

    @override(PPOTorchPolicy)
    def stats_fn(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        return convert_to_numpy(
            {
                "cur_kl_coeff": self.kl_coeff,
                "cur_lr": self.cur_lr,
                "total_loss": torch.mean(
                    torch.stack(self.get_tower_stats("total_loss"))
                ),
                "policy_loss": torch.mean(
                    torch.stack(self.get_tower_stats("mean_policy_loss"))
                ),
                "vf_loss": torch.mean(
                    torch.stack(self.get_tower_stats("mean_vf_loss"))
                ),
                "vf_explained_var": torch.mean(
                    torch.stack(self.get_tower_stats("vf_explained_var"))
                ),
                "kl": torch.mean(torch.stack(self.get_tower_stats("mean_kl_loss"))),
                "entropy": torch.mean(
                    torch.stack(self.get_tower_stats("mean_entropy"))
                ),
                "entropy_coeff": self.entropy_coeff,
                # FIXME: It do not change value
                # "kl_human_e": self.kl_il_rl,
                # "ep_reward": self.my_cur_rewards,
                # "regularized_rewards": self.regularized_rewards,
            }
        )

    @override(PPOTorchPolicy)
    def postprocess_trajectory(
        self, sample_batch, other_agent_batches=None, episode=None
    ):
        # Do all post-processing always with no_grad().
        # Not using this here will introduce a memory leak
        # in torch (issue #6962).
        # TODO: no_grad still necessary?
        with torch.no_grad():
            if self.pretrained_policy == None:
                return super().postprocess_trajectory(
                    sample_batch, other_agent_batches, episode)

            # Convert observations to tensor
            # logging.debug('obs', sample_batch['obs'])
            # print(sample_batch['obs'])
            # obs = torch.as_tensor(sample_batch['obs']).to(device)

            # Compute the KL divergence reward
            # ppo_action_dist = self.action_dist(sample_batch['obs'])
            # ppo_logits, state = self.model(sample_batch)
            # ppo_action_dist = self.dist_class(ppo_logits, self.model)
            # ppo_action_dist = TorchActionDiagGaussian(
            # , self.model)

            # pretrained_action_dist = pretrained_policy.dist_class(pretrained_policy.model.outputs, pretrained_policy.model)

            # logging.debug('ppo traj' + str(self.model.outputs))
            # obs = {}
            # for k,v in sample_batch[SampleBatch.CUR_OBS].items():
            #     obs[k] = torch.as_tensor(v).to('cpu')

            # Cal distribution of pretrain model 's action
            obs = sample_batch[SampleBatch.CUR_OBS]
            if type(obs) == Dict:
                obs = {k: torch.as_tensor(v).to(self.device) for k, v in sample_batch[SampleBatch.CUR_OBS].items()}
            elif type(obs) == np.ndarray:
                obs = torch.as_tensor(obs).to(self.device)
            logits = self.pretrained_policy(obs)
            # print('pretrained device:', self.device)

            if type(logits) == Dict: # TODO: Change Vectorized output from dict -> numpy.ndarray
                pred_x = logits['positions'][:,0, 0].view(-1,1)# take the first action 
                pred_y = logits['positions'][:,0, 1].view(-1,1)# take the first action
                pred_yaw = logits['yaws'][:,0,:].view(-1,1)# take the first action
            else: # np.ndarray type
                batch_size = len(obs)
                predicted = logits.view(batch_size, -1, 3) # B, N, 3 (X,Y,yaw)
                pred_x = predicted[:, 0, 0].view(-1,1) # take the first action 
                pred_y = predicted[:, 0, 1].view(-1,1) # take the first action
                pred_yaw = predicted[:, 0, 2].view(-1,1)# take the first action
            ones = torch.ones_like(pred_x) 

            lx, ly, lyaw= self.model.log_std_x, self.model.log_std_y, self.model.log_std_yaw
            output_logits = torch.cat((pred_x,pred_y, pred_yaw), dim = -1)
            output_logits_std = torch.cat((ones*lx, ones * ly, ones * lyaw), dim = -1)
            
            pretrained_action_dist = TorchDiagGaussian(loc=output_logits, scale=torch.exp(output_logits_std))

            
            # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # print('device:', device)
            # ppo_logits, _ = self.model.forward({'obs': obs.to(device)} ,None,None)
            # print(f'ppo logits: {ppo_logits}')
            # print('----------------')
            # # ppo_logits = ppo_logits.to(self.device)
            # assert ppo_logits.shape[1] == 6, f'{ppo_logits.shape} != torch.Size([x,6])'
            ppo_action_dist = self.dist_class(torch.as_tensor(sample_batch[SampleBatch.ACTION_DIST_INPUTS]).to(self.device), self.model)

            
            
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
            kl_div = ppo_action_dist.kl(pretrained_action_dist)
            # print(f'kl_div: {kl_div}, shape: {kl_div.shape}')
            # kl_div = pretrained_action_dist.kl(ppo_action_dist)
            # kl_div = kl_divergence(pretrained_dist)
            # reversed_kl_div = pretrained_action_dist.kl(ppo_action_dist)
            # print(f'reversed kl_div: {reversed_kl_div}, shape: {reversed_kl_div.shape}')
            # print('----------------')
            
            # Add the KL penalty to the rewards
            # self.my_cur_rewards = sample_batch[SampleBatch.REWARDS]
            # print(f'reward before: {sample_batch[SampleBatch.REWARDS]},\
                #   shape: {sample_batch[SampleBatch.REWARDS].shape}')
            # logging.debug(f'reward shape{sample_batch[SampleBatch.REWARDS].shape}')
            # logging.debug(f'kl shape{kl_div.shape}, kl_div: {kl_div}')
            kl_div = kl_div.cpu().numpy()
            # self.kl_il_rl = kl_div.mean()
            logging.debug(f'kl_div {kl_div[0]}, shape: {kl_div.shape}')
            #logging.debug('kl div:', kl_div* self.kl_div_weight)
            # self.rs_after = kl_div.cpu().numpy().mean()
            sample_batch[SampleBatch.REWARDS] -=  kl_div* self.kl_div_weight
            # print( sample_batch[SampleBatch.REWARDS].device)
            # self.regularized_rewards= sample_batch[SampleBatch.REWARDS]

            # print(f'reward after: {sample_batch[SampleBatch.REWARDS]},\
                #   shape: {sample_batch[SampleBatch.REWARDS].shape}')

            # print('----------------')
            return super().postprocess_trajectory(
                sample_batch, other_agent_batches, episode)
