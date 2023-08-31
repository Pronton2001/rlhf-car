import logging
from typing import Dict, List, Type, Union

import ray
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.algorithms.ppo.ppo_tf_policy import validate_config
from ray.rllib.policy.sample_batch import SampleBatch
# from ray.rllib.models.torch.torch_distributions import TorchDiagGaussian
from ray.rllib.models.torch.torch_action_dist import TorchDiagGaussian
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TensorType
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.numpy import convert_to_numpy

import numpy as np
from src.constant import SRC_PATH

from src.customEnv.action_utils import inverseUnicycle, standard_normalizer_kin, standard_normalizer_nonKin
torch, nn = try_import_torch()

logger = logging.getLogger(__name__)

actions = []
class KLRewardPPOTorchPolicy(
    # ValueNetworkMixin,
    # LearningRateSchedule,
    # EntropyCoeffSchedule,
    # KLCoeffMixin,
    # TorchPolicyV2,
    PPOTorchPolicy,
):
    """PyTorch policy class used with PPO."""

    def __init__(self, observation_space, action_space, config):
        self.kl_div_weight = 1
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
        self.pretrained_policy = self.getBCPretrained()
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

    def getBCPretrained(self):
        model_path = f"{SRC_PATH}src/model/OL_HS.pt"
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = torch.load(model_path).to(device)
        # model = SAC.load("/home/pronton/rl/l5kit/examples/RL/gg colabs/logs/SAC_640000_steps.zip")
        model = model.eval()
        # torch.set_grad_enabled(False)
        for  name, param in model.named_parameters():
            param.requires_grad = False
        return model
    @override(PPOTorchPolicy)
    def postprocess_trajectory(
        self, sample_batch, other_agent_batches, episode
    ):
        global actions
        # Do all post-processing always with no_grad().
        # Not using this here will introduce a memory leak
        # in torch (issue #6962).
        # TODO: no_grad still necessary?
        with torch.no_grad():
            if self.pretrained_policy == None:
                return super().postprocess_trajectory(
                    sample_batch, other_agent_batches, episode)

            obs = sample_batch[SampleBatch.CUR_OBS]
            # Convert observations to tensor
            # if type(obs) == dict:
            #     obs = {k: torch.as_tensor(v).to(self.device) for k, v in sample_batch[SampleBatch.CUR_OBS].items()}
            #     # print('-----------------------------> OKOK')
            # elif type(obs) == np.ndarray:
            #     obs = torch.as_tensor(obs).to(self.device)
            if type(obs) == dict:
                obs = {k: torch.as_tensor(v).to(self.device) for k, v in sample_batch[SampleBatch.CUR_OBS].items()}
                # for  k, v in obs.items():
                #     if k == 'agent_polyline_availability':
                #         obs[k] = torch.as_tensor(v.shape[0] * [True, True, False, False], dtype = bool).view(v.shape[0], 4).to(self.device)
                #     else: 
                #         obs[k] = torch.as_tensor(v).to(self.device)
                # obs = {k: torch.as_tensor(v).view(1, *torch.as_tensor(v).shape) for k, v in obs.items()}
                # print('-----------------------------> OKOK')
            elif type(obs) == np.ndarray:
                obs = torch.as_tensor(obs).to(self.device)

            # logging.debug(f"obs: {obs['agent_polyline_availability']}")

            logits = self.pretrained_policy(obs)

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
            ones = torch.ones_like(pred_x) 

            # lx, ly, lyaw= self.model.log_std_x, self.model.log_std_y, self.model.log_std_yaw
            # output_logits = torch.cat((pred_x,pred_y, pred_yaw), dim = -1)
            # Normalize actions
            # output_logits = standard_normalizer_nonKin(output_logits)
            # output_logits_std = torch.cat((ones*lx, ones * ly, ones * lyaw), dim = -1)
            # output_logits = torch.cat((pred_x, pred_y, pred_yaw),  dim = -1)
            # convert to acc, steer
            output_logits = inverseUnicycle(pred_x, pred_y, pred_yaw, obs['old_speed']) # B, 1
            # logging.debug(f'inverse Unicycle [low,high]: {output_logits}')
            # logging.debug(f'old v: {obs["old_speed"]}')
            actions.append(output_logits.detach().cpu().numpy().reshape(-1))
            # steer, acc = actions['steer'], actions['acc']
            
            # print('-----------------------------> before', output_logits)
            output_logits = standard_normalizer_kin(output_logits) # scale actions
            # logging.debug(f'normalize actions [-1,1]: {output_logits}')
            # print('-----------------------------> after normalize', output_logits)
            
            std_acc, std_steer= self.model.log_std_acc, self.model.log_std_steer
            output_logits = torch.cat((output_logits, ones * std_steer, ones * std_acc), dim = -1)
            
            pretrained_action_dist = TorchDiagGaussian(output_logits, None)
            ppo_action_dist = self.dist_class(torch.as_tensor(sample_batch[SampleBatch.ACTION_DIST_INPUTS]).to(self.device), self.model)
            # logging.debug(f'SampleBatch.ACTION_DIST_INPUTS: {sample_batch[SampleBatch.ACTION_DIST_INPUTS]}')

            kl_div = ppo_action_dist.kl(pretrained_action_dist)
            kl_div = kl_div.cpu().numpy()
            # self.kl_il_rl = kl_div.mean()
            # logging.debug(f'kl_div {kl_div}, shape: {kl_div.shape}')
            # self.rs_after = kl_div.cpu().numpy().mean()
            sample_batch[SampleBatch.REWARDS] -=  kl_div* self.model.kl_div_weight 
            # print( sample_batch[SampleBatch.REWARDS].device)
            # self.regularized_rewards= sample_batch[SampleBatch.REWARDS]

            return super().postprocess_trajectory(
                sample_batch, other_agent_batches, episode)
