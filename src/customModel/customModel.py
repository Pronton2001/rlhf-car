from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch.nn as nn
import torch
from torch.nn import functional as F
from src.customEnv.normalize_action import standard_normalizer
from l5kit.planning.vectorized.open_loop_model import VectorizedModel, CustomVectorizedModel
from l5kit.planning.rasterized.model import RasterizedPlanningModelFeature
from torchvision.models.resnet import resnet18, resnet50
import os
from src.constant import SRC_PATH
import numpy as np
# os.environ['CUDA_VISIBLE_DEVICES']= '0'

import logging
logging.basicConfig(filename=SRC_PATH + 'src/log/info2.log', level=logging.DEBUG, filemode='w')
from ray.rllib.algorithms.sac.sac_torch_model import SACTorchModel
import gym
from gym.spaces import Box, Discrete
import numpy as np
import tree  # pip install dm_tree
from typing import Dict, List, Optional

from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.spaces.simplex import Simplex
from ray.rllib.utils.typing import ModelConfigDict, TensorType, TensorStructType

from l5kit.environment import models
import warnings
torch, nn = try_import_torch()

# class TorchRasterQNet(TorchModelV2):
#     def __init__(self, obs_space, action_space, num_outputs, model_config, name):
#             super().__init__(obs_space, action_space, num_outputs, model_config, name)


class TorchRasterNet(TorchModelV2, nn.Module): #TODO: recreate rasternet for PPO, check again
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.log_std_x = -5
        self.log_std_y = -5
        self.log_std_yaw = -5
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # print('torchrasternet init device', self.device)

        # cfg = model_config["custom_model_config"]['cfg'] # TODO: Pass necessary params, not cfg
        future_num_frames = model_config["custom_model_config"]['future_num_frames']
        freeze_actor = model_config["custom_model_config"]['freeze_actor'] 
        self.non_kin_rescale = model_config["custom_model_config"]['non_kin_rescale'] 
        # d_model = model_config["custom_model_config"]['critic_feature_dim'] 
        self.pretrained_policy = RasterizedPlanningModelFeature(
            model_arch="resnet50",
            num_input_channels=5,
            num_targets=3 * future_num_frames,  # feature dim of critic
            weights_scaling=[1., 1., 1.],
            criterion=nn.MSELoss(reduction="none"),)
        self._actor_head =RasterizedPlanningModelFeature(
            model_arch="resnet50",
            num_input_channels=5,
            num_targets=3 * future_num_frames,  # feature dim of critic
            weights_scaling=[1., 1., 1.],
            criterion=nn.MSELoss(reduction="none"),)

        critic_same_as_actor = False
        if critic_same_as_actor:
            d_model = 256 # NOTE: Old version, num_targets = d_model and require critic_FF
            self._critic_head =RasterizedPlanningModelFeature(
                model_arch="resnet50",
                num_input_channels=5,
                num_targets=1,  # feature dim of critic
                weights_scaling=[1., 1., 1.],
                criterion=nn.MSELoss(reduction="none"),)
        else:
            # d_model = 128
            self._critic_head = RasterizedPlanningModelFeature(
                model_arch="simple_cnn",
                num_input_channels=5,
                num_targets=1,  # feature dim of critic
                weights_scaling=[1., 1., 1.],
                criterion=nn.MSELoss(reduction="none"),)

        model_path = SRC_PATH + "src/model/planning_model_20201208.pt"
        self.pretrained_policy.load_state_dict(torch.load(model_path).state_dict())
        self._actor_head.load_state_dict(torch.load(model_path).state_dict()) # NOTE: somehow actor_head in cuda device
        device = next(self.pretrained_policy.parameters()).device
        print('>>>>>>>>>>>>>>pretrained_cuda:', device)
        device = next(self._actor_head.parameters()).device
        print('>>>>>>>>>>>>>>actor_head device:', device)
        device = next(self._critic_head.parameters()).device
        print('>>>>>>>>>>>>>>critic_head device:', device)
        if freeze_actor:
            for param in self._actor_head.parameters():
                param.requires_grad = False
    
    def forward(self, input_dict, state, seq_lens):
        obs_transformed = input_dict['obs']
        logits = self._actor_head(obs_transformed)
        batch_size = len(input_dict)
        pretrained_logits = logits.view(batch_size, -1, 3) # B, N, 3 (X,Y,yaw)
        action = pretrained_logits[:,0,:].view(-1,3)

        # print(f'predicted actions: {action}, shape: {action.shape}')
        action = standard_normalizer(self.non_kin_rescale, action) # take first action
        # print(f'rescaled actions: {action}, shape: {action.shape}')
        ones = torch.ones(batch_size,1).to(action.device) # 32,

        # print(ones.device, action.device)
        output_logits = torch.cat((action, ones * self.log_std_x, ones * self.log_std_y, ones * self.log_std_yaw), dim = -1)
        # assert output_logits.shape[1] == 6, f'{output_logits.shape[1]}'
        value = self._critic_head(obs_transformed)
        self._value = value.view(-1)

        return output_logits, state
    def value_function(self):
        return self._value

class TorchRasterNet2(TorchModelV2, nn.Module): #TODO: recreate rasternet for PPO, check again
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # print('torchrasternet init device', self.device)

        # cfg = model_config["custom_model_config"]['cfg'] # TODO: Pass necessary params, not cfg
        future_num_frames = model_config["custom_model_config"]['future_num_frames']
        freeze_actor = model_config["custom_model_config"]['freeze_actor'] 
        n_channels = model_config["custom_model_config"]['n_channels'] 
        critic_net = model_config["custom_model_config"]['critic_net']
        actor_net = model_config["custom_model_config"]['actor_net']
        # self.non_kin_rescale = model_config["custom_model_config"]['non_kin_rescale'] 
        # d_model = model_config["custom_model_config"]['critic_feature_dim'] 
        self.simple_actor = False

        actor_feature_dim = 3 * future_num_frames
        actor_feature_dim = num_outputs
        crtitic_feature_dim = 1
        self.pretrained_policy = RasterizedPlanningModelFeature(
            model_arch="resnet50",
            num_input_channels=5, # pretrained model input: 5x224x224
            num_targets=3 * 12, 
            weights_scaling=[1., 1., 1.],
            criterion=nn.MSELoss(reduction="none"),)

        self._actor_head =RasterizedPlanningModelFeature(
            model_arch=actor_net,
            num_input_channels=n_channels,
            num_targets= actor_feature_dim , # or same as pretrained_policy to load
            weights_scaling=[1., 1., 1.],
            criterion=nn.MSELoss(reduction="none"),)

        self._critic_head =RasterizedPlanningModelFeature(
            model_arch=critic_net,
            num_input_channels=n_channels,
            num_targets=crtitic_feature_dim,  # feature dim of critic
            weights_scaling=[1., 1., 1.],
            criterion=nn.MSELoss(reduction="none"),)

        if freeze_actor:
            model_path = SRC_PATH + "src/model/planning_model_20201208.pt"
            self.pretrained_policy.load_state_dict(torch.load(model_path).state_dict())
            self._actor_head.load_state_dict(torch.load(model_path).state_dict()) # NOTE: somehow actor_head in cuda device
            for param in self._actor_head.parameters():
                param.requires_grad = False

        device = next(self.pretrained_policy.parameters()).device
        print('>>>>>>>>>>>>>>pretrained device:', device)
        device = next(self._actor_head.parameters()).device
        print('>>>>>>>>>>>>>>actor_head device:', device)
        device = next(self._critic_head.parameters()).device
        print('>>>>>>>>>>>>>>critic_head device:', device)
    
        # self._critic_mlp = nn.Sequential(
        #     nn.BatchNorm1d(crtitic_feature_dim),
        #     nn.ReLU(),
        #     nn.Linear(crtitic_feature_dim, 1),
        # )
        if actor_feature_dim != num_outputs:
            self._actor_mlp = nn.Sequential(
                nn.Linear(actor_feature_dim, 256),
                nn.ReLU(),
                nn.Linear(256, num_outputs),
            )
        else:
            self._actor_mlp = nn.Sequential()
        if crtitic_feature_dim != 1:
            self._critic_mlp = nn.Sequential(
                nn.Linear(crtitic_feature_dim, 1)
            )
        else:
            self._critic_mlp = nn.Sequential()

    def forward(self, input_dict, state, seq_lens):
        obs_transformed = input_dict['obs']
        # if self.simple_actor:
        #     logits = self._actor_head(obs_transformed)
        # else:
        traj = self._actor_head(obs_transformed)
        logits = self._actor_mlp(traj)

        # batch_size = len(input_dict)
        # pretrained_logits = logits.view(batch_size, -1, 3) # B, N, 3 (X,Y,yaw)
        # action = pretrained_logits[:,0,:].view(-1,3)

        # print(f'predicted actions: {action}, shape: {action.shape}')
        # action = standard_normalizer(self.non_kin_rescale, action) # take first action
        # print(f'rescaled actions: {action}, shape: {action.shape}')
        # ones = torch.ones(batch_size,1).to(action.device) # 32,

        # print(ones.device, action.device)
        # output_logits = torch.cat((action, ones * self.log_std_x, ones * self.log_std_y, ones * self.log_std_yaw), dim = -1)
        # assert output_logits.shape[1] == 6, f'{output_logits.shape[1]}'
        value = self._critic_mlp(self._critic_head(obs_transformed))
        self._value = value.view(-1)

        # return output_logits, state
        return logits, state
    def value_function(self):
        return self._value

def freeze(model):
    for  name, param in model.named_parameters():
        param.requires_grad = False
def freeze_except_last_fc(model):
    for  name, param in model.named_parameters():
        if 'model.fc' not in name: 
                param.requires_grad = False
        else:
                param.requires_grad = True
def freeze_except_last_conv(model):
    for  name, param in model.named_parameters():
        if 'model.layer4.2' in name or 'model.fc' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

def load__model_except_last_fc(model):
    weights_subset = {}
    for key, value in model.state_dict().items():
        if key == 'model.fc.weight':
            continue
        elif key == 'model.fc.bias':
            continue
        else:
            weights_subset[key] = value
    return weights_subset

def load_num_action_model(model, num_output):
    weights_subset = {}
    for key, value in model.state_dict().items():
        if key == 'model.fc.weight':
            weights_subset[key] = value[:num_output,:]
        elif key == 'model.fc.bias':
            weights_subset[key] = value[:num_output]
        else:
            weights_subset[key] = value
    return weights_subset
class TorchRasterNet3(TorchModelV2, nn.Module): #TODO: recreate rasternet for PPO, check again
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # print('torchrasternet init device', self.device)

        self.log_std_x = -5
        self.log_std_y = -5
        self.log_std_yaw = -5
        # cfg = model_config["custom_model_config"]['cfg'] # TODO: Pass necessary params, not cfg
        n_channels = model_config["custom_model_config"]['n_channels'] 
        critic_net = model_config["custom_model_config"]['critic_net']
        actor_net = model_config["custom_model_config"]['actor_net']
        # self.non_kin_rescale = model_config["custom_model_config"]['non_kin_rescale'] 
        future_num_frames = model_config["custom_model_config"]['future_num_frames']
        freeze_actor = model_config["custom_model_config"]['freeze_actor'] 
        shared_feature_extractor =  model_config["custom_model_config"]['shared_feature_extractor'] 
        # d_model = model_config["custom_model_config"]['critic_feature_dim'] 

        # actor_feature_dim = 3 * future_num_frames
        # actor_feature_dim = num_outputs 
        # crtitic_feature_dim = 1
        actor_feature_dim = 128 
        crtitic_feature_dim = 128
        if actor_net == 'resnet50':
            self.pretrained_policy = RasterizedPlanningModelFeature(
                model_arch="resnet50",
                num_input_channels=5, # pretrained model input: 5x224x224
                num_targets=3 * 12, 
                weights_scaling=[1., 1., 1.],
                criterion=nn.MSELoss(reduction="none"),)
            model_path = SRC_PATH + "src/model/planning_model_20201208.pt"
            self.pretrained_policy.load_state_dict(torch.load(model_path).state_dict())
            logging.debug('loaded pretrained model')

        self._actor_head =RasterizedPlanningModelFeature(
            model_arch=actor_net,
            num_input_channels=n_channels,
            num_targets= actor_feature_dim , # or same as pretrained_policy to load
            weights_scaling=[1., 1., 1.],
            criterion=nn.MSELoss(reduction="none"),)

        if actor_net == 'resnet50':
            # self._actor_head.load_state_dict(load_num_action_model(self.pretrained_policy, actor_feature_dim))
            self._actor_head.load_state_dict(load__model_except_last_fc(self.pretrained_policy, actor_feature_dim))
            # freeze_except_last_conv(self._actor_head)
            freeze_except_last_fc(self._actor_head)
            logging.debug('loaded actornet model, freeze except last fc')

        if shared_feature_extractor:
            self._critic_head = self._actor_head
        else:
            self._critic_head =RasterizedPlanningModelFeature(
                model_arch=critic_net,
                num_input_channels=n_channels,
                num_targets=crtitic_feature_dim,  # feature dim of critic
                weights_scaling=[1., 1., 1.],
                criterion=nn.MSELoss(reduction="none"),)

        if critic_net == 'resnet50':
            # self._critic_head.load_state_dict(load_num_action_model(self.pretrained_policy, crtitic_feature_dim))
            self._critic_head.load_state_dict(load__model_except_last_fc(self.pretrained_policy, crtitic_feature_dim))
            # freeze_except_last_conv(self._critic_head)
            freeze_except_last_fc(self._critic_head)
            logging.debug('loaded critic model, freeze except last fc')

        if freeze_actor:
            # self._actor_head.load_state_dict(torch.load(model_path).state_dict()) # NOTE: somehow actor_head in cuda device
            for param in self._actor_head.parameters():
                param.requires_grad = False

        # device = next(self.pretrained_policy.parameters()).device
        # logging.debug('>>>>>>>>>>>>>>pretrained device:', device)
        device = next(self._actor_head.parameters()).device
        logging.debug(f'>>>>>>>>>>>>>>actor_head device: {device}')
        device = next(self._critic_head.parameters()).device
        logging.debug(f'>>>>>>>>>>>>>>critic_head device: {device}')
    
        if actor_feature_dim != num_outputs:
            self._actor_mlp = nn.Sequential(
                nn.Linear(actor_feature_dim, 256),
                nn.ReLU(),
                nn.Linear(256, num_outputs),
            )
        else:
            self._actor_mlp = nn.Sequential()
        if crtitic_feature_dim != 1:
            self._critic_mlp = nn.Sequential(
                # nn.BatchNorm1d(crtitic_feature_dim),
                # nn.ReLU(),
                nn.Linear(crtitic_feature_dim, 1)
            )
        else:
            self._critic_mlp = nn.Sequential()

    def forward(self, input_dict, state, seq_lens):
        obs_transformed = input_dict['obs']
        # if self.simple_actor:
        #     logits = self._actor_head(obs_transformed)
        # else:
        traj = self._actor_head(obs_transformed)

        logits = self._actor_mlp(traj)

        # batch_size = len(input_dict)
        # pretrained_logits = logits.view(batch_size, -1, 3) # B, N, 3 (X,Y,yaw)
        # action = pretrained_logits[:,0,:].view(-1,3)

        # logging.debug(f'predicted actions: {action}, shape: {action.shape}')
        # action = standard_normalizer(self.non_kin_rescale, action) # take first action
        # logging.debug(f'rescaled actions: {action}, shape: {action.shape}')
        # ones = torch.ones(batch_size,1).to(action.device) # 32,

        # logging.debug(ones.device, action.device)
        # logits = torch.cat((action, ones * self.log_std_x, ones * self.log_std_y, ones * self.log_std_yaw), dim = -1)
        # assert logits.shape[1] == 6, f'{logits.shape[1]}'
        value = self._critic_mlp(self._critic_head(obs_transformed))
        self._value = value.view(-1)

        # return output_logits, state
        return logits, state
    def value_function(self):
        return self._value

class TorchRasterNetSAC(SACTorchModel):
    """
    RasterNet Model agent
    """
    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: Optional[int],
        model_config: ModelConfigDict,
        name: str,
        policy_model_config: ModelConfigDict = None,
        q_model_config: ModelConfigDict = None,
        twin_q: bool = True,
        initial_alpha: float = 1.0,
        target_entropy: Optional[float] = None,
    ):

#         model_path = "/home/pronton/rl/l5kit/examples/urban_driver/OL_HS.pt"
        # self._critic_head.load_state_dict(torch.load(model_path).state_dict(), strict = False)
        # self._actor_head = torch.load(model_path).to(self.device)
        # self._actor_head.load_state_dict(torch.load(model_path)).to(self.device)

        # self._critic_head.load_state_dict()
        # self.outputs = nn.ModuleList()
        # for i in range(action_space.shape[0]):
        #     self.outputs.append(nn.Linear(num_outputs, 1)) # 6x
        super(TorchRasterNetSAC, self).__init__(
            obs_space, action_space, num_outputs, model_config, name, policy_model_config, q_model_config, twin_q, initial_alpha, target_entropy
        )
        
    def build_policy_model(self, obs_space, num_outputs, policy_model_config, name):
        """Builds the policy model used by this SAC.

        Override this method in a sub-class of SACTFModel to implement your
        own policy net. Alternatively, simply set `custom_model` within the
        top level SAC `policy_model` config key to make this default
        implementation of `build_policy_model` use your custom policy network.

        Returns:
            TorchModelV2: The TorchModelV2 policy sub-model.
        """
        logging.debug(f'policy_model: {obs_space}. {self.action_space}. {num_outputs}')
        model = ModelCatalog.get_model_v2(
            obs_space,
            self.action_space,
            num_outputs,
            policy_model_config,
            framework="torch",
            name=name,
        )
        return model 

    def build_q_model(self, obs_space, action_space, num_outputs, q_model_config, name): # TODO: take resnet50  224x224x5, how to concate input with action to compute Q?
        """Builds one of the (twin) Q-nets used by this SAC.

        Override this method in a sub-class of SACTFModel to implement your
        own Q-nets. Alternatively, simply set `custom_model` within the
        top level SAC `q_model_config` config key to make this default implementation
        of `build_q_model` use your custom Q-nets.

        Returns:
            TorchModelV2: The TorchModelV2 Q-net sub-model.
        """
        self.concat_obs_and_actions = False
        if self.discrete:
            input_space = obs_space
        else:
            orig_space = getattr(obs_space, "original_space", obs_space)
            if isinstance(orig_space, Box) and len(orig_space.shape) == 1:
                input_space = Box(
                    float("-inf"),
                    float("inf"),
                    shape=(orig_space.shape[0] + action_space.shape[0],),
                )
                self.concat_obs_and_actions = True
            else:
                input_space = gym.spaces.Tuple([orig_space, action_space])

        logging.debug(f'q_model: {input_space}. {obs_space}. {action_space}. {num_outputs}')
        model = ModelCatalog.get_model_v2(
            input_space,
            action_space,
            num_outputs,
            q_model_config,
            framework="torch",
            name=name,
        )
        return model
class TorchRasterQNet_test(TorchModelV2, nn.Module):
    """
    RasterNet Model agent
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        # raise ValueError(num_outputs)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        cfg = model_config["custom_model_config"]['cfg'] # TODO: Pass necessary params, not cfg

        # self.outputs = None
        # self.log_std_x = np.log(5.373758673667908/10)
        # self.log_std_y = np.log(0.08619927801191807/10)
        # self.log_std_yaw = np.log(0.04215553868561983 / 10)
        # self.outputs = None

        self.log_std_x = -10
        self.log_std_y = -10
        self.log_std_yaw = -10

        # self._num_predicted_frames = 1

        # self._actor_head =RasterizedPlanningModelFeature(
        #     model_arch="resnet50",
        #     num_input_channels=5,
        #     num_targets=3 * cfg["model_params"]["future_num_frames"],  # X, Y, Yaw * number of future states
        #     weights_scaling=[1., 1., 1.],
        #     criterion=nn.MSELoss(reduction="none"),)

        self._state_net =RasterizedPlanningModelFeature(
            model_arch="resnet50",
            num_input_channels=5,
            num_targets=256,  # feature dim of critic
            weights_scaling=[1., 1., 1.],
            criterion=nn.MSELoss(reduction="none"),)
        # TODO: Build q model (copy reward model or resnet50 or any other CNN model)
        self._action_net = nn.Sequential(
            nn.Linear(action_space.shape[0], 256),
            nn.ReLU()
        )
        assert action_space.shape[0] == num_outputs, f'{action_space.shape[0]} != {num_outputs}'
        self.q_value = nn.Linear(512 + action_space.shape[0], 1)
        # for param in self._actor_head.parameters():
        #     param.requires_grad = False

    def forward(self, input_dict, state, seq_lens):
        raise ValueError(input_dict, input_dict['obs'].shape, input_dict['obs'])
        obs_transformed = input_dict['obs']
        # logging.debug('obs forward:'+ str(obs_transformed))
        logits = self._actor_head(obs_transformed)
        # logging.debug('predict traj' + str(logits))
        STEP_TIME = 1
# <<<<<<< HEAD
#         pred_x = logits['positions'][:,0, 0].view(-1,1).to(self.device) * STEP_TIME# take the first action 
#         pred_y = logits['positions'][:,0, 1].view(-1,1).to(self.device) * STEP_TIME# take the first action
#         pred_yaw = logits['yaws'][:,0,:].view(-1,1).to(self.device) * STEP_TIME# take the first action
        
#         std = torch.ones_like(pred_x).to(self.device) *-10 # 32,
# #         raise ValueError(self.device, pred_x.device, std.device)
#         # assert ones.shape[1] == 1, f'{ones.shape[1]}'
#         # output_logits_mean = torch.cat((pred_x, pred_y, pred_yaw), dim = -1)
#         output_logits = torch.cat((pred_x,pred_y, pred_yaw, std, std, std), dim = -1).to(self.device)
# =======
        
        batch_size = len(input_dict)

        predicted = logits.view(batch_size, -1, 3) # B, N, 3 (X,Y,yaw)
        # print(f'predicted {predicted}, shape: {predicted.shape}')
        pred_x = predicted[:, 0, 0].view(-1,1) * STEP_TIME# take the first action 
        pred_y = predicted[:, 0, 1].view(-1,1) * STEP_TIME# take the first action
        pred_yaw = predicted[:, 0, 2].view(-1,1)* STEP_TIME# take the first action

        # print(f'pred_x {pred_x}, shape: {pred_x.shape}')
        # print(f'pred_y {pred_y}, shape: {pred_y.shape}')
        # print(f'pred_yaw {pred_yaw}, shape: {pred_yaw.shape}')
        # pred_x = logits['positions'][:,0, 0].view(-1,1) * STEP_TIME# take the first action 
        # pred_y = logits['positions'][:,0, 1].view(-1,1) * STEP_TIME# take the first action
        # pred_yaw = logits['yaws'][:,0,:].view(-1,1) * STEP_TIME# take the first action
        ones = torch.ones_like(pred_x) # 32,
        # assert ones.shape[1] == 1, f'{ones.shape[1]}'
        # output_logits_mean = torch.cat((pred_x, pred_y, pred_yaw), dim = -1)
        output_logits = torch.cat((pred_x, pred_y, pred_yaw, ones * self.log_std_x, ones * self.log_std_y, ones * self.log_std_yaw), dim = -1)
# >>>>>>> a67daa30820ac7621232e8d1a33832b30093f810
        # print('pretrained action', output_logits)
        assert output_logits.shape[1] == 6, f'{output_logits.shape[1]}'
        # self.outputs = output_logits

        # dist = torch.distributions.Normal(output_logits_mean, torch.ones_like(output_logits_mean)*0.0005)
        # print('-----------------------------sample', dist.rsample())

        feature_value = self._critic_head(obs_transformed)
        value = self._critic_FF(feature_value)
        self._value = value.view(-1)

        return output_logits, state
    def value_function(self):
        return self._value

class TorchGNCNN(TorchModelV2, nn.Module):
    """
    Simple Convolution agent that calculates the required linear output layer
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # raise ValueError(obs_space.shape)
        self._num_objects = obs_space.shape[2] # num_of_channels of input, size x size x channels
        assert self._num_objects < 15, f'wrong shape: {obs_space.shape}'
        self._num_actions = num_outputs
        self._feature_dim = model_config["custom_model_config"]['feature_dim']
        assert obs_space.shape[0] > self._num_objects, str(obs_space.shape) + '!=  (size, size, # channels)'

        self.network = nn.Sequential(
            nn.Conv2d(7, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.GroupNorm(4, 64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 32, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.GroupNorm(2, 32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=1568, out_features=self._feature_dim),
        )

        self._actor_head = nn.Sequential(
            nn.Linear(self._feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self._num_actions),
        )

        self._critic_head = nn.Sequential(
            nn.Linear(self._feature_dim, 1),
        )

# <<<<<<< HEAD
#     def forward(self, input_dict, state, seq_lens):
#         obs_transformed = input_dict['obs'].permute(0, 3, 1, 2) # input_dict['obs'].shape = [B, size, size, # channels] => obs_transformed.shape = [B, # channels, size, size]
#         assert input_dict['obs'].shape[3] < input_dict['obs'].shape[2] , \
#             str(input_dict['obs'].shape) + ' != (_ ,size,size,n_channels),  obs_transformed: ' + str(obs_transformed.shape)
#         network_output = self.network(obs_transformed) #  [B, # channels, size, size]
# =======
    def forward(self, input_dict, state, seq_lens): # from dataloader? get 32, 112, 112, 7
        # obs_transformed = input_dict['obs'].permute(0, 3, 1, 2) # 32 x 112 x 112 x 7 [B, size, size, channels]
        obs_transformed = input_dict['obs'].permute(0, 3, 1, 2) # [B, C, W, H] -> [B, W, H, C]
        # print('forward', obs_transformed.shape)
        network_output = self.network(obs_transformed)
# >>>>>>> 82fd9a0ee83cd280c7d1bcc9c254b002f5a103b1
        value = self._critic_head(network_output)
        self._value = value.reshape(-1)
        logits = self._actor_head(network_output)
        return logits, state

    def value_function(self):
        return self._value
class TorchGNCNN_separated(TorchModelV2, nn.Module):
    """
    Simple Convolution agent that calculates the required linear output layer
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # raise ValueError(obs_space.shape)
        self._num_objects = obs_space.shape[2] # num_of_channels of input, size x size x channels
        self._num_actions = num_outputs
        self._feature_dim = model_config["custom_model_config"]['feature_dim']

        self._actor_head = nn.Sequential(
            nn.Conv2d(self._num_objects, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.GroupNorm(4, 64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 32, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.GroupNorm(2, 32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=1568, out_features=self._feature_dim),
            nn.Linear(self._feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self._num_actions),
        )

        self._critic_head = nn.Sequential(
            nn.Conv2d(self._num_objects, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.GroupNorm(4, 64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 32, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.GroupNorm(2, 32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=1568, out_features=self._feature_dim),
            nn.Linear(self._feature_dim, 1),
        )

    def forward(self, input_dict, state, seq_lens):
        obs_transformed = input_dict['obs'].permute(0, 3, 1, 2) # input_dict['obs'].shape = [B, size, size, # channels] => obs_transformed.shape = [B, # channels, size, size]
        assert input_dict['obs'].shape[3] < input_dict['obs'].shape[2] , \
            str(input_dict['obs'].shape) + ' != (_ ,size,size,n_channels),  obs_transformed: ' + str(obs_transformed.shape)
        # network_output = self.network(obs_transformed)
        value = self._critic_head(obs_transformed)
        self._value = value.reshape(-1)
        logits = self._actor_head(obs_transformed)
        return logits, state

    def value_function(self):
        return self._value

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.layers.children():
            nn.init.zeros_(layer.bias)
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class TorchAttentionModel2(TorchModelV2, nn.Module):
    """
    Attention Model agent
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        # raise ValueError(num_outputs)

        cfg = model_config["custom_model_config"]['cfg']
        d_model = 256
        self._num_actions = num_outputs
        self._actor_head = MLP(d_model, d_model * 4, output_dim= num_outputs, num_layers=3)
        self._critic_head = MLP(d_model, d_model * 4, output_dim= 1, num_layers=1)

        self.feature_extractor= CustomVectorizedModel(
            history_num_frames_ego=cfg["model_params"]["history_num_frames_ego"],
            history_num_frames_agents=cfg["model_params"]["history_num_frames_agents"],
            global_head_dropout=cfg["model_params"]["global_head_dropout"],
            disable_other_agents=cfg["model_params"]["disable_other_agents"],
            disable_map=cfg["model_params"]["disable_map"],
            disable_lane_boundaries=cfg["model_params"]["disable_lane_boundaries"])
        

    def forward(self, input_dict, state, seq_lens):
        obs_transformed = input_dict['obs']
        out, attns = self.feature_extractor(obs_transformed)
        logits = self._actor_head(out).view(-1, self._num_actions)
        self._value = self._critic_head(out).view(-1)
        return logits, state

    def value_function(self):
        return self._value

class TorchAttentionModel(TorchModelV2, nn.Module):
    """
    Attention Model agent
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        # raise ValueError(num_outputs)

        cfg = model_config["custom_model_config"]['cfg']
        weights_scaling = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        self._num_predicted_frames = cfg["model_params"]["future_num_frames"]
        self._num_predicted_params = len(weights_scaling)
        weights_scaling_critic = [1.0]
        _num_predicted_params_critic = len(weights_scaling_critic)

        self._actor_head = VectorizedModel(
            history_num_frames_ego=cfg["model_params"]["history_num_frames_ego"],
            history_num_frames_agents=cfg["model_params"]["history_num_frames_agents"],
            num_targets=self._num_predicted_params * self._num_predicted_frames, # N (X,Y,Yaw) 72
            weights_scaling=weights_scaling, # 6
            criterion=nn.L1Loss(reduction="none"),
            global_head_dropout=cfg["model_params"]["global_head_dropout"],
            disable_other_agents=cfg["model_params"]["disable_other_agents"],
            disable_map=cfg["model_params"]["disable_map"],
            disable_lane_boundaries=cfg["model_params"]["disable_lane_boundaries"])

        self._critic_head = VectorizedModel(
            history_num_frames_ego=cfg["model_params"]["history_num_frames_ego"],
            history_num_frames_agents=cfg["model_params"]["history_num_frames_agents"],
            num_targets=_num_predicted_params_critic, # just 1 (X,Y,Yaw)
            weights_scaling=weights_scaling_critic,
            criterion=nn.L1Loss(reduction="none"),
            global_head_dropout=cfg["model_params"]["global_head_dropout"],
            disable_other_agents=cfg["model_params"]["disable_other_agents"],
            disable_map=cfg["model_params"]["disable_map"],
            disable_lane_boundaries=cfg["model_params"]["disable_lane_boundaries"])

    def forward(self, input_dict, state, seq_lens):
        # obs_transformed = input_dict['obs'].permute(0, 3, 1, 2) # 32 x 112 x 112 x 7 [b, size, size, channels]
        # raise ValueError(input_dict)
        obs_transformed = input_dict['obs']
        # raise ValueError(input_dict['obs'])
        # network_output = self.network(obs_transformed)
        logits = self._actor_head(obs_transformed)
        # raise ValueError(str(logits['positions'].shape))
        # logits = torch.cat((logits['positions'], logits['yaws']),axis=-1)
        logits = logits.view(-1, int(self._num_predicted_frames * self._num_predicted_params))
        # raise ValueError(logits.shape)
        value = self._critic_head(obs_transformed)
        self._value = value.view(-1)
        # raise ValueError(logits.shape)
        # raise ValueError('positions: ' + str(logits['positions'].shape) + 'yaw:' + str(logits['yaws'].shape))
        return logits, state

    def value_function(self):
        return self._value

class TorchAttentionModel3(TorchModelV2, nn.Module):
    """
    Attention Model agent
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cfg = model_config["custom_model_config"]['cfg']
        # print('action space:', action_space)
        # print('num output:', num_outputs)
        weights_scaling = [1.0, 1.0, 1.0]
        # self.outputs = None
        # self.log_std_x = np.log(5.373758673667908/10)
        # self.log_std_y = np.log(0.08619927801191807/10)
        # self.log_std_yaw = np.log(0.04215553868561983 / 10)
        # self.outputs = None

        self.log_std_x = -5
        self.log_std_y = -5
        self.log_std_yaw = -5

        self._num_predicted_frames = cfg["model_params"]["future_num_frames"]
        # self._num_predicted_frames = 1
        self._num_predicted_params = len(weights_scaling) #6
        weights_scaling_critic = [1.0]
        _num_predicted_params_critic = len(weights_scaling_critic)

        self._actor_head = VectorizedModel(
            history_num_frames_ego=cfg["model_params"]["history_num_frames_ego"],
            history_num_frames_agents=cfg["model_params"]["history_num_frames_agents"],
            num_targets=self._num_predicted_params * self._num_predicted_frames, # N (X,Y,Yaw) 36
            weights_scaling=weights_scaling, # 3
            criterion=nn.L1Loss(reduction="none"),
            global_head_dropout=cfg["model_params"]["global_head_dropout"],
            disable_other_agents=cfg["model_params"]["disable_other_agents"],
            disable_map=cfg["model_params"]["disable_map"],
            disable_lane_boundaries=cfg["model_params"]["disable_lane_boundaries"])

        self._critic_head = CustomVectorizedModel(
            history_num_frames_ego=cfg["model_params"]["history_num_frames_ego"],
            history_num_frames_agents=cfg["model_params"]["history_num_frames_agents"],
            global_head_dropout=cfg["model_params"]["global_head_dropout"],
            disable_other_agents=cfg["model_params"]["disable_other_agents"],
            disable_map=cfg["model_params"]["disable_map"],
            disable_lane_boundaries=cfg["model_params"]["disable_lane_boundaries"])

        d_model = 256
        self._critic_FF = MLP(d_model, d_model * 4, output_dim= 1, num_layers=1)
        model_path = "/home/pronton/rlhf-car/src/model/OL_HS.pt"
#         model_path = "/home/pronton/rl/l5kit/examples/urban_driver/OL_HS.pt"
        # self._critic_head.load_state_dict(torch.load(model_path).state_dict(), strict = False)
        self._actor_head.load_state_dict(torch.load(model_path).state_dict())
        for param in self._actor_head.parameters():
            param.requires_grad = False

        # self._critic_head.load_state_dict()
        # self.outputs = nn.ModuleList()
        # for i in range(action_space.shape[0]):
        #     self.outputs.append(nn.Linear(num_outputs, 1)) # 6x
        
    def forward(self, input_dict, state, seq_lens):
        obs_transformed = input_dict['obs']
        # logging.debug('obs forward:'+ str(obs_transformed))
        logits = self._actor_head(obs_transformed)
        logging.debug('predict traj' + str(logits))
        STEP_TIME = 1
# <<<<<<< HEAD
#         pred_x = logits['positions'][:,0, 0].view(-1,1).to(self.device) * STEP_TIME# take the first action 
#         pred_y = logits['positions'][:,0, 1].view(-1,1).to(self.device) * STEP_TIME# take the first action
#         pred_yaw = logits['yaws'][:,0,:].view(-1,1).to(self.device) * STEP_TIME# take the first action
        
#         std = torch.ones_like(pred_x).to(self.device) *-10 # 32,
# #         raise ValueError(self.device, pred_x.device, std.device)
#         # assert ones.shape[1] == 1, f'{ones.shape[1]}'
#         # output_logits_mean = torch.cat((pred_x, pred_y, pred_yaw), dim = -1)
#         output_logits = torch.cat((pred_x,pred_y, pred_yaw, std, std, std), dim = -1).to(self.device)
# =======
        
        pred_x = logits['positions'][:,0, 0].view(-1,1) * STEP_TIME# take the first action 
        pred_y = logits['positions'][:,0, 1].view(-1,1) * STEP_TIME# take the first action
        pred_yaw = logits['yaws'][:,0,:].view(-1,1) * STEP_TIME# take the first action
        ones = torch.ones_like(pred_x) # 32,
        # assert ones.shape[1] == 1, f'{ones.shape[1]}'
        # output_logits_mean = torch.cat((pred_x, pred_y, pred_yaw), dim = -1)
        output_logits = torch.cat((pred_x,pred_y, pred_yaw, ones * self.log_std_x, ones * self.log_std_y, ones * self.log_std_yaw), dim = -1)
# >>>>>>> a67daa30820ac7621232e8d1a33832b30093f810
        # print('pretrained action', output_logits[:,:3])
        assert output_logits.shape[1] == 6, f'{output_logits.shape[1]}'
        # self.outputs = output_logits

        # dist = torch.distributions.Normal(output_logits_mean, torch.ones_like(output_logits_mean)*0.0005)
        # print('-----------------------------sample', dist.rsample())

        feature_value, attns = self._critic_head(obs_transformed)
        value = self._critic_FF(feature_value)
        self._value = value.view(-1)

        return output_logits, state
    def value_function(self):
        return self._value
class AssembleModel(nn.Module):
    def __init__(self, cfg, num_outputs):
        super(AssembleModel, self).__init__()
        self.attention = CustomVectorizedModel(
            history_num_frames_ego=cfg["model_params"]["history_num_frames_ego"],
            history_num_frames_agents=cfg["model_params"]["history_num_frames_agents"],
            global_head_dropout=cfg["model_params"]["global_head_dropout"],
            disable_other_agents=cfg["model_params"]["disable_other_agents"],
            disable_map=cfg["model_params"]["disable_map"],
            disable_lane_boundaries=cfg["model_params"]["disable_lane_boundaries"])

        d_model = 256
        if num_outputs > 1: # actor
            self.fc = MLP(d_model, d_model , output_dim = num_outputs, num_layers=1)
        else:
            self.fc = MLP(d_model, d_model , output_dim= 1, num_layers=1)

    def forward(self, x):
        return self.fc(self.attention(x))

class TorchAttentionModel4SAC(SACTorchModel):
    """
    Attention Model agent
    """

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: Optional[int],
        model_config: ModelConfigDict,
        name: str,
        policy_model_config: ModelConfigDict = None,
        q_model_config: ModelConfigDict = None,
        twin_q: bool = True,
        initial_alpha: float = 1.0,
        target_entropy: Optional[float] = None,
    ):
    # def __init__(self, obs_space, action_space, num_outputs, model_config, name):
    #     super().__init__(obs_space, action_space, num_outputs, model_config, name)
    #     nn.Module.__init__(self)

        self.cfg = model_config["custom_model_config"]['cfg']
        weights_scaling = [1.0, 1.0, 1.0]

        self.log_std_x = -5
        self.log_std_y = -5
        self.log_std_yaw = -5

        # self._num_predicted_frames = self.cfg["model_params"]["future_num_frames"]
        # self._num_predicted_frames = 1
        self._num_predicted_params = len(weights_scaling) #6
        # weights_scaling_critic = [1.0]
        # actor_feature_dim = 128 
        # crtitic_feature_dim = 128

        # freeze_actor = model_config["custom_model_config"]['freeze_actor'] 
        # shared_feature_extractor =  model_config["custom_model_config"]['shared_feature_extractor'] 

        # self.pretrained_policy = VectorizedModel(
        #     history_num_frames_ego=self.cfg["model_params"]["history_num_frames_ego"],
        #     history_num_frames_agents=self.cfg["model_params"]["history_num_frames_agents"],
        #     num_targets=self._num_predicted_params * self._num_predicted_frames, # N (X,Y,Yaw) 36
        #     weights_scaling=weights_scaling, # 3
        #     criterion=nn.L1Loss(reduction="none"),
        #     global_head_dropout=self.cfg["model_params"]["global_head_dropout"],
        #     disable_other_agents=self.cfg["model_params"]["disable_other_agents"],
        #     disable_map=self.cfg["model_params"]["disable_map"],
        #     disable_lane_boundaries=self.cfg["model_params"]["disable_lane_boundaries"])
        
        # self._actor_head = CustomVectorizedModel(
        #     history_num_frames_ego=self.cfg["model_params"]["history_num_frames_ego"],
        #     history_num_frames_agents=self.cfg["model_params"]["history_num_frames_agents"],
        #     global_head_dropout=self.cfg["model_params"]["global_head_dropout"],
        #     disable_other_agents=self.cfg["model_params"]["disable_other_agents"],
        #     disable_map=self.cfg["model_params"]["disable_map"],
        #     disable_lane_boundaries=self.cfg["model_params"]["disable_lane_boundaries"])

        # self._actor_head = AssembleModel(self.cfg, num_outputs)
        # self._critic_head = AssembleModel(self.cfg, 1)
        # if shared_feature_extractor:
        #     self._critic_head = self._actor_head
        # else:
        #     self._critic_head = CustomVectorizedModel(
        #         history_num_frames_ego=self.cfg["model_params"]["history_num_frames_ego"],
        #         history_num_frames_agents=self.cfg["model_params"]["history_num_frames_agents"],
        #         global_head_dropout=self.cfg["model_params"]["global_head_dropout"],
        #         disable_other_agents=self.cfg["model_params"]["disable_other_agents"],
        #         disable_map=self.cfg["model_params"]["disable_map"],
        #         disable_lane_boundaries=self.cfg["model_params"]["disable_lane_boundaries"])

        
        d_model = 256

        # self._actor_mlp = MLP(d_model, d_model * 4, num_outputs, num_layers=3) # similar to pretrained
        # self._actor_mlp = MLP(d_model, d_model , output_dim = num_outputs, num_layers=1)
        # self._critic_mlp = MLP(d_model, d_model , output_dim= 1, num_layers=1)

#         model_path = "/home/pronton/rl/l5kit/examples/urban_driver/OL_HS.pt"
        # self._critic_head.load_state_dict(torch.load(model_path).state_dict(), strict = False)
        # if freeze_actor:
        #     model_path = "/home/pronton/rlhf-car/src/model/OL_HS.pt"
        #     self._actor_head.load_state_dict(torch.load(model_path).state_dict())
        #     for param in self._actor_head.parameters():
        #         param.requires_grad = False

        # self._critic_head.load_state_dict()
        # self.outputs = nn.ModuleList()
        # for i in range(action_space.shape[0]):
        #     self.outputs.append(nn.Linear(num_outputs, 1)) # 6x
        super(TorchAttentionModel4SAC, self).__init__(
            obs_space, action_space, num_outputs, model_config, name, policy_model_config, q_model_config, twin_q, initial_alpha, target_entropy
        )
        self.policy_net = TorchVectorPolicyNet(obs_space, action_space, num_outputs, model_config, name)
        self.q_net = TorchVectorQNet(obs_space, action_space, num_outputs, model_config, name)

    def build_policy_model(self, obs_space, num_outputs, policy_model_config, name):
        """Builds the policy model used by this SAC.

        Override this method in a sub-class of SACTFModel to implement your
        own policy net. Alternatively, simply set `custom_model` within the
        top level SAC `policy_model` config key to make this default
        implementation of `build_policy_model` use your custom policy network.

        Returns:
            TorchModelV2: The TorchModelV2 policy sub-model.
        """
        # self._actor_head = CustomVectorizedModel(
        #     history_num_frames_ego=cfg["model_params"]["history_num_frames_ego"],
        #     history_num_frames_agents=cfg["model_params"]["history_num_frames_agents"],
        #     global_head_dropout=cfg["model_params"]["global_head_dropout"],
        #     disable_other_agents=cfg["model_params"]["disable_other_agents"],
        #     disable_map=cfg["model_params"]["disable_map"],
        #     disable_lane_boundaries=cfg["model_params"]["disable_lane_boundaries"])
        # d_model = 256
        # self._actor_mlp = MLP(d_model, d_model , output_dim = num_outputs, num_layers=1)
        
        return self.policy_net

        
    def forward(self, input_dict, state, seq_lens):
        obs_transformed = input_dict['obs']
        # logging.debug('obs forward:'+ str(obs_transformed))
        actor_feature_value, attns = self._actor_head(obs_transformed)
        logits = self._actor_mlp(actor_feature_value)

        critic_feature_value, attns = self._critic_head(obs_transformed)
        value = self._critic_mlp(critic_feature_value)
        self._value = value.view(-1)

        return logits, state
    def value_function(self):
        return self._value

class TorchVectorPolicyNet(TorchModelV2, nn.Module):
    """
    RasterNet Model agent
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        cfg = model_config["custom_model_config"]['cfg'] # TODO: Pass necessary params, not cfg

        self.attention = CustomVectorizedModel(
            history_num_frames_ego=cfg["model_params"]["history_num_frames_ego"],
            history_num_frames_agents=cfg["model_params"]["history_num_frames_agents"],
            global_head_dropout=cfg["model_params"]["global_head_dropout"],
            disable_other_agents=cfg["model_params"]["disable_other_agents"],
            disable_map=cfg["model_params"]["disable_map"],
            disable_lane_boundaries=cfg["model_params"]["disable_lane_boundaries"])

        d_model = 256
        # assert num_outputs  == 1, f'wrong {num_outputs}'
        # if num_outputs > 1: # actor
        #     self.fc = MLP(d_model, d_model , output_dim = num_outputs, num_layers=1)
        # else:
        logging.debug(f'num_outputs in policy:{num_outputs}')
        self.fc = MLP(d_model, d_model , output_dim= num_outputs, num_layers=1)

    def forward(self, input_dict, state, seq_lens):
        # logging.debug(f'policy input: {input_dict["obs"]}')
        # logging.debug(f'policy input: {input_dict["action"]}')
        obs = input_dict['obs']
        # logging.debug(f'policy input types: {[v.dtype for v in input_dict["obs"].values()]}')
        #TODO: Why all type is torch.float32??? How to convert to bool, int as in l5env2, Why PPO can run?
        features, attns = self.attention(obs)
        return self.fc(features), state

from ray.rllib.utils.spaces.space_utils import flatten_space
class TorchVectorQNet(TorchModelV2, nn.Module):
    """
    RasterNet Model agent
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # self.original_space = (
        #     obs_space.original_space
        #     if hasattr(obs_space, "original_space")
        #     else obs_space
        # )

        # self.processed_obs_space = (
        #     self.original_space
        #     if model_config.get("_disable_preprocessor_api")
        #     else obs_space
        # )
        # self.flattened_input_space = flatten_space(self.original_space)
        # logging.debug(f'flattened_input_space: {self.flattened_input_space}')
        # for i, component in enumerate(self.flattened_input_space):
        #     print(component.shape)
        # logging.debug(f'obs_space: {obs_space[0]}, action_space: {obs_space[1]}')
        # logging.debug(f'action_space: {action_space}')

        cfg = model_config["custom_model_config"]['cfg'] # TODO: Pass necessary params, not cfg

        self.attention = CustomVectorizedModel(
            history_num_frames_ego=cfg["model_params"]["history_num_frames_ego"],
            history_num_frames_agents=cfg["model_params"]["history_num_frames_agents"],
            global_head_dropout=cfg["model_params"]["global_head_dropout"],
            disable_other_agents=cfg["model_params"]["disable_other_agents"],
            disable_map=cfg["model_params"]["disable_map"],
            disable_lane_boundaries=cfg["model_params"]["disable_lane_boundaries"])
        
        self.action_dim = np.product(action_space.shape)
        # action_outs = 2 * self.action_dim
        q_outs = 1
        d_model = 256
        self.action_feature = nn.Sequential(
            nn.Linear(in_features=3, out_features=256),
            nn.Tanh(),
            nn.Linear(in_features=256, out_features=256),
            nn.Tanh(),
        )
        self.q_net = MLP(d_model + d_model, d_model, output_dim = q_outs , num_layers=1)


    def forward(self, input_dict, state, seq_lens):
        # TODO: test input shape
        # logging.debug(f"q obs input_dict: {input_dict['obs'][0]}")
        # logging.debug(f"q action input_dict: {input_dict['obs'][1]}")
        obs, action = input_dict['obs']
        state_features, attns = self.attention(obs)
        action_features = self.action_feature(action)
        logging.debug(f"features dim: {state_features.shape}")
        logging.debug(f"action dim: {action_features.shape}")
        return self.q_net(torch.cat((state_features, action_features), dim=1)), state

class TorchRasterQNet(TorchModelV2, nn.Module):
    """
    RasterNet Model agent
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        d_model = 256
        self.state_feature =RasterizedPlanningModelFeature(
            model_arch="simple_cnn",
            num_input_channels= 7,
            num_targets=d_model,  # feature dim of critic
            weights_scaling=[1., 1., 1.],
            criterion=nn.MSELoss(reduction="none"),)
        self.action_dim = np.product(action_space.shape)
        # action_outs = 2 * self.action_dim
        q_outs = 1
        self.action_feature =MLP(self.action_dim, d_model, output_dim = d_model , num_layers=1)
        self.action_feature = nn.Sequential(
            nn.Linear(in_features=3, out_features=256),
            nn.Tanh(),
            nn.Linear(in_features=256, out_features=256),
            nn.Tanh(),
        )
        self.q_net = MLP(2* d_model, 2* d_model, output_dim = q_outs , num_layers=1)


    def forward(self, input_dict, state, seq_lens):
        obs, action = input_dict['obs']
        state_features = self.state_feature(obs)
        action_features = self.action_feature(action)
        logging.debug(f"state features dim: {state_features.shape}")
        logging.debug(f"action features dim: {action_features.shape}")
        return self.q_net(torch.cat((state_features, action_features), dim=1)), state

class TorchRasterQNet2(TorchModelV2, nn.Module):
    # TODO: TEST THIS
    """
    RasterNet Model agent
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        logging.debug(f'{obs_space}. {action_space}. {num_outputs}. ')

        model = ModelCatalog.get_model_v2(
            obs_space,
            action_space,
            num_outputs,
            model_config,
            framework="torch",
            name=name,
        )
        return model


    def forward(self, input_dict, state, seq_lens):
        obs, action = input_dict['obs']
        state_features = self.state_feature(obs)
        action_features = self.action_feature(action)
        logging.debug(f"state features dim: {state_features.shape}")
        logging.debug(f"action features dim: {action_features.shape}")
        return self.q_net(torch.cat((state_features, action_features), dim=1)), state
class TorchRasterPolicyNet(TorchModelV2, nn.Module):
    """
    RasterNet Model agent
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        d_model = 256
        self.state_feature =RasterizedPlanningModelFeature(
            model_arch="simple_cnn",
            num_input_channels= 7,
            num_targets=d_model,  # feature dim of critic
            weights_scaling=[1., 1., 1.],
            criterion=nn.MSELoss(reduction="none"),)

        self.fc = MLP(d_model, d_model , output_dim= num_outputs, num_layers=1)

    def forward(self, input_dict, state, seq_lens):
        # logging.debug(f'policy input: {input_dict["obs"]}')
        # logging.debug(f'policy input: {input_dict["action"]}')
        obs = input_dict['obs']
        # logging.debug(f'policy input types: {[v.dtype for v in input_dict["obs"].values()]}')
        features = self.state_feature(obs)
        return self.fc(features), state

class TorchAttentionModel4(TorchModelV2, nn.Module):
    """
    Attention Model agent
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        cfg = model_config["custom_model_config"]['cfg']
        weights_scaling = [1.0, 1.0, 1.0]

        self.log_std_x = -5
        self.log_std_y = -5
        self.log_std_yaw = -5

        self._num_predicted_frames = cfg["model_params"]["future_num_frames"]
        # self._num_predicted_frames = 1
        self._num_predicted_params = len(weights_scaling) #6
        weights_scaling_critic = [1.0]
        actor_feature_dim = 128 
        crtitic_feature_dim = 128

        freeze_actor = model_config["custom_model_config"]['freeze_actor'] 
        shared_feature_extractor =  model_config["custom_model_config"]['shared_feature_extractor'] 

        self.pretrained_policy = VectorizedModel(
            history_num_frames_ego=cfg["model_params"]["history_num_frames_ego"],
            history_num_frames_agents=cfg["model_params"]["history_num_frames_agents"],
            num_targets=self._num_predicted_params * self._num_predicted_frames, # N (X,Y,Yaw) 36
            weights_scaling=weights_scaling, # 3
            criterion=nn.L1Loss(reduction="none"),
            global_head_dropout=cfg["model_params"]["global_head_dropout"],
            disable_other_agents=cfg["model_params"]["disable_other_agents"],
            disable_map=cfg["model_params"]["disable_map"],
            disable_lane_boundaries=cfg["model_params"]["disable_lane_boundaries"])
        
        self._actor_head = CustomVectorizedModel(
            history_num_frames_ego=cfg["model_params"]["history_num_frames_ego"],
            history_num_frames_agents=cfg["model_params"]["history_num_frames_agents"],
            global_head_dropout=cfg["model_params"]["global_head_dropout"],
            disable_other_agents=cfg["model_params"]["disable_other_agents"],
            disable_map=cfg["model_params"]["disable_map"],
            disable_lane_boundaries=cfg["model_params"]["disable_lane_boundaries"])

        if shared_feature_extractor:
            self._critic_head = self._actor_head
        else:
            self._critic_head = CustomVectorizedModel(
                history_num_frames_ego=cfg["model_params"]["history_num_frames_ego"],
                history_num_frames_agents=cfg["model_params"]["history_num_frames_agents"],
                global_head_dropout=cfg["model_params"]["global_head_dropout"],
                disable_other_agents=cfg["model_params"]["disable_other_agents"],
                disable_map=cfg["model_params"]["disable_map"],
                disable_lane_boundaries=cfg["model_params"]["disable_lane_boundaries"])

        d_model = 256

        # self._actor_mlp = MLP(d_model, d_model * 4, num_outputs, num_layers=3) # similar to pretrained
        self._actor_mlp = MLP(d_model, d_model , output_dim = num_outputs, num_layers=1)
        self._critic_mlp = MLP(d_model, d_model , output_dim= 1, num_layers=1)

#         model_path = "/home/pronton/rl/l5kit/examples/urban_driver/OL_HS.pt"
        # self._critic_head.load_state_dict(torch.load(model_path).state_dict(), strict = False)
        if freeze_actor:
            model_path = "/home/pronton/rlhf-car/src/model/OL_HS.pt"
            self._actor_head.load_state_dict(torch.load(model_path).state_dict())
            for param in self._actor_head.parameters():
                param.requires_grad = False

        # self._critic_head.load_state_dict()
        # self.outputs = nn.ModuleList()
        # for i in range(action_space.shape[0]):
        #     self.outputs.append(nn.Linear(num_outputs, 1)) # 6x
        
    def forward(self, input_dict, state, seq_lens):
        obs_transformed = input_dict['obs']
        # logging.debug('obs forward:'+ str(obs_transformed))
        actor_feature_value, attns = self._actor_head(obs_transformed)
        logits = self._actor_mlp(actor_feature_value)

        critic_feature_value, attns = self._critic_head(obs_transformed)
        value = self._critic_mlp(critic_feature_value)
        self._value = value.view(-1)

        return logits, state
    def value_function(self):
        return self._value
if __name__ == '__main__':

    import ray.rllib.algorithms.ppo as ppo
    from pprint import PrettyPrinter
    import numpy as np
    from l5kit.configs.config import load_config_data
    from l5kit.data.local_data_manager import LocalDataManager
    from l5kit.environment.envs.l5_env2 import GymStepOutput, SimulationConfigGym, L5Env2
    import os
    from ray.rllib.models import ModelCatalog
    # model = TorchGNCNN(np.zeros((112,112,7)), np.array((3,)),3, model_config= {'custom_model_config': {'feature_dim': 128}}, name='')

    # # In L5env
    # batch_data = {'obs': torch.ones((32,7, 112, 112))}
    # print('batch', batch_data['obs'].shape)

    # # After process in L5envWrapper
    # batch_data = {'obs': torch.ones((32, 112, 112, 7))}


    # # obs_transformed = obs_batch.permute(0, 3, 1, 2) # 32 x 112 x 112 x 7 [B, size, size, channels]
    # # print('transformed', obs_transformed.shape)
    # # print(obs_transformed.shape)
    # model(input_dict=batch_data)
    os.environ["L5KIT_DATA_FOLDER"] = "/workspace/datasets"
    env_config_path = '/workspace/source/src/configs/gym_vectorizer_config.yaml'
    dmg = LocalDataManager(None)
    cfg = load_config_data(env_config_path)
    # rollout_sim_cfg = SimulationConfigGym()
    # rollout_sim_cfg.num_simulation_steps = None
    # env_kwargs = {'env_config_path': env_config_path, 'use_kinematic': False, 'sim_cfg': rollout_sim_cfg,  'train': False, 'return_info': True, 'rescale_action': False}
    # print(rollout_env.action_space)
    # model = TorchAttentionModel3(np.zeros((112,112,7)), np.array((3,)),3, model_config= {"custom_model_config": {'cfg':cfg}}, name='')
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    ###################### TRAINING ######################
    ModelCatalog.register_custom_model( "TorchSeparatedAttentionModel", TorchAttentionModel3)
    from ray import tune
    import ray
    train_eps_length = 32
    train_sim_cfg = SimulationConfigGym()
    train_sim_cfg.num_simulation_steps = train_eps_length + 1
    env_kwargs = {'env_config_path': env_config_path, 'use_kinematic': False, 'sim_cfg': train_sim_cfg, 'rescale_action': False}
    tune.register_env("L5-CLE-V2", lambda config: L5Env2(**env_kwargs))
    l5_env2 = L5Env2(**env_kwargs)
    ray.init(num_cpus=5, ignore_reinit_error=True, log_to_driver=False, local_mode=False)
    # algo = ppo.PPO(
    #         env="L5-CLE-V2",
    #         config={
                # 'disable_env_checking':True,
    #             "framework": "torch",
    #             'log_level': 'INFO',
    #             'num_gpu': 0,
    #             'train_batch_size': 1,
    #             'sgd_minibatch_size': 1,
    #             'num_sgd_iter': 1,
    #             'seed': 42,
    #             'batch_mode': 'truncate_episodes',
    #             # "rollout_fragment_length": 32,
    #             "model": {
    #                 "custom_model": "TorchSeparatedAttentionModel",
    #                 # Extra kwargs to be passed to your model's c'tor.
    #                 "custom_model_config": {'cfg':cfg},
    #             },
    #             # "output": "/home/pronton/rl/l5kit/examples/RL/notebooks/logs/l5env2-out", 
    #             # "output_max_file_size": 5000000,
    #             '_disable_preprocessor_api': True,
    #         },
    #     )

    # for i in range(1):
    #     result = algo.train()
    #     print(PrettyPrinter(result))
    import ray
    from ray import air, tune
    import pytz
    import datetime
    hcmTz = pytz.timezone("Asia/Ho_Chi_Minh") 
    date = datetime.datetime.now(hcmTz).strftime("%d-%m-%Y_%H-%M-%S")
    ray_result_logdir = '/workspace/datasets/ray_results/debug' + date

    train_envs = 4
    lr = 3e-3
    from src.customModel.utils import kl_divergence, PretrainedDistribution
    from ray.rllib.evaluation.postprocessing import compute_advantages


    pretrained_policy = VectorizedModel(
        history_num_frames_ego=cfg["model_params"]["history_num_frames_ego"],
        history_num_frames_agents=cfg["model_params"]["history_num_frames_agents"],
        num_targets=3 * 12, # N (X,Y,Yaw) 72
        weights_scaling=[1.0, 1.0, 1.0], # 6
        criterion=nn.L1Loss(reduction="none"),
        global_head_dropout=cfg["model_params"]["global_head_dropout"],
        disable_other_agents=cfg["model_params"]["disable_other_agents"],
        disable_map=cfg["model_params"]["disable_map"],
        disable_lane_boundaries=cfg["model_params"]["disable_lane_boundaries"])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = "/workspace/source/src/model/OL_HS.pt"
    pretrained_policy.load_state_dict(torch.load(model_path).state_dict()).to(device)
    # pretrain_dist = PretrainedDistribution(pretrained_policy)

    config_param_space = {
        "env": "L5-CLE-V2",
        "framework": "torch",
        "num_gpus": 1,
        "num_workers": 3,
        "num_envs_per_worker": 4,
        'disable_env_checking':False,
        # "postprocess_fn": my_postprocess_fn,
        "pretrained_policy": pretrained_policy,
        "model": {
                "custom_model": "TorchSeparatedAttentionModel",
                # Extra kwargs to be passed to your model's c'tor.
                "custom_model_config": {'cfg':cfg},
                "custom_options": {
                    "squash_output": False, # Turn off output squashing
                },
                # "custom_action_distribution_fn": gaussian_action_distribution_fn,
                # "custom_action_dist": DiagGaussianDistribution,
                # "custom_action_dist_cls": "CustomTorchActionDist",
                # "squash_to_range": True,
                # "logit_dim": 6,
                # "free_log_std": True,
                # "std_share_network": False,
                },

        '_disable_preprocessor_api': True,
        "eager_tracing": True,
        "restart_failed_sub_environments": True,
        "lr": 0.00001,
        'seed': 42,
        # "lr_schedule": [
        #     [1e6, lr_start],
        #     [2e6, lr_end],
        # ],
        'train_batch_size': 128, # 8000 
        'sgd_minibatch_size': 32, #2048
        'num_sgd_iter': 10,#16,
        'seed': 42,
        # 'batch_mode': 'truncate_episodes',
        # "rollout_fragment_length": 32,
        'gamma': 0.8,    
    }


    # result_grid = tune.Tuner(
    #     "PPO",
    #     run_config=air.RunConfig(
    #         stop={"episode_reward_mean": 0, 'timesteps_total': int(6e6)},
    #         local_dir=ray_result_logdir,
    #         checkpoint_config=air.CheckpointConfig(num_to_keep=2, 
    #                                             checkpoint_frequency = 10, 
    #                                             checkpoint_score_attribute = 'episode_reward_mean'),
    #         # callbacks=[WandbLoggerCallback(project="l5kit2", save_code = True, save_checkpoints = False),],
    #         ),
    #     param_space=config_param_space).fit()
    # from ray.rllib.agents.ppo import PPOTrainer
    # from ray.rllib.agents.ppo.ppo import PPOTrainer
    from ray.rllib.algorithms.ppo import PPO
    
    # trainer = KLPPO(obs_space= l5_env2.observation_space,
    #                 action_space =l5_env2.action_space,
    #                 config=config_param_space)

    # CustomTrainer = PPO.with_updates(get_policy_class=lambda config:KLPPOPolicy)
    from src.customModel.customPPOTrainer import KLPPO
    # class KLPPO(PPO):
    #     def get_default_policy_class(
    #         cls, config
    #     ):
    #         return KLPPOTorchPolicy(l5_env2.observation_space, l5_env2.action_space, config_param_space)

    trainer = KLPPO(config=config_param_space)
    from ray.tune.logger import pretty_print
    for i in range(10000):
        print('alo')
        result = trainer.train()
        print(pretty_print(result))

        
# from ray.rllib.models.tf.misc import normc_initializer
# from ray.rllib.models.tf.tf_modelv2 import TFModelV2
# from ray.rllib.utils.framework import try_import_tf
# from tensorflow import keras


# tf1, tf, tfv = try_import_tf()
# layers = tf.keras.layers

# class TFGNNCNN(TFModelV2):
#     """Custom model for policy gradient algorithms."""

#     def __init__(self, obs_space, action_space, num_outputs, model_config, name):
#         super(TFGNNCNN, self).__init__(
#             obs_space, action_space, num_outputs, model_config, name
#         )
#         self._num_objects = obs_space.shape[2] # num_of_channels of input, size x size x channels
#         self._num_actions = num_outputs
#         self._feature_dim = model_config["custom_model_config"]['feature_dim']
#         self.inputs = tf.keras.layers.Input(shape=obs_space.shape, name="observations")
#         layer_1 = layers.Conv2D(64,kernel_size= (7,7), strides=(2,2), padding='same', use_bias=False, kernel_initializer=normc_initializer(1.0))(self.inputs)
#         layer_2 = layers.GroupNormalization(64)(layer_1)
#         layer_3 = layers.ReLU()(layer_2)
#         layer_4 = layers.MaxPool2D(pool_size=(2,2), strides=2)(layer_3) # x
#         layer_5 = layers.Conv2D(32,kernel_size= (7,7), strides=(2,2), padding='same', use_bias=False, kernel_initializer=normc_initializer(1.0))(layer_4)
#         layer_6 = layers.GroupNormalization(32)(layer_5),
#         # raise ValueError(layer_6.__repr__())
#         layer_7 = layers.ReLU()(layer_6)
#         layer_8 = layers.MaxPool2D(pool_size=(2,2), strides=2)(layer_7)
#         layer_9 = layers.Flatten()(layer_8)
#         layer_10 = layers.Dense(self._feature_dim)(layer_9)
#             # nn.Conv2d(self._num_objects, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
#             # nn.GroupNorm(4, 64),
#             # nn.ReLU(),
#             # nn.MaxPool2d(kernel_size=2, stride=2),
#             # nn.Conv2d(64, 32, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
#             # nn.GroupNorm(2, 32),
#             # nn.ReLU(),
#             # nn.MaxPool2d(kernel_size=2, stride=2),
#             # nn.Flatten(),
#             # nn.Linear(in_features=1568, out_features=self._feature_dim),

#         actor_out_1 = layers.Dense( 256, kernel_initializer=normc_initializer(0.01),)(layer_10)
#         actor_out_2= layers.ReLU()(actor_out_1)
#         actor_out_3 = layers.Dense( num_outputs, kernel_initializer=normc_initializer(0.01),name='critic_out')(actor_out_2)
#         critic_out = tf.keras.layers.Dense( 1, name="value_out", activation=None, kernel_initializer=normc_initializer(0.01),)(layer_1)

#         self.base_model = tf.keras.Model(self.inputs, [actor_out_3, critic_out])

#         # self.network = tf.keras.Sequential(
#         #     [
#         #         tf.keras.Input(shape=obs_space.shape),
#         #         layers.Conv2D(64,kernel_size= (7,7), strides=(2,2), padding='same', use_bias=False),
#         #         layers.GroupNormalization(64),
#         #         layers.ReLU(),
#         #         layers.MaxPool2D(pool_size=(2,2), strides=2),
#         #         layers.Conv2D(32,kernel_size= (7,7), strides=(2,2), padding='same', use_bias=False),
#         #         layers.GroupNormalization(32),
#         #         layers.ReLU(),
#         #         layers.MaxPool2D(pool_size=(2,2), strides=2),
#         #         layers.Flatten(),
#         #         layers.Dense(self._feature_dim)
#         #     ]
#         # )
#         # self._actor_head = tf.keras.Sequential(
#         #     [
#         #         layers.Dense(256),
#         #         layers.ReLU(),
#         #         layers.Dense(self._num_actions)
#         #     ]
#         # )

#         # self._critic_head = tf.keras.Sequential([
#         #         layers.Dense(1),
#         # ]
#         # )
#             # nn.Conv2d(self._num_objects, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
#             # nn.GroupNorm(4, 64),
#             # nn.ReLU(),
#             # nn.MaxPool2d(kernel_size=2, stride=2),
#             # nn.Conv2d(64, 32, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
#             # nn.GroupNorm(2, 32),
#             # nn.ReLU(),
#             # nn.MaxPool2d(kernel_size=2, stride=2),
#             # nn.Flatten(),
#             # nn.Linear(in_features=1568, out_features=self._feature_dim),
#         # self.inputs = tf.keras.layers.Input(shape=obs_space.shape, name="observations")
#         # layer_1 = tf.keras.layers.Dense( 128,name="my_layer1",activation=tf.nn.relu,kernel_initializer=normc_initializer(1.0),)(self.inputs)
#         # layer_out = tf.keras.layers.Dense(
#         #     num_outputs,
#         #     name="my_out",
#         #     activation=None,
#         #     kernel_initializer=normc_initializer(0.01),
#         # )(layer_1)
#         # value_out = tf.keras.layers.Dense(
#         #     1,
#         #     name="value_out",
#         #     activation=None,
#         #     kernel_initializer=normc_initializer(0.01),
#         # )(layer_1)
#         # self.base_model = tf.keras.Model(self.inputs, [layer_out, value_out])

#     def forward(self, input_dict, state, seq_lens):
#         # obs_transformed = input_dict['obs'].permute(0, 3, 1, 2) # 32 x 112 x 112 x 7 [B, size, size, channels]
#         # raise ValueError(input_dict["obs"].shape)
#         actor_out, self._value_out = self.base_model(input_dict["obs"])
#         return actor_out, state

#     def value_function(self):
#         return tf.reshape(self._value_out, [-1])
# if __name__ == '__main__':
#     def testGCNN():
