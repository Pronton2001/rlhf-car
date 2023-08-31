from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch.nn as nn
import torch
from torch.nn import functional as F
from l5kit.planning.vectorized.common import pad_avail, pad_points
from l5kit.planning.vectorized.global_graph import VectorizedEmbedding
from l5kit.planning.vectorized.local_graph import LocalSubGraph, SinusoidalPositionalEmbedding
from src.customEnv.action_utils import standard_normalizer_nonKin
from l5kit.planning.vectorized.open_loop_model import VectorizedModel
from torchvision.models.resnet import resnet18, resnet50
import os
import numpy as np
# os.environ['CUDA_VISIBLE_DEVICES']= '0'

import logging
from src.constant import SRC_PATH
logging.basicConfig(filename=SRC_PATH + 'src/log/info.log', level=logging.DEBUG, filemode='w')
from src.customModel.CustomVectorizedModel import CustomVectorizedModel
from ray.rllib.algorithms.sac.sac_torch_model import SACTorchModel
import gym
from gym.spaces import Box, Discrete
import tree  # pip install dm_tree
from typing import Dict, List, Optional, Tuple

from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.spaces.simplex import Simplex
from ray.rllib.utils.typing import ModelConfigDict, TensorType, TensorStructType
from ray.rllib.models.torch.misc import (
    normc_initializer as torch_normc_initializer,
    SlimFC,
    normc_initializer,
    same_padding,
    SlimConv2d,
)
from ray.rllib.models.torch.fcnet import (
    FullyConnectedNetwork as TorchFullyConnectedNetwork,
)

from ray.rllib.models.utils import get_activation_fn, get_filter_config

from l5kit.environment import models
import warnings
torch, nn = try_import_torch()

# class TorchRasterQNet(TorchModelV2):
#     def __init__(self, obs_space, action_space, num_outputs, model_config, name):
#             super().__init__(obs_space, action_space, num_outputs, model_config, name)


def SimpleCNN_GN(num_input_channels: int, features_dim: int) -> nn.Module:
    """A simplified feature extractor with GroupNorm.

    :param num_input_channels: the number of input channels in the input
    :param features_dim: the number of features to extract from input
    """
    in_features = 1568 if num_input_channels == 7 else 6272 #7x122x122, 5x224x224 image
    # in_features = 800
    model = nn.Sequential(
        nn.Conv2d(num_input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
        nn.GroupNorm(4, 64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(64, 32, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
        nn.GroupNorm(2, 32),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(in_features=in_features, out_features=features_dim),
    )

    return model

def visionNet(num_input_channels: int, features_dim: int) -> nn.Module:

    _convs = nn.Sequential(
        nn.ZeroPad2d((2, 2, 2, 2)),
        nn.Conv2d(num_input_channels, 16, kernel_size=(8, 8), stride=(4, 4)),
        # nn.GroupNorm(4, 64),
        nn.ZeroPad2d((1, 2, 1, 2)),
        nn.ReLU(),
        # nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(16, 32, kernel_size=(4, 4), stride=(2, 2)),
        # nn.GroupNorm(2, 32),
        nn.ReLU(),

        nn.Conv2d(32, 256, kernel_size=(11, 11), stride=(1, 1)),
        nn.ReLU(),
        # nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(256, 256, bias=True),
    )

    return _convs


class RasterizedPlanningModelFeature(nn.Module):
    """Raster-based planning model."""

    def __init__(
        self,
        model_arch: str,
        num_input_channels: int,
        num_targets: int,
        weights_scaling: List[float],
        criterion: nn.Module,
        pretrained: bool = True,
    ) -> None:
        """Initializes the planning model.

        :param model_arch: model architecture to use
        :param num_input_channels: number of input channels in raster
        :param num_targets: number of output targets
        :param weights_scaling: target weights for loss calculation
        :param criterion: loss function to use
        :param pretrained: whether to use pretrained weights
        """
        super().__init__()
        self.model_arch = model_arch
        self.num_input_channels = num_input_channels
        self.num_targets = num_targets
        self.register_buffer("weights_scaling", torch.tensor(weights_scaling))
        self.pretrained = pretrained
        self.criterion = criterion

        if pretrained and self.num_input_channels != 3:
            warnings.warn("There is no pre-trained model with num_in_channels != 3, first layer will be reset")

        if model_arch == "resnet18":
            self.model = resnet18(pretrained=pretrained)
            self.model.fc = nn.Linear(in_features=512, out_features=num_targets)
        elif model_arch == "resnet50":
            self.model = resnet50(pretrained=pretrained)
            self.model.fc = nn.Linear(in_features=2048, out_features=num_targets)
        elif model_arch == "simple_cnn":
            self.model = SimpleCNN_GN(self.num_input_channels, num_targets)
        else:
            raise NotImplementedError(f"Model arch {model_arch} unknown")

        if model_arch in {"resnet18", "resnet50"} and self.num_input_channels != 3:
            self.model.conv1 = nn.Conv2d(
                in_channels=self.num_input_channels,
                out_channels=64,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            )
    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        image_batch = data_batch
        return self.model(image_batch)



class RasterizedPlanningModelFeature2(nn.Module):
    """Raster-based planning model."""

    def __init__(
        self,
        num_input_channels: int,
        num_targets: int,
    ) -> None:
        """Initializes the planning model.

        :param model_arch: model architecture to use
        :param num_input_channels: number of input channels in raster
        :param num_targets: number of output targets
        :param weights_scaling: target weights for loss calculation
        :param criterion: loss function to use
        :param pretrained: whether to use pretrained weights
        """
        super().__init__()
        self.num_input_channels = num_input_channels
        self.num_targets = num_targets
        self.model = SimpleCNN_GN(self.num_input_channels, num_targets)
       
    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        image_batch = data_batch.float()
        return self.model(image_batch)


class RasterizedPlanningModelFeature3(nn.Module):
    """Raster-based planning model."""

    def __init__(
        self,
        num_input_channels: int,
        num_targets: int,
    ) -> None:
        """Initializes the planning model.

        :param model_arch: model architecture to use
        :param num_input_channels: number of input channels in raster
        :param num_targets: number of output targets
        :param weights_scaling: target weights for loss calculation
        :param criterion: loss function to use
        :param pretrained: whether to use pretrained weights
        """
        super().__init__()
        self.num_input_channels = num_input_channels
        self.num_targets = num_targets
        self.model = visionNet(self.num_input_channels, num_targets)
       
    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        image_batch = data_batch.float()
        return self.model(image_batch)

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
        action = standard_normalizer_nonKin(action, self.non_kin_rescale) # take first action
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
def freeze_attention_model_except_last_3fc(model):
    for  name, param in model.named_parameters():
        if 'global_head.output_embed.layers' not in name: 
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

def load_attention_model_except_last_3fc(model):
    weights_subset = {}
    for key, value in model.state_dict().items():
        if 'global_head.output_embed.layers' in key:
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
        actor_feature_dim = 256 
        crtitic_feature_dim = 256
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
            action_layers=[
                SlimFC(actor_feature_dim, out_size=num_outputs, initializer= torch_normc_initializer(1.0)) ,
                # SlimFC(256, out_size=num_outputs, initializer= torch_normc_initializer(1.0), activation_fn=None)
            ]
            self._actor_mlp =nn.Sequential(*action_layers)
        else:
            self._actor_mlp = nn.Sequential()
        if crtitic_feature_dim != 1:
            critic_layers=[
                SlimFC(crtitic_feature_dim, out_size=1, initializer= torch_normc_initializer(0.01))
            ]
            self._critic_mlp =nn.Sequential(*critic_layers)
        else:
            self._critic_mlp = nn.Sequential()
        # if actor_feature_dim != num_outputs:
        #     self._actor_mlp = nn.Sequential(
        #         nn.Linear(actor_feature_dim, 256),
        #         nn.ReLU(),
        #         nn.Linear(256, num_outputs),
        #     )
        # else:
        #     self._actor_mlp = nn.Sequential()
        # if crtitic_feature_dim != 1:
        #     self._critic_mlp = nn.Sequential(
        #         # nn.BatchNorm1d(crtitic_feature_dim),
        #         # nn.ReLU(),
        #         nn.Linear(crtitic_feature_dim, 1)
        #     )
        # else:
        #     self._critic_mlp = nn.Sequential()

    def forward(self, input_dict, state, seq_lens):
        obs_transformed = input_dict['obs'].float()
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
        logging.debug(f'policy in custom model obs {obs_transformed.mean()}, logits: {logits}, value: {self._value}')

        # return output_logits, state
        return logits, state
    def value_function(self):
        return self._value

class TorchRasterNetMixedActor(TorchModelV2, nn.Module): #TODO: recreate rasternet for PPO, check again
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        if not model_config.get("conv_filters"):
            model_config["conv_filters"] = get_filter_config(obs_space.shape)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.log_std_x = -5
        self.log_std_y = -5
        self.log_std_yaw = -5

        n_channels = model_config["custom_model_config"]['n_channels'] 
        critic_net = model_config["custom_model_config"]['critic_net']
        actor_net = model_config["custom_model_config"]['actor_net']
        future_num_frames = model_config["custom_model_config"]['future_num_frames']
        freeze_actor = model_config["custom_model_config"]['freeze_actor'] 
        shared_feature_extractor =  model_config["custom_model_config"]['shared_feature_extractor'] 
        actor_feature_dim = 256 
        crtitic_feature_dim = 256
        activation = self.model_config.get("conv_activation")
        filters = self.model_config["conv_filters"]
        assert len(filters) > 0, "Must provide at least 1 entry in `conv_filters`!"
        self.last_layer_is_flattened = False
        self._logits = None
        no_final_linear = self.model_config.get("no_final_linear")
        vf_share_layers = self.model_config.get("vf_share_layers")

        # Build the action layers
        activation = self.model_config.get("conv_activation")
        filters = self.model_config["conv_filters"]
        assert len(filters) > 0, "Must provide at least 1 entry in `conv_filters`!"

        # Post FC net config.
        # post_fcnet_hiddens = model_config.get("post_fcnet_hiddens", [])
        # post_fcnet_activation = get_activation_fn(
        #     model_config.get("post_fcnet_activation"), framework="torch"
        # )

        no_final_linear = self.model_config.get("no_final_linear")
        vf_share_layers = self.model_config.get("vf_share_layers")

        # Whether the last layer is the output of a Flattened (rather than
        # a n x (1,1) Conv2D).
        self.last_layer_is_flattened = False
        self._logits = None
        layers = []
        (w, h, in_channels) = obs_space.shape

        in_size = [w, h]
        for out_channels, kernel, stride in filters[:-1]:
            padding, out_size = same_padding(in_size, kernel, stride)
            layers.append(
                SlimConv2d(
                    in_channels,
                    out_channels,
                    kernel,
                    stride,
                    padding,
                    activation_fn=activation,
                )
            )
            in_channels = out_channels
            in_size = out_size

        out_channels, kernel, stride = filters[-1]
        layers.append(
                SlimConv2d(
                    in_channels,
                    out_channels,
                    kernel,
                    stride,
                    None,  # padding=valid
                    activation_fn=activation,
                )
            )
        in_size = [
                    np.ceil((in_size[0] - kernel[0]) / stride),
                    np.ceil((in_size[1] - kernel[1]) / stride),
                ]
        padding, _ = same_padding(in_size, [1, 1], [1, 1])
        layers.append(nn.Flatten())
        in_size = out_channels
        # Add (optional) post-fc-stack after last Conv2D layer.
        post_fcnet_hiddens = [256]
        post_fcnet_activation = get_activation_fn(
            'tanh', framework="torch"
        )
        for i, out_size in enumerate(post_fcnet_hiddens + [num_outputs]):
            layers.append(
                SlimFC(
                    in_size=in_size,
                    out_size=out_size,
                    activation_fn=post_fcnet_activation
                    if i < len(post_fcnet_hiddens) - 1
                    else None,
                    initializer=normc_initializer(1.0),
                )
            )
            in_size = out_size
        # Last layer is logits layer.
        self._logits = layers.pop()


        # self._convs = nn.Sequential(*layers)
        actor_feature_dim = 256
        self._convs =RasterizedPlanningModelFeature(
            model_arch=actor_net,
            num_input_channels=n_channels,
            num_targets= actor_feature_dim , # or same as pretrained_policy to load
            weights_scaling=[1., 1., 1.],
            criterion=nn.MSELoss(reduction="none"),)
        if actor_feature_dim != num_outputs:
            action_layers=[
                SlimFC(actor_feature_dim, out_size=num_outputs, initializer= torch_normc_initializer(1.0)) ,
                # SlimFC(256, out_size=num_outputs, initializer= torch_normc_initializer(1.0), activation_fn=None)
            ]
            self._logits =nn.Sequential(*action_layers)
        else:
            self._logits_actor_mlp = nn.Sequential()


        # If our num_outputs still unknown, we need to do a test pass to
        # figure out the output dimensions. This could be the case, if we have
        # the Flatten layer at the end.
        if self.num_outputs is None:
            # Create a B=1 dummy sample and push it through out conv-net.
            dummy_in = (
                torch.from_numpy(self.obs_space.sample())
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
            )
            dummy_out = self._convs(dummy_in)
            self.num_outputs = dummy_out.shape[1]

        # Build the value layers
        self._value_branch_separate = self._value_branch = None
        if vf_share_layers:
            self._value_branch = SlimFC(
                out_channels, 1, initializer=normc_initializer(0.01), activation_fn=None
            )
        else:
            vf_layers = []
            (w, h, in_channels) = obs_space.shape
            in_size = [w, h]
            for out_channels, kernel, stride in filters[:-1]:
                padding, out_size = same_padding(in_size, kernel, stride)
                vf_layers.append(
                    SlimConv2d(
                        in_channels,
                        out_channels,
                        kernel,
                        stride,
                        padding,
                        activation_fn=activation,
                    )
                )
                in_channels = out_channels
                in_size = out_size

            out_channels, kernel, stride = filters[-1]
            vf_layers.append(
                SlimConv2d(
                    in_channels,
                    out_channels,
                    kernel,
                    stride,
                    None,
                    activation_fn=activation,
                )
            )

            vf_layers.append(
                SlimConv2d(
                    in_channels=out_channels,
                    out_channels=1,
                    kernel=1,
                    stride=1,
                    padding=None,
                    activation_fn=None,
                )
            )
            self._value_branch_separate = nn.Sequential(*vf_layers)

        # Holds the current "base" output (before logits layer).
        self._features = None 
    
    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):

        self._features = input_dict["obs"].float()
        self._features = self._features.permute(0, 3, 1, 2)
        conv_out = self._convs(self._features)
        logits = self._logits(conv_out)
        return logits, state
        # obs = input_dict['obs']
        # traj = self._actor_head(obs)
        # logits = self._actor_mlp(traj)
        # value = self._critic_mlp(self._critic_head(obs))
        # self._value = value.view(-1)
        # logging.debug(f'policy in custom model obs {obs.mean()}, logits: {logits}, value: {self._value}')
        return logits, state

    # @override(TorchModelV2)
    # def forward(
    #     self,
    #     input_dict: Dict[str, TensorType],
    #     state: List[TensorType],
    #     seq_lens: TensorType,
    # ) -> (TensorType, List[TensorType]):
    #     self._features = input_dict["obs"].float()
    #     # Permuate b/c data comes in as [B, dim, dim, channels]:
    #     self._features = self._features.permute(0, 3, 1, 2)
    #     conv_out = self._convs(self._features)
    #     # Store features to save forward pass when getting value_function out.
    #     if not self._value_branch_separate:
    #         self._features = conv_out

    #     if not self.last_layer_is_flattened: # TODO: check this
    #         if self._logits:
    #             conv_out = self._logits(conv_out)
    #         if len(conv_out.shape) == 4:
    #             if conv_out.shape[2] != 1 or conv_out.shape[3] != 1:
    #                 raise ValueError(
    #                     "Given `conv_filters` ({}) do not result in a [B, {} "
    #                     "(`num_outputs`), 1, 1] shape (but in {})! Please "
    #                     "adjust your Conv2D stack such that the last 2 dims "
    #                     "are both 1.".format(
    #                         self.model_config["conv_filters"],
    #                         self.num_outputs,
    #                         list(conv_out.shape),
    #                     )
    #                 )
    #             logits = conv_out.squeeze(3)
    #             logits = logits.squeeze(2)
    #         else:
    #             logits = conv_out
    #         return logits, state
    #     else:
    #         return conv_out, state
    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        # if self._value_branch_separate:
        value = self._value_branch_separate(self._features)
        value = value.squeeze(3)
        value = value.squeeze(2)
        return value.squeeze(1)
        # else:
        #     if not self.last_layer_is_flattened:
        #         features = self._features.squeeze(3)
        #         features = features.squeeze(2)
        #     else:
        #         features = self._features
        #     return self._value_branch(features).squeeze(1)
    def _hidden_layers(self, obs: TensorType) -> TensorType:
        res = self._convs(obs)  # switch to channel-major
        res = res.squeeze(3)
        res = res.squeeze(2)
        return res

class TorchRasterNetCustomVisionNetActor(TorchModelV2, nn.Module): #TODO: recreate rasternet for PPO, check again
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        if not model_config.get("conv_filters"):
            model_config["conv_filters"] = get_filter_config(obs_space.shape)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.log_std_x = -5
        self.log_std_y = -5
        self.log_std_yaw = -5

        n_channels = model_config["custom_model_config"]['n_channels'] 
        critic_net = model_config["custom_model_config"]['critic_net']
        actor_net = model_config["custom_model_config"]['actor_net']
        future_num_frames = model_config["custom_model_config"]['future_num_frames']
        freeze_actor = model_config["custom_model_config"]['freeze_actor'] 
        shared_feature_extractor =  model_config["custom_model_config"]['shared_feature_extractor'] 
        actor_feature_dim = 256 
        crtitic_feature_dim = 256
        activation = self.model_config.get("conv_activation")
        filters = self.model_config["conv_filters"]
        assert len(filters) > 0, "Must provide at least 1 entry in `conv_filters`!"
        self.last_layer_is_flattened = False
        self._logits = None
        no_final_linear = self.model_config.get("no_final_linear")
        vf_share_layers = self.model_config.get("vf_share_layers")

        # Build the action layers
        activation = self.model_config.get("conv_activation")
        filters = self.model_config["conv_filters"]
        assert len(filters) > 0, "Must provide at least 1 entry in `conv_filters`!"

        # Post FC net config.
        # post_fcnet_hiddens = model_config.get("post_fcnet_hiddens", [])
        # post_fcnet_activation = get_activation_fn(
        #     model_config.get("post_fcnet_activation"), framework="torch"
        # )

        no_final_linear = self.model_config.get("no_final_linear")
        vf_share_layers = self.model_config.get("vf_share_layers")

        self.last_layer_is_flattened = False
        self._logits = None
        self._convs = RasterizedPlanningModelFeature3(
            num_input_channels=n_channels,
            num_targets=num_outputs,
        )
        self._logits = nn.Sequential(
            nn.Linear(256, num_outputs, bias = True)
        )
        
        
        if self.num_outputs is None:
            # Create a B=1 dummy sample and push it through out conv-net.
            dummy_in = (
                torch.from_numpy(self.obs_space.sample())
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
            )
            dummy_out = self._convs(dummy_in)
            self.num_outputs = dummy_out.shape[1]

        # Build the value layers
        self._value_branch_separate = self._value_branch = None
        if vf_share_layers:
            self._value_branch = SlimFC(
                out_channels, 1, initializer=normc_initializer(0.01), activation_fn=None
            )
        else:
            vf_layers = []
            (w, h, in_channels) = obs_space.shape
            in_size = [w, h]
            for out_channels, kernel, stride in filters[:-1]:
                padding, out_size = same_padding(in_size, kernel, stride)
                vf_layers.append(
                    SlimConv2d(
                        in_channels,
                        out_channels,
                        kernel,
                        stride,
                        padding,
                        activation_fn=activation,
                    )
                )
                in_channels = out_channels
                in_size = out_size

            out_channels, kernel, stride = filters[-1]
            vf_layers.append(
                SlimConv2d(
                    in_channels,
                    out_channels,
                    kernel,
                    stride,
                    None,
                    activation_fn=activation,
                )
            )

            vf_layers.append(
                SlimConv2d(
                    in_channels=out_channels,
                    out_channels=1,
                    kernel=1,
                    stride=1,
                    padding=None,
                    activation_fn=None,
                )
            )
            self._value_branch_separate = nn.Sequential(*vf_layers)

        # Holds the current "base" output (before logits layer).
        self._features = None 
    
    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):

        self._features = input_dict["obs"].float()
        self._features = self._features.permute(0, 3, 1, 2)
        conv_out = self._convs(self._features)
        logits = self._logits(conv_out)
        return logits, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        value = self._value_branch_separate(self._features)
        value = value.squeeze(3)
        value = value.squeeze(2)
        return value.squeeze(1)

    def _hidden_layers(self, obs: TensorType) -> TensorType:
        res = self._convs(obs)  # switch to channel-major
        res = res.squeeze(3)
        res = res.squeeze(2)
        return res

class TorchRasterNetMixedCritic(TorchModelV2, nn.Module): #TODO: recreate rasternet for PPO, check again
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        if not model_config.get("conv_filters"):
            model_config["conv_filters"] = get_filter_config(obs_space.shape)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.log_std_x = -5
        self.log_std_y = -5
        self.log_std_yaw = -5

        n_channels = model_config["custom_model_config"]['n_channels'] 
        critic_net = model_config["custom_model_config"]['critic_net']
        actor_net = model_config["custom_model_config"]['actor_net']
        future_num_frames = model_config["custom_model_config"]['future_num_frames']
        freeze_actor = model_config["custom_model_config"]['freeze_actor'] 
        shared_feature_extractor =  model_config["custom_model_config"]['shared_feature_extractor'] 
        actor_feature_dim = 256 
        crtitic_feature_dim = 256
        activation = self.model_config.get("conv_activation")
        filters = self.model_config["conv_filters"]
        assert len(filters) > 0, "Must provide at least 1 entry in `conv_filters`!"
        self.last_layer_is_flattened = False
        self._logits = None
        no_final_linear = self.model_config.get("no_final_linear")
        vf_share_layers = self.model_config.get("vf_share_layers")

        # Build the action layers
        activation = self.model_config.get("conv_activation")
        filters = self.model_config["conv_filters"]
        assert len(filters) > 0, "Must provide at least 1 entry in `conv_filters`!"

        # Post FC net config.
        # post_fcnet_hiddens = model_config.get("post_fcnet_hiddens", [])
        # post_fcnet_activation = get_activation_fn(
        #     model_config.get("post_fcnet_activation"), framework="torch"
        # )

        no_final_linear = self.model_config.get("no_final_linear")
        vf_share_layers = self.model_config.get("vf_share_layers")

        # Whether the last layer is the output of a Flattened (rather than
        # a n x (1,1) Conv2D).
        self.last_layer_is_flattened = False
        self._logits = None
        layers = []
        (w, h, in_channels) = obs_space.shape

        in_size = [w, h]
        for out_channels, kernel, stride in filters[:-1]:
            padding, out_size = same_padding(in_size, kernel, stride)
            layers.append(
                SlimConv2d(
                    in_channels,
                    out_channels,
                    kernel,
                    stride,
                    padding,
                    activation_fn=activation,
                )
            )
            in_channels = out_channels
            in_size = out_size

        out_channels, kernel, stride = filters[-1]
        layers.append(
                SlimConv2d(
                    in_channels,
                    out_channels,
                    kernel,
                    stride,
                    None,  # padding=valid
                    activation_fn=activation,
                )
            )
        in_size = [
                    np.ceil((in_size[0] - kernel[0]) / stride),
                    np.ceil((in_size[1] - kernel[1]) / stride),
                ]
        padding, _ = same_padding(in_size, [1, 1], [1, 1])
        layers.append(nn.Flatten())
        in_size = out_channels
        # Add (optional) post-fc-stack after last Conv2D layer.
        post_fcnet_hiddens = [256]
        post_fcnet_activation = get_activation_fn(
            'tanh', framework="torch"
        )
        for i, out_size in enumerate(post_fcnet_hiddens + [num_outputs]):
            layers.append(
                SlimFC(
                    in_size=in_size,
                    out_size=out_size,
                    activation_fn=post_fcnet_activation
                    if i < len(post_fcnet_hiddens) - 1
                    else None,
                    initializer=normc_initializer(1.0),
                )
            )
            in_size = out_size
        # Last layer is logits layer.
        self._logits = layers.pop()
        self._convs = nn.Sequential(*layers)

        # If our num_outputs still unknown, we need to do a test pass to
        # figure out the output dimensions. This could be the case, if we have
        # the Flatten layer at the end.
        if self.num_outputs is None:
            # Create a B=1 dummy sample and push it through out conv-net.
            dummy_in = (
                torch.from_numpy(self.obs_space.sample())
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
            )
            dummy_out = self._convs(dummy_in)
            self.num_outputs = dummy_out.shape[1]

        # Build the value layers
        self._value_branch_separate = self._value_branch = None
        if vf_share_layers:
            self._value_branch = SlimFC(
                out_channels, 1, initializer=normc_initializer(0.01), activation_fn=None
            )
        else:
            vf_layers = []
            (w, h, in_channels) = obs_space.shape
            in_size = [w, h]
            for out_channels, kernel, stride in filters[:-1]:
                padding, out_size = same_padding(in_size, kernel, stride)
                vf_layers.append(
                    SlimConv2d(
                        in_channels,
                        out_channels,
                        kernel,
                        stride,
                        padding,
                        activation_fn=activation,
                    )
                )
                in_channels = out_channels
                in_size = out_size

            out_channels, kernel, stride = filters[-1]
            vf_layers.append(
                SlimConv2d(
                    in_channels,
                    out_channels,
                    kernel,
                    stride,
                    None,
                    activation_fn=activation,
                )
            )

            vf_layers.append(
                SlimConv2d(
                    in_channels=out_channels,
                    out_channels=1,
                    kernel=1,
                    stride=1,
                    padding=None,
                    activation_fn=None,
                )
            )
            self._value_branch_separate = nn.Sequential(*vf_layers)

        # Holds the current "base" output (before logits layer).
        self._features = None 
        


        self._value_branch_separate = RasterizedPlanningModelFeature2(
                num_input_channels=n_channels,
                num_targets=crtitic_feature_dim,  # feature dim of critic
        )
        if crtitic_feature_dim != 1: # TODO: Continnue
            critic_layers=[
                SlimFC(crtitic_feature_dim, out_size=1, initializer= torch_normc_initializer(0.01))
            ]
            self._critic_mlp =nn.Sequential(*critic_layers)
        else:
            self._critic_mlp = nn.Sequential()
    
    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):

        self._features = input_dict["obs"].float()
        self._features = self._features.permute(0, 3, 1, 2)
        conv_out = self._convs(self._features)
        logits = self._logits(conv_out)
        return logits, state
        # obs = input_dict['obs']
        # traj = self._actor_head(obs)
        # logits = self._actor_mlp(traj)
        # value = self._critic_mlp(self._critic_head(obs))
        # self._value = value.view(-1)
        # logging.debug(f'policy in custom model obs {obs.mean()}, logits: {logits}, value: {self._value}')
        return logits, state

    # @override(TorchModelV2)
    # def forward(
    #     self,
    #     input_dict: Dict[str, TensorType],
    #     state: List[TensorType],
    #     seq_lens: TensorType,
    # ) -> (TensorType, List[TensorType]):
    #     self._features = input_dict["obs"].float()
    #     # Permuate b/c data comes in as [B, dim, dim, channels]:
    #     self._features = self._features.permute(0, 3, 1, 2)
    #     conv_out = self._convs(self._features)
    #     # Store features to save forward pass when getting value_function out.
    #     if not self._value_branch_separate:
    #         self._features = conv_out

    #     if not self.last_layer_is_flattened: # TODO: check this
    #         if self._logits:
    #             conv_out = self._logits(conv_out)
    #         if len(conv_out.shape) == 4:
    #             if conv_out.shape[2] != 1 or conv_out.shape[3] != 1:
    #                 raise ValueError(
    #                     "Given `conv_filters` ({}) do not result in a [B, {} "
    #                     "(`num_outputs`), 1, 1] shape (but in {})! Please "
    #                     "adjust your Conv2D stack such that the last 2 dims "
    #                     "are both 1.".format(
    #                         self.model_config["conv_filters"],
    #                         self.num_outputs,
    #                         list(conv_out.shape),
    #                     )
    #                 )
    #             logits = conv_out.squeeze(3)
    #             logits = logits.squeeze(2)
    #         else:
    #             logits = conv_out
    #         return logits, state
    #     else:
    #         return conv_out, state
    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        # value = self._value_branch_separate(self._features)
        # value = value.squeeze(3)
        # value = value.squeeze(2)
        # return value.squeeze(1)

        value = self._value_branch_separate(self._features)
        value = self._critic_mlp(value).view(-1)
        
        return value

    def _hidden_layers(self, obs: TensorType) -> TensorType:
        res = self._convs(obs)  # switch to channel-major
        res = res.squeeze(3)
        res = res.squeeze(2)
        return res

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

    def forward(self, input_dict, state, seq_lens): # from dataloader? get 32, 112, 112, 7
        # obs_transformed = input_dict['obs'].permute(0, 3, 1, 2) # 32 x 112 x 112 x 7 [B, size, size, channels]
        obs_transformed = input_dict['obs'].permute(0, 3, 1, 2).float() # [B, C, W, H] -> [B, W, H, C]
        # print('forward', obs_transformed.shape)
        network_output = self.network(obs_transformed)
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
        obs_transformed = input_dict['obs'].permute(0, 3, 1, 2).float() # input_dict['obs'].shape = [B, size, size, # channels] => obs_transformed.shape = [B, # channels, size, size]
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

class TorchAttentionModel2(TorchModelV2, nn.Module): # TODO: Delete
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

class TorchAttentionModel(TorchModelV2, nn.Module): #TODO: Delete
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

class TorchAttentionModel3(TorchModelV2, nn.Module): #TODO: Delete
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

        output_logits = torch.cat((pred_x,pred_y, pred_yaw), dim = -1)
        print(f'----------> before actions: {output_logits}')
        output_logits = standard_normalizer_nonKin(output_logits)
        print(f'----------> after actions: {output_logits}')
        ones = torch.ones_like(pred_x) # 32,
        # assert ones.shape[1] == 1, f'{ones.shape[1]}'
        # output_logits_mean = torch.cat((pred_x, pred_y, pred_yaw), dim = -1)
        output_logits = torch.cat((output_logits, ones * self.log_std_x, ones * self.log_std_y, ones * self.log_std_yaw), dim = -1)
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

class TorchVectorPPO(TorchModelV2, nn.Module): # updated
    """
    Attention Model agent
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        load_pretrained = model_config["custom_model_config"]['load_pretrained']
        freeze_for_RLtuning=model_config["custom_model_config"]['freeze_for_RLtuning']
        self.shared_feature_extractor = model_config["custom_model_config"]['shared_feature_extractor']
        # KL_pretrained = model_config["custom_model_config"]['KL_pretrained']
        cfg = model_config["custom_model_config"]['cfg']
        weights_scaling = [1.0, 1.0, 1.0]
        self.kl_div_weight = model_config["custom_model_config"]['kl_div_weight']
        self.log_std_acc = model_config["custom_model_config"]['log_std_acc']
        self.log_std_steer = model_config["custom_model_config"]['log_std_steer']
        self._num_predicted_frames = cfg["model_params"]["future_num_frames"]
        self._num_predicted_params = len(weights_scaling) #6

        self._actor_head = CustomVectorizedModel(
            history_num_frames_ego=cfg["model_params"]["history_num_frames_ego"],
            history_num_frames_agents=cfg["model_params"]["history_num_frames_agents"],
            global_head_dropout=cfg["model_params"]["global_head_dropout"],
            disable_other_agents=cfg["model_params"]["disable_other_agents"],
            disable_map=cfg["model_params"]["disable_map"],
            disable_lane_boundaries=cfg["model_params"]["disable_lane_boundaries"])

        if not self.shared_feature_extractor:
            self._critic_head = CustomVectorizedModel(
                history_num_frames_ego=cfg["model_params"]["history_num_frames_ego"],
                history_num_frames_agents=cfg["model_params"]["history_num_frames_agents"],
                global_head_dropout=cfg["model_params"]["global_head_dropout"],
                disable_other_agents=cfg["model_params"]["disable_other_agents"],
                disable_map=cfg["model_params"]["disable_map"],
                disable_lane_boundaries=cfg["model_params"]["disable_lane_boundaries"])
        model_path = f"{SRC_PATH}src/model/OL_HS.pt"
        # if KL_pretrained:
        #     # model_path = "./BPTT.pt"
        #     model_path = f'{SRC_PATH}src/model/OL_HS.pt'
        #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #     model = torch.load(model_path).to(device)
        #     # model = SAC.load("/home/pronton/rl/l5kit/examples/RL/gg colabs/logs/SAC_640000_steps.zip")
        #     self.pretrained_policy = model.eval()
        #     torch.set_grad_enabled(False)
        #     for  name, param in self.pretrained_policy.named_parameters():
        #         param.requires_grad = False

        if load_pretrained:
            pretrained_model = torch.load(model_path)
            weights = load_attention_model_except_last_3fc(pretrained_model)
            self._actor_head.load_state_dict(weights, strict=False)# ignore "weights_scaling", "xy_scale"
            if not self.shared_feature_extractor:
                self._critic_head.load_state_dict(weights, strict=False)
            
        if freeze_for_RLtuning:
            for  name, param in self._actor_head.named_parameters():
                param.requires_grad = False
            if not self.shared_feature_extractor:
                for  name, param in self._critic_head.named_parameters():
                    param.requires_grad = False

        d_model = 256
        self._actor_mlp = nn.Sequential(
                SlimFC(d_model, out_size=256, initializer= torch_normc_initializer(1.0), activation_fn=nn.ReLU) ,
                SlimFC(256, out_size=256, initializer= torch_normc_initializer(1.0), activation_fn=nn.ReLU) ,
                SlimFC(256, out_size=num_outputs, initializer= torch_normc_initializer(1.0)),
        )
        self._critic_mlp = nn.Sequential(
                SlimFC(d_model, out_size=256, initializer= torch_normc_initializer(0.01), activation_fn=nn.ReLU) ,
                SlimFC(256, out_size=256, initializer= torch_normc_initializer(0.01), activation_fn=nn.ReLU) ,
                SlimFC(256, out_size=1, initializer= torch_normc_initializer(0.01)),
        )
    def forward(self, input_dict, state, seq_lens):
        obs_transformed = input_dict['obs']
        for k in ['agent_trajectory_polyline', 'other_agents_polyline', 'lanes_mid', 'lanes', 'crosswalks']:
            # print(f'key: {k}, value shape: {obs_transformed[k].shape}, value type: {obs_transformed[k].dtype}')
            obs_transformed[k] = obs_transformed[k].float()
            # print(f'key: {k}, value shape: {obs_transformed[k].shape}, value type: {obs_transformed[k].dtype}')
            # print('---------------------------------------')
        # logging.debug('obs forward:'+ str(obs_transformed))
        actor_feature_value, attns = self._actor_head(obs_transformed)
        logits = self._actor_mlp(actor_feature_value)
        if not self.shared_feature_extractor:
            critic_feature_value, attns = self._critic_head(obs_transformed)
            value = self._critic_mlp(critic_feature_value)
        else:
            value = self._critic_mlp(actor_feature_value) # NOTE: since both actor and critic use same freeze pretrained net
        self._value = value.view(-1)
        # logging.debug(f'policy in custom model logits: {logits}, value: {self._value}')

        return logits, state
    def value_function(self):
        return self._value

class TorchVectorSharedSAC(SACTorchModel): # TODO: Delete
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
        self._feature_extractor = None
        self.kl_div_weight = model_config["custom_model_config"]['kl_div_weight']
        self.log_std_acc = model_config["custom_model_config"]['log_std_acc']
        self.log_std_steer = model_config["custom_model_config"]['log_std_steer']
        nn.Module.__init__(self)
        super(TorchVectorSharedSAC, self).__init__(
            obs_space, action_space, num_outputs, model_config, name, policy_model_config, q_model_config, twin_q, initial_alpha, target_entropy
        )
        
        self.cfg = model_config["custom_model_config"]['cfg']
        freezing = model_config["custom_model_config"]['freezing']
        weights_scaling = [1.0, 1.0, 1.0]
        self._num_predicted_params = len(weights_scaling) #6


        self._feature_extractor = CustomVectorizedModel(
            history_num_frames_ego=self.cfg["model_params"]["history_num_frames_ego"],
            history_num_frames_agents=self.cfg["model_params"]["history_num_frames_agents"],
            global_head_dropout=self.cfg["model_params"]["global_head_dropout"],
            disable_other_agents=self.cfg["model_params"]["disable_other_agents"],
            disable_map=self.cfg["model_params"]["disable_map"],
            disable_lane_boundaries=self.cfg["model_params"]["disable_lane_boundaries"])
        model_path = f'{SRC_PATH}/src/model/OL_HS.pt'
        pretrained_model = torch.load(model_path)
            
        weights = load_attention_model_except_last_3fc(pretrained_model)
        self._feature_extractor.load_state_dict(weights, strict=False)# ignore "weights_scaling", "xy_scale"

        if freezing:
            self._feature_extractor.eval()
            for  name, param in self._feature_extractor.named_parameters():
                param.requires_grad = False


    def build_policy_model(self, obs_space, num_outputs, policy_model_config, name):
        """Builds the policy model used by this SAC.

        Override this method in a sub-class of SACTFModel to implement your
        own policy net. Alternatively, simply set `custom_model` within the
        top level SAC `policy_model` config key to make this default
        implementation of `build_policy_model` use your custom policy network.

        Returns:
            TorchModelV2: The TorchModelV2 policy sub-model.
        """
        # policy_model_config = policy_model_config if policy_model_config else {}
        policy_model_config = {
            'kl_div_weight': self.kl_div_weight,
            'log_std_acc': self.log_std_acc,
            'log_std_steer': self.log_std_steer,
        }
        return TorchMLPPolicyNet(obs_space, self.action_space, num_outputs, policy_model_config, name)

    def build_q_model(self, obs_space, action_space, num_outputs, q_model_config, name):
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

        # model = ModelCatalog.get_model_v2(
        #     input_space,
        #     action_space,
        #     num_outputs,
        #     q_model_config,
        #     framework="torch",
        #     name=name,
        # )
        # return model
        q_model_config = {} if q_model_config == None else q_model_config
        return TorchMLPQNet(input_space, self.action_space, num_outputs, q_model_config, name)

        
    def forward(self, input_dict, state, seq_lens):
        obs= input_dict['obs']
        for k in ['agent_trajectory_polyline', 'other_agents_polyline', 'lanes_mid', 'lanes', 'crosswalks']:
            obs[k] = obs[k].float()
        # logging.debug('obs forward:'+ str(obs_transformed))
        if self._feature_extractor:
            feature_value, attns = self._feature_extractor(obs)
        
        return feature_value, state

class TorchMLPPolicyNet(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        d_attention_outs = 256
        self.log_std_acc = model_config['log_std_acc']
        self.log_std_steer = model_config['log_std_steer']
        self.kl_div_weight = model_config['kl_div_weight']
        self.policy_mlp = nn.Sequential(
            SlimFC(d_attention_outs, 256, initializer= torch_normc_initializer(1.0), activation_fn=nn.ReLU) ,
            SlimFC(256, 256, initializer= torch_normc_initializer(1.0), activation_fn=nn.ReLU) ,
            SlimFC(256, num_outputs, initializer= torch_normc_initializer(1.0)),
        )

    def forward(self, input_dict, state, seq_lens):
        # logging.debug(f'input policy: {input_dict["obs"]}')
        out = self.policy_mlp(input_dict['obs'])
        return out, state

class TorchMLPQNet(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        # d_attention_outs = 256
        # self.policy_mlp = nn.Sequential(
        #     SlimFC(d_attention_outs, 256, initializer= torch_normc_initializer(1.0), activation_fn=nn.ReLU) ,
        #     SlimFC(256, 256, initializer= torch_normc_initializer(1.0), activation_fn=nn.ReLU) ,
        #     SlimFC(256, num_outputs, initializer= torch_normc_initializer(1.0)),
        # )

        self.action_dim = np.product(action_space.shape)
        # action_outs = 2 * self.action_dim
        q_outs = 1
        d_attention_outs = 256
        self.action_mlp = nn.Sequential(
            SlimFC(self.action_dim, 256, activation_fn=nn.Tanh),
            SlimFC(256,256, activation_fn=nn.Tanh),
        )
        # self.q_net = MLP(d_model + d_model, d_model, output_dim = q_outs , num_layers=1)
        self.q_mlp = nn.Sequential(
            SlimFC(256 + d_attention_outs, 256, initializer=normc_initializer(0.01), activation_fn=nn.ReLU),
            SlimFC(256, q_outs, initializer=normc_initializer(0.01)),
        )

    def forward(self, input_dict, state, seq_lens):
        # logging.debug(f'input q: {input_dict["obs"]}')
        # logging.debug(f"q obs input_dict: {input_dict['obs'][0]}")
        # logging.debug(f"q action input_dict: {input_dict['obs'][1]}")
        model_out, action = input_dict['obs']
        # logging.debug(f'q net in custom model - obs[type]: {obs["type"]} |  action: {action}')
        # logging.debug(f'q net in custom model - state features mean: {state_features}')
        action_features = self.action_mlp(action)
        # logging.debug(f'q net in custom model - action features mean: {action_features}')
        #logging.debug(f"features dim: {state_features.shape}")
        #logging.debug(f"action dim: {action_features.shape}")
        q = self.q_mlp(torch.cat((model_out, action_features), dim=1))
        # logging.debug(f'q value : {q}')
        return q, state


############### Shared pretrained layer for SAC
shared_pretrained_head = None
features = None

class TorchVectorPolicyNet(TorchModelV2, nn.Module):
    """
    RasterNet Model agent
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        load_pretrained = model_config["custom_model_config"]['load_pretrained']
        freeze_for_RLtuning = model_config["custom_model_config"]['freeze_for_RLtuning']
        cfg = model_config["custom_model_config"]['cfg'] # TODO: Pass necessary params, not cfg
        self.share_feature_extractor = model_config["custom_model_config"]['share_feature_extractor']
        d_attention_outs = 256

        self.kl_div_weight = model_config["custom_model_config"]['kl_div_weight']
        self.log_std_acc = model_config["custom_model_config"]['log_std_acc']
        self.log_std_steer = model_config["custom_model_config"]['log_std_steer']
        self.m_tau = model_config["custom_model_config"]['m_tau']
        self.m_alpha = model_config["custom_model_config"]['m_alpha']
        self.m_l0 = model_config["custom_model_config"]['m_l0']
        self.m_entropy = model_config["custom_model_config"]['m_entropy']
        self.m_kl = model_config["custom_model_config"]['m_kl']
        self.use_entropy_kl_params = model_config["custom_model_config"]['use_entropy_kl_params']
        self.sac_entropy_equal_m_entropy = model_config["custom_model_config"]['sac_entropy_equal_m_entropy']
        
        # self.log_std_acc = -1
        # self.log_std_steer = -1

        # weights_scaling = [1.0, 1.0, 1.0]
        # self._num_predicted_frames = cfg["model_params"]["future_num_frames"]
        # self._num_predicted_params = len(weights_scaling) #6

        model_path = f"{SRC_PATH}src/model/OL_HS.pt"
        # if KL_pretrained:
        #     self.pretrained_policy = VectorizedModel(
        #         history_num_frames_ego=cfg["model_params"]["history_num_frames_ego"],
        #         history_num_frames_agents=cfg["model_params"]["history_num_frames_agents"],
        #         num_targets=self._num_predicted_params * self._num_predicted_frames, # N (X,Y,Yaw) 36
        #         weights_scaling=weights_scaling, # 3
        #         criterion=nn.L1Loss(reduction="none"),
        #         global_head_dropout=cfg["model_params"]["global_head_dropout"],
        #         disable_other_agents=cfg["model_params"]["disable_other_agents"],
        #         disable_map=cfg["model_params"]["disable_map"],
        #         disable_lane_boundaries=cfg["model_params"]["disable_lane_boundaries"])
        #     self.pretrained_policy.load_state_dict(torch.load(model_path).state_dict())
        #     for  name, param in self.pretrained_policy.named_parameters():
        #         param.requires_grad = False


        if self.share_feature_extractor:
            global shared_pretrained_head
            shared_pretrained_head= CustomVectorizedModel(
                        history_num_frames_ego=cfg["model_params"]["history_num_frames_ego"],
                        history_num_frames_agents=cfg["model_params"]["history_num_frames_agents"],
                        global_head_dropout=cfg["model_params"]["global_head_dropout"],
                        disable_other_agents=cfg["model_params"]["disable_other_agents"],
                        disable_map=cfg["model_params"]["disable_map"],
                        disable_lane_boundaries=cfg["model_params"]["disable_lane_boundaries"])
            self.policy_head = shared_pretrained_head

        else:
            self.policy_head = CustomVectorizedModel(
                history_num_frames_ego=cfg["model_params"]["history_num_frames_ego"],
                history_num_frames_agents=cfg["model_params"]["history_num_frames_agents"],
                global_head_dropout=cfg["model_params"]["global_head_dropout"],
                disable_other_agents=cfg["model_params"]["disable_other_agents"],
                disable_map=cfg["model_params"]["disable_map"],
                disable_lane_boundaries=cfg["model_params"]["disable_lane_boundaries"])

        if load_pretrained:
            pretrained_model = torch.load(model_path)
            if freeze_for_RLtuning:
                pretrained_model = pretrained_model.eval()
                
            weights = load_attention_model_except_last_3fc(pretrained_model)
            self.policy_head.load_state_dict(weights, strict=False)# ignore "weights_scaling", "xy_scale"

            if freeze_for_RLtuning:
                for  name, param in self.policy_head.named_parameters():
                    param.requires_grad = False
            

        self.policy_mlp = nn.Sequential(
            SlimFC(d_attention_outs, 256, initializer= torch_normc_initializer(1.0), activation_fn=nn.ReLU) ,
            SlimFC(256, 256, initializer= torch_normc_initializer(1.0), activation_fn=nn.ReLU) ,
            SlimFC(256, num_outputs, initializer= torch_normc_initializer(1.0)),
        )

    def forward(self, input_dict, state, seq_lens):
        # logging.debug(f'policy net in custom model - obs[type]: {input_dict["obs"]["type"]}')
        # logging.debug(f'policy input: {input_dict["action"]}')
        # global features
        obs = input_dict['obs']
        for k in ['agent_trajectory_polyline', 'other_agents_polyline', 'lanes_mid', 'lanes', 'crosswalks']:
            obs[k] = obs[k].float()
        # logging.debug(f'policy input types: {[v.dtype for v in input_dict["obs"].values()]}')
        features, attns = self.policy_head(obs)
        # logging.debug(f'---------------------------->\nshared features in policy: {features}')
        # logging.debug(f'policy net in custom model - features mean: {features.mean()}')
        logits = self.policy_mlp(features)
        # logging.debug(f'policy net in custom model - logits: {logits}')
        return logits, state

class TorchVectorQNet(TorchModelV2, nn.Module):
    """
    RasterNet Model agent
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        load_pretrained = model_config["custom_model_config"]['load_pretrained']
        freeze_for_RLtuning = model_config["custom_model_config"]['freeze_for_RLtuning']
        cfg = model_config["custom_model_config"]['cfg'] # TODO: Pass necessary params, not cfg
        self.share_feature_extractor = model_config["custom_model_config"]['share_feature_extractor']

        if self.share_feature_extractor:
            global shared_pretrained_head
            self.q_head = shared_pretrained_head
        else:
            self.q_head = CustomVectorizedModel(
                history_num_frames_ego=cfg["model_params"]["history_num_frames_ego"],
                history_num_frames_agents=cfg["model_params"]["history_num_frames_agents"],
                global_head_dropout=cfg["model_params"]["global_head_dropout"],
                disable_other_agents=cfg["model_params"]["disable_other_agents"],
                disable_map=cfg["model_params"]["disable_map"],
                disable_lane_boundaries=cfg["model_params"]["disable_lane_boundaries"])

        if load_pretrained:
            model_path = f"{SRC_PATH}src/model/OL_HS.pt"
            pretrained_model = torch.load(model_path)
            weights = load_attention_model_except_last_3fc(pretrained_model)
            self.q_head.load_state_dict(weights, strict=False)# ignore "weights_scaling", "xy_scale"
            # self._critic_head.load_state_dict(weights, strict=False)
            
        if freeze_for_RLtuning:
            self.q_head = self.q_head.eval() # turn off dropout, batchnorm, ...
            for  name, param in self.q_head.named_parameters():
                param.requires_grad = False
        
        self.action_dim = np.product(action_space.shape)
        # action_outs = 2 * self.action_dim
        q_outs = 1
        d_attention_outs = 256
        self.action_mlp = nn.Sequential(
            SlimFC(self.action_dim, 256, activation_fn=nn.Tanh),
            SlimFC(256,256, activation_fn=nn.Tanh),
        )
        # self.q_net = MLP(d_model + d_model, d_model, output_dim = q_outs , num_layers=1)
        self.q_mlp = nn.Sequential(
            SlimFC(256 + d_attention_outs, 256, initializer=normc_initializer(0.01), activation_fn=nn.ReLU),
            SlimFC(256, q_outs, initializer=normc_initializer(0.01)),
        )

    def forward(self, input_dict, state, seq_lens):
        # logging.debug(f"q obs input_dict: {input_dict['obs'][0]}")
        # logging.debug(f"q action input_dict: {input_dict['obs'][1]}")
        obs, action = input_dict['obs']
        for k in ['agent_trajectory_polyline', 'other_agents_polyline', 'lanes_mid', 'lanes', 'crosswalks']:
            obs[k] = obs[k].float()
        # logging.debug(f'q net in custom model - obs[type]: {obs["type"]} |  action: {action}')
        if self.share_feature_extractor:
            assert features != None, 'features should be computed in policy net'
            # logging.debug(f'shared features in Q: {features}')
            state_features = features
        else:
            state_features, attns = self.q_head(obs)
        # logging.debug(f'q net in custom model - state features mean: {state_features}')
        action_features = self.action_mlp(action)
        # logging.debug(f'q net in custom model - action features mean: {action_features}')
        #logging.debug(f"features dim: {state_features.shape}")
        #logging.debug(f"action dim: {action_features.shape}")
        q = self.q_mlp(torch.cat((state_features, action_features), dim=1))
        # logging.debug(f'q value : {q}')
        return q, state

class TorchVectorQNet2(TorchModelV2):
    """A simple, q-value-from-cont-action model (for e.g. SAC type algos)."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        nn.Module.__init__(self)
        # Pass num_outputs=None into super constructor (so that no action/
        # logits output layer is built).
        # Alternatively, you can pass in num_outputs=[last layer size of
        # config[model][fcnet_hiddens]] AND set no_last_linear=True, but
        # this seems more tedious as you will have to explain users of this
        # class that num_outputs is NOT the size of your Q-output layer.
        super(TorchVectorQNet2, self).__init__(
            obs_space, action_space, None, model_config, name
        )
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


        # Now: self.num_outputs contains the last layer's size, which
        # we can use to construct the single q-value computing head.

        # Nest an RLlib FullyConnectedNetwork (torch or tf) into this one here
        # to be used for Q-value calculation.
        # Use the current value of self.num_outputs, which is the wrapped
        # model's output layer size.
        combined_space = Box(-1.0, 1.0, (self.num_outputs + action_space.shape[0],))

        self.q_head = TorchFullyConnectedNetwork(
            combined_space, action_space, 1, model_config, "q_head"
        )

        # Missing here: Probably still have to provide action output layer
        # and value layer and make sure self.num_outputs is correctly set.

    def get_single_q_value(self, underlying_output, action):
        # Calculate the q-value after concating the underlying output with
        # the given action.
        input_ = torch.cat([underlying_output, action], dim=-1)
        # Construct a simple input_dict (needed for self.q_head as it's an
        # RLlib ModelV2).
        input_dict = {"obs": input_}
        # Ignore state outputs.
        q_values, _ = self.q_head(input_dict)
        return q_values

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
        obs = obs.float()
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
        obs = input_dict['obs'].float()
        # logging.debug(f'policy input types: {[v.dtype for v in input_dict["obs"].values()]}')
        features = self.state_feature(obs)
        return self.fc(features), state


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
