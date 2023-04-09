from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch.nn as nn
import torch
from torch.nn import functional as F
from l5kit.planning.vectorized.open_loop_model import VectorizedModel, CustomVectorizedModel
from l5kit.planning.rasterized.model import RasterizedPlanningModelFeature
from torchvision.models.resnet import resnet18, resnet50
import os
import numpy as np
#os.environ['CUDA_VISIBLE_DEVICES']= '0'

import logging
logging.basicConfig(filename='/workspace/source/src/log/info.log', level=logging.DEBUG, filemode='w')
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
_actor_head = RasterizedPlanningModelFeature(
            model_arch="resnet50",
            num_input_channels=5,
            num_targets=3 ,#3 * cfg["model_params"]["future_num_frames"],  # X, Y, Yaw * number of future states 3 x 12
            weights_scaling=[1., 1., 1.],
            criterion=nn.MSELoss(reduction="none"),)
 
model_path = "/workspace/source/src/model/planning_model_20201208.pt"
pretrained = torch.load(model_path)
weights_subset = {}
for key, value in pretrained.state_dict().items():
    if key == 'model.fc.weight':
        weights_subset[key] = value[:3,:]
    elif key == 'model.fc.bias':
        weights_subset[key] = value[:3]
    else:
        weights_subset[key] = value
_actor_head.load_state_dict(weights_subset)