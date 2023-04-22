import time

SRC_PATH = '/home/pronton/rlhf-car/'
# from src.constant import SRC_PATH
start = time.time()
import os
os.environ["L5KIT_DATA_FOLDER"] = '/workspace/datasets'
# os.environ['CUDA_VISIBLE_DEVICES']= '0'
os.environ['RAY_memory_monitor_refresh_ms']='0'
# os.environ["TUNE_RESULT_DIR"] =  '/DATA/l5kit/rllib_tb_logs'
import gym
from l5kit.configs import load_config_data
from l5kit.environment.envs.l5_env import SimulationConfigGym, GymStepOutput, L5Env
from l5kit.visualization.visualizer.zarr_utils import episode_out_to_visualizer_scene_gym_cle
from l5kit.visualization.visualizer.visualizer import visualize
from bokeh.io import output_notebook, show
from l5kit.environment.gym_metric_set import L2DisplacementYawMetricSet, CLEMetricSet
from prettytable import PrettyTable
import datetime
from src.customModel.customModel import TorchRasterNet3

from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch.nn as nn
import numpy as np
import gym
from typing import Dict
import numpy as np
import ray
import pytz
from ray import tune
import torch
from l5kit.planning.rasterized.model import RasterizedPlanningModelFeature
ray.init(num_cpus=32, ignore_reinit_error=True, log_to_driver=False, object_store_memory = 5*10**9, local_mode= False)


from l5kit.configs import load_config_data

env_config_path = SRC_PATH +'src/configs/gym_rasterizer_config.yaml'
env_config_path = SRC_PATH +'src/configs/gym_rasterizer_config_7x112x112.yaml'
# env_config_path = SRC_PATH +'src/configs/gym_config.yaml'
self.cfg = load_config_data(env_config_path)

ModelCatalog.register_custom_model( "TorchSeparatedRasterNet3", TorchRasterNet3)

# os.environ['CUDA_VISIBLE_DEVICES']= '1'
n_channels = (self.cfg['model_params']['history_num_frames'] + 1)* 2 + 3
print('num channels:', n_channels)
from ray import tune
from src.customEnv.wrapper import L5EnvRasterizerTorch, L5EnvWrapperTorch, L5EnvWrapper
train_eps_length = 32
train_sim_cfg = SimulationConfigGym()
train_sim_cfg.num_simulation_steps = train_eps_length + 1


env_kwargs = {'env_config_path': env_config_path, 'use_kinematic': False, 'sim_cfg': train_sim_cfg, 'rescale_action': True}
# non_kin_rescale = L5Env(**env_kwargs).non_kin_rescale
# tune.register_env("L5-CLE-V1", lambda config: L5EnvRasterizerTorch(env = L5Env(**env_kwargs), \
#                                                            raster_size= cfg['raster_params']['raster_size'][0], \
#                                                            n_channels = n_channels))

tune.register_env("L5-CLE-V1", lambda config: L5EnvRasterizerTorch(env = L5Env(**env_kwargs), \
                                                           raster_size= self.cfg['raster_params']['raster_size'][0], \
                                                           n_channels = n_channels))
# tune.register_env("L5-CLE-V1", lambda config: L5EnvWrapper(env = L5Env(**env_kwargs), \
#                                                            raster_size= cfg['raster_params']['raster_size'][0], \
#                                                            n_channels = n_channels))
#################### Train ####################

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
            # nn.Linear(in_features=1568, out_features=self._feature_dim),
            nn.Linear(in_features=6272, out_features=self._feature_dim),
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
ModelCatalog.register_custom_model("GN_CNN_torch_model", TorchGNCNN)

import ray
from ray import air, tune
train_envs = 4

hcmTz = pytz.timezone("Asia/Ho_Chi_Minh") 
date = datetime.datetime.now(hcmTz).strftime("%d-%m-%Y_%H-%M-%S")
ray_result_logdir = '/home/pronton/ray_results/debug_simple_cnn_7x112x112_nonkin_shared_fc_RasterNet3' + date
# ray_result_logdir = '/home/pronton/ray_results/debug_simple_cnn_nonfreeze_actorNet' + date

train_envs = 4
if 'gym_rasterizer_config' in env_config_path:
    lr = 3e-4
    lr_start = 3e-5
    lr_end = 3e-6
    lr_time = int(4e6)
else:
    lr = 3e-3
    lr_start = 3e-4
    lr_end = 3e-5
    lr_time = int(4e6)
config_param_space = {
    "env": "L5-CLE-V1",
    "framework": "torch",
    "num_gpus": 1,
    "num_workers": 31,
    # 'grad_clip': 5.0,
    # 'kl_coeff': 0.01,
    "num_envs_per_worker": train_envs, #8 * 32
    'disable_env_checking':True,
    # "model": {
    #     "custom_model": "GN_CNN_torch_model",
    #     "custom_model_config": {'feature_dim':128},
    # },
    "model": {
            "custom_model": "TorchSeparatedRasterNet3",
            "custom_model_config": {
                    'n_channels':n_channels,
                    'actor_net': 'simple_cnn', # resnet50
                    'critic_net': 'simple_cnn', # resnet50
                    'future_num_frames':self.cfg["model_params"]["future_num_frames"],
                    'freeze_actor': False, # only true when actor_net is restnet50 (loaded from resnet50 pretrained)
                    # 'non_kin_rescale': non_kin_rescale,
                    'shared_feature_extractor' : True,
                    },
            },
    '_disable_preprocessor_api': True,
    "eager_tracing": True,
    "restart_failed_sub_environments": True,
    "lr": lr,
    'seed': 42,
    'use_critic': True,
    "lr_schedule": [
         [1e6, lr_start],
         [2e6, lr_end],
     ],
    'train_batch_size': 1024, #8000,# 8000 
    'sgd_minibatch_size': 64, #2048, #2048
    'num_sgd_iter': 10,#10,#16,
    'seed': 42,
    'batch_mode': 'truncate_episodes',
    # "rollout_fragment_length": 32,
    'gamma': 0.8,    
}

# train_envs = 4
# lr = 3e-3
# lr_start = 3e-4
# lr_end = 3e-5
# lr_time = int(4e6)
# config_param_space = {
#     "env": "L5-CLE-V1",
#     "framework": "torch",
#     "num_gpus": 1,
#     "num_workers": 8,
#     "num_envs_per_worker": train_envs,
#     "model": {
#         "custom_model": "GN_CNN_torch_model",
#         "custom_model_config": {'feature_dim':128},
#     },
    
#     # 'model' : {
#     #         # "dim": 84,
#     #         # "conv_filters" : [[64, [7,7], 3], [32, [11,11], 3], [32, [11,11], 3]],
#     #         # "conv_activation": "relu",
#     #         "post_fcnet_hiddens": [256],
#     #         "post_fcnet_activation": "relu",
#     #         "vf_share_layers": False,   
#     # },
    
#     '_disable_preprocessor_api': True,
#      "eager_tracing": True,
#      "restart_failed_sub_environments": True,
#     "lr": lr,
#     'seed': 42,
#     "lr_schedule": [
#         [7e5, lr_start],
#         [2e6, lr_end],
#     ],
#     'train_batch_size': 1024, # 8000 
#     'sgd_minibatch_size': 512, #2048
#     'num_sgd_iter': 10,#16,
#     'batch_mode': 'truncate_episodes',
#     # "rollout_fragment_length": 32,
#     'gamma': 0.8,    
# }

# ray.tune.run(KLPPO, config=config_param_space, restore=path_to_trained_agent_checkpoint)
# checkpoint_path = '/workspace/datasets/ray_results/08-04-2023_14-17-36(RasterPPO_vf~2)/KLPPO_2023-04-08_07-17-36/KLPPO_L5-CLE-V1_70625_00000_0_2023-04-08_07-17-37/checkpoint_000030'
# model.restore(checkpoint_path)
result_grid = tune.Tuner( 
    'PPO',
    run_config=air.RunConfig(
        stop={"episode_reward_mean": 0, 'timesteps_total': int(6e6)},
        local_dir=ray_result_logdir,
        checkpoint_config=air.CheckpointConfig(num_to_keep=2, 
                                            checkpoint_frequency = 10, 
                                            checkpoint_score_attribute = 'episode_reward_mean'),
        # callbacks=[WandbLoggerCallback(project="l5kit2", save_code = True, save_checkpoints = False),],
        ),
    param_space=config_param_space).fit()

# from ray.rllib.algorithms.ppo import PPO
# from src.customModel.customPPOTrainer import KLPPO
# model = PPO(config=config_param_space, env='L5-CLE-V1')
# model = KLPPO(config=config_param_space, env='L5-CLE-V1')
# from ray.tune.logger import pretty_print
# for i in range(10000):
#     print('alo')
#     result = model.train()
#     print(pretty_print(result))
