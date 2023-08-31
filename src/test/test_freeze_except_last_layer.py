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
ray.init(num_cpus=9, ignore_reinit_error=True, log_to_driver=False, object_store_memory = 5*10**9, local_mode= False)


from l5kit.configs import load_config_data

env_config_path = SRC_PATH +'src/configs/gym_rasterizer_config.yaml'
env_config_path = SRC_PATH +'src/configs/gym_rasterizer_config_7x112x112.yaml'
env_config_path = SRC_PATH +'src/configs/gym_config84.yaml'
# env_config_path = SRC_PATH +'src/configs/gym_config.yaml'
cfg = load_config_data(env_config_path)

from src.customModel.customModel import TorchRasterNet3, TorchRasterNetMixedActor, TorchRasterNetMixedCritic, TorchRasterNetCustomVisionNetActor
from src.customModel.customModelTest import MyVisionNetwork
ModelCatalog.register_custom_model( "TorchSeparatedRasterNet3", TorchRasterNet3)
ModelCatalog.register_custom_model( "TorchSeparatedRasterNetMixedCritic", TorchRasterNetMixedCritic)
ModelCatalog.register_custom_model( "TorchSeparatedRasterNetMixedActor", TorchRasterNetMixedActor)
ModelCatalog.register_custom_model( "TorchRasterNetCustomVisionNetActor", TorchRasterNetCustomVisionNetActor)
ModelCatalog.register_custom_model( "torchVisionNet", MyVisionNetwork)

# os.environ['CUDA_VISIBLE_DEVICES']= '1'
n_channels = (cfg['model_params']['history_num_frames'] + 1)* 2 + 3
print('num channels:', n_channels)
from ray import tune
from src.customEnv.wrapper import L5EnvRasterizerTorch, L5EnvWrapperTorch, L5EnvWrapper
train_eps_length = 32
train_sim_cfg = SimulationConfigGym()
train_sim_cfg.num_simulation_steps = train_eps_length + 1


env_kwargs = {'env_config_path': env_config_path, 'use_kinematic': True, 'sim_cfg': train_sim_cfg}
# non_kin_rescale = L5Env(**env_kwargs).non_kin_rescale
# tune.register_env("L5-CLE-V1", lambda config: L5EnvRasterizerTorch(env = L5Env(**env_kwargs), \
#                                                            raster_size= cfg['raster_params']['raster_size'][0], \
#                                                            n_channels = n_channels))

# tune.register_env("L5-CLE-V1", lambda config: L5EnvRasterizerTorch(env = L5Env(**env_kwargs), \
#                                                            raster_size= cfg['raster_params']['raster_size'][0], \
#                                                            n_channels = n_channels))
tune.register_env("L5-CLE-V1", lambda config: L5EnvWrapperTorch(env = L5Env(**env_kwargs), \
                                                           raster_size= cfg['raster_params']['raster_size'][0], \
                                                           n_channels = n_channels))
#################### Train ####################

import ray
from ray import air, tune
train_envs = 4

hcmTz = pytz.timezone("Asia/Ho_Chi_Minh") 
date = datetime.datetime.now(hcmTz).strftime("%d-%m-%Y_%H-%M-%S")
# ray_result_logdir = '/home/pronton/ray_results/just_debug_simple_cnn_84_kin_separated_myVisionNet_lowlr_postrelu_fcnet_hiddens' + date
# ray_result_logdir = '/home/pronton/ray_results/just_debug_simple_cnn_84_kin_separated_mixedModel_lowlr_256dim' + date
ray_result_logdir = '/home/pronton/ray_results/just_debug_simple_cnn_84_kin_separated_mixedVisionNetActor' + date

train_envs = 4
# if 'gym_rasterizer_config' in env_config_path:
lr = 3e-4
lr_start = 3e-5
lr_end = 3e-6
lr_time = int(4e6)
# else:
#     lr = 3e-3
#     lr_start = 3e-4
#     lr_end = 3e-5
#     lr_time = int(4e6)

config_param_space = {
    "env": "L5-CLE-V1",
    "framework": "torch",
    "num_gpus": 1,
    "num_workers": 8,
    
    # 'grad_clip': 5.0,
    # 'kl_coeff': 0.01,
    "num_envs_per_worker": train_envs, #8 * 32
    'disable_env_checking':True,
    "model": {
            "custom_model": "TorchRasterNetCustomVisionNetActor", # similar to visionnet
            # "post_fcnet_hiddens": [256, 256],
            # "post_fcnet_activation": "tanh",
            "vf_share_layers": False,   
            "custom_model_config": {
                    # 'post_fcnet_activation': 'relu',


                    'n_channels':n_channels,
                    'actor_net': 'simple_cnn', # resnet50
                    'critic_net': 'simple_cnn', # resnet50
                    'future_num_frames':cfg["model_params"]["future_num_frames"],
                    'freeze_actor': False, # only true when actor_net is restnet50 (loaded from resnet50 pretrained)
                    # 'non_kin_rescale': non_kin_rescale,
                    'shared_feature_extractor' : False,
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
    'sgd_minibatch_size': 256, #2048, #2048
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
