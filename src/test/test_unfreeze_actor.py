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
from src.customModel.customModel import TorchRasterNet, TorchRasterNet2, TorchGNCNN_separated

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
ray.init(num_cpus=5, ignore_reinit_error=True, log_to_driver=False, object_store_memory = 5*10**9, local_mode= False)


from l5kit.configs import load_config_data

env_config_path = SRC_PATH +'src/configs/gym_rasterizer_config.yaml'
env_config_path = SRC_PATH +'src/configs/gym_config.yaml'
cfg = load_config_data(env_config_path)

ModelCatalog.register_custom_model( "TorchSeparatedRasterNet", TorchRasterNet)
ModelCatalog.register_custom_model( "TorchSeparatedRasterNet2", TorchRasterNet2)
ModelCatalog.register_custom_model( "TorchSeparatedPPO", TorchGNCNN_separated)

# os.environ['CUDA_VISIBLE_DEVICES']= '1'
n_channels = (cfg['model_params']['history_num_frames'] + 1)* 2 + 3
print('num channels:', n_channels)
from ray import tune
from src.customEnv.wrapper import L5EnvRasterizerTorch, L5EnvWrapperTorch
train_eps_length = 32
train_sim_cfg = SimulationConfigGym()
train_sim_cfg.num_simulation_steps = train_eps_length + 1


env_kwargs = {'env_config_path': env_config_path, 'use_kinematic': False, 'sim_cfg': train_sim_cfg, 'rescale_action': True}
# non_kin_rescale = L5Env(**env_kwargs).non_kin_rescale
# tune.register_env("L5-CLE-V1", lambda config: L5EnvRasterizerTorch(env = L5Env(**env_kwargs), \
#                                                            raster_size= cfg['raster_params']['raster_size'][0], \
#                                                            n_channels = n_channels))

tune.register_env("L5-CLE-V1", lambda config: L5EnvRasterizerTorch(env = L5Env(**env_kwargs), \
                                                           raster_size= cfg['raster_params']['raster_size'][0], \
                                                           n_channels = n_channels))
#################### Train ####################

import ray
from ray import air, tune
train_envs = 4

hcmTz = pytz.timezone("Asia/Ho_Chi_Minh") 
date = datetime.datetime.now(hcmTz).strftime("%d-%m-%Y_%H-%M-%S")
ray_result_logdir = '/home/pronton/ray_results/debug_restnet50_tanh_nonfreeze_actorNet' + date
# ray_result_logdir = '/home/pronton/ray_results/debug_simple_cnn_nonfreeze_actorNet' + date

# lr = 3e-3
lr_start = 3e-5
lr_end = 3e-6

config_param_space = {
    "env": "L5-CLE-V1",
    "framework": "torch",
    "num_gpus": 1,
    "num_workers": 4,
    # 'grad_clip': 5.0,
    # 'kl_coeff': 0.01,
    "num_envs_per_worker": train_envs, #8 * 32
    'disable_env_checking':True,
    # "model": {
    #         "custom_model": "TorchSeparatedPPO",
    #         "custom_model_config": {
    #                 'future_num_frames':cfg["model_params"]["future_num_frames"],
    #                 'freeze_actor': False, # only true when actor_net is restnet50 (loaded from resnet50 pretrained)
    #                 'n_channels':n_channels,
    #                 'actor_net': 'resnet50', # resnet50
    #                 'critic_net': 'resnet50', # resnet50
    #                 },
    #         },
    "model": {
            "custom_model": "TorchSeparatedRasterNet2",
            "custom_model_config": {
                    'future_num_frames':cfg["model_params"]["future_num_frames"],
                    'freeze_actor': False, # only true when actor_net is restnet50 (loaded from resnet50 pretrained)
                    # 'non_kin_rescale': non_kin_rescale,
                    'n_channels':n_channels,
                    'actor_net': 'resnet50', # resnet50
                    'critic_net': 'resnet50', # resnet50
                    },
            },
    '_disable_preprocessor_api': True,
    "eager_tracing": True,
    "restart_failed_sub_environments": True,
    # "lr": lr,
    'seed': 42,
    'use_critic': True,
    "lr_schedule": [
         [1e6, lr_start],
         [2e6, lr_end],
     ],
    'train_batch_size': 2048, #8000,# 8000 
    'sgd_minibatch_size': 64, #2048, #2048
    'num_sgd_iter': 16,#10,#16,
    'seed': 42,
    'batch_mode': 'truncate_episodes',
    "rollout_fragment_length": 32,
    'gamma': 0.8,    
}


from src.customModel.customPPOTrainer import KLPPO
# ray.tune.run(KLPPO, config=config_param_space, restore=path_to_trained_agent_checkpoint)
# checkpoint_path = '/workspace/datasets/ray_results/08-04-2023_14-17-36(RasterPPO_vf~2)/KLPPO_2023-04-08_07-17-36/KLPPO_L5-CLE-V1_70625_00000_0_2023-04-08_07-17-37/checkpoint_000030'
# model = KLPPO(config=config_param_space, env='L5-CLE-V1')
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
# model = PPO(config=config_param_space, env='L5-CLE-V1')
# from ray.tune.logger import pretty_print
# for i in range(10000):
#     print('alo')
#     result = model.train()
#     print(pretty_print(result))