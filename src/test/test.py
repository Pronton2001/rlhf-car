import os
os.environ["L5KIT_DATA_FOLDER"] = '/workspace/datasets'
os.environ['CUDA_VISIBLE_DEVICES']= '1'
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
import ray
import pytz
from ray import tune

ray.init(object_store_memory=5e9, num_cpus=5, log_to_driver=False, local_mode=False)

from l5kit.configs import load_config_data

# get environment config
# env_config_path = '/workspace/source/configs/gym_config_history3.yaml'
env_config_path = '/workspace/source/src/configs/gym_config84.yaml'
# env_config_path = '/workspace/source/configs/gym_config.yaml'
# env_config_path = '/workspace/source/src/configs/gym_vectorizer_config.yaml'
cfg = load_config_data(env_config_path)


#################### Define Training and Evaluation Environments ####################
n_channels = (cfg['model_params']['future_num_frames'] + 1)* 2 + 3
print(cfg['model_params']['future_num_frames'], cfg['model_params']['history_num_frames'], n_channels)
from ray import tune
from src.customEnv.wrapper import L5EnvWrapper, L5EnvWrapperTorch, L5EnvWrapperTF
train_eps_length = 32
train_sim_cfg = SimulationConfigGym()
train_sim_cfg.num_simulation_steps = train_eps_length + 1


# Register , how your env should be constructed (always with 5, or you can take values from the `config` EnvContext object):
env_kwargs = {'env_config_path': env_config_path, 'use_kinematic': True, 'sim_cfg': train_sim_cfg}

tune.register_env("L5-CLE-V0", lambda config: L5Env(**env_kwargs))
# tune.register_env("L5-CLE-V1", lambda config: L5EnvWrapper(env = L5Env(**env_kwargs), \
#                                                            raster_size= cfg['raster_params']['raster_size'][0], \
#                                                            n_channels = n_channels))
tune.register_env("L5-CLE-V1", lambda config: L5EnvWrapperTorch(env = L5Env(**env_kwargs), \
                                                           raster_size= cfg['raster_params']['raster_size'][0], \
                                                           n_channels = n_channels))

#################### Wandb ####################


# from ray.air.integrations.wandb import setup_wandb
# from ray.air.integrations.wandb import WandbLoggerCallback
# os.environ['WANDB_NOTEBOOK_NAME'] = '/workspace/source/rllib_ppo.py'
# os.environ["WANDB_API_KEY"] = '083592c84134c040dcca598c644c348d32540a08'

# import wandb
# wandb.init(project="l5kit2", reinit = True,
#     save_code=True,  # optional
#     group = 'PPO',
#     name = 'debug',)

#################### Train ####################
import ray
from ray import air, tune
hcmTz = pytz.timezone("Asia/Ho_Chi_Minh") 
date = datetime.datetime.now(hcmTz).strftime("%d-%m-%Y_%H-%M-%S")
ray_result_logdir = '/workspace/datasets/ray_results/debug' + date

train_envs = 4
lr = 3e-3
lr_start = 3e-4
lr_end = 3e-5
lr_time = int(4e6)

config_param_space = {
    "env": "L5-CLE-V1",
    "framework": "torch",
    "num_gpus": 1,
    "num_workers": 3,
    "num_envs_per_worker": train_envs,
    # "model": {
    #     "custom_model": "GN_CNN_torch_model",
    #     "custom_model_config": {'feature_dim':128},
    # },
    
    'model' : {
            # "dim": 84,
            # "conv_filters" : [[64, [7,7], 3], [32, [11,11], 3], [32, [11,11], 3]],
            # "conv_activation": "relu",
            "post_fcnet_hiddens": [256],
            "post_fcnet_activation": "relu",
#             "vf_share_layers": True,   
    },
    
    '_disable_preprocessor_api': True,
     "eager_tracing": True,
     "restart_failed_sub_environments": True,
    "lr": lr,
    'seed': 42,
    "lr_schedule": [
        [7e5, lr_start],
        [2e6, lr_end],
    ],
    'train_batch_size': 1024, #2048, # 8000 
    'sgd_minibatch_size': 256, #2048 # 64
    'num_sgd_iter': 10, #16,
    'seed': 42,
    'batch_mode': 'truncate_episodes',
    # "rollout_fragment_length": 32,
    'gamma': 0.8,    
}

result_grid = tune.Tuner(
    "PPO",
    run_config=air.RunConfig(
        stop={"episode_reward_mean": 0, 'timesteps_total': int(6e6)},
        local_dir=ray_result_logdir,
        checkpoint_config=air.CheckpointConfig(num_to_keep=2, 
                                               checkpoint_frequency = 10, 
                                               checkpoint_score_attribute = 'episode_reward_mean'),
        # callbacks=[WandbLoggerCallback(project="l5kit2", save_checkpoints=False, save_code = True),],
        ),
    param_space=config_param_space).fit()
    

#################### Retrain ####################
# ray_result_logdir = '/workspace/datasets/ray_results/29-03-2023_13-22-52/'

# tuner = tune.Tuner.restore(
#     path=ray_result_logdir + 'PPO',
# )
# tuner.fit()