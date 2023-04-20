import os
os.environ["L5KIT_DATA_FOLDER"] = '/workspace/datasets'
# os.environ['CUDA_VISIBLE_DEVICES']= '1'
import gym

# from stable_baselines3 import PPO
# from stable_baselines3.common.callbacks import CheckpointCallback
# from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.utils import get_linear_fn
# from stable_baselines3.common.vec_env import SubprocVecEnv

from l5kit.configs import load_config_data
# from l5kit.environment.feature_extractor import CustomFeatureExtractor
# from l5kit.environment.callbacks import L5KitEvalCallback
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
from src.constant import SRC_PATH


# Dataset is assumed to be on the folder specified
# in the L5KIT_DATA_FOLDER environment variable
from l5kit.configs import load_config_data

# get environment config
env_config_path = SRC_PATH + 'src/configs/gym_config84.yaml'
cfg = load_config_data(env_config_path)
ray.init(num_cpus=3, ignore_reinit_error=True, log_to_driver=False, object_store_memory = 5e9)


from src.customEnv.wrapper import L5EnvWrapper, L5EnvWrapperTorch
from ray import tune
train_eps_length = 32
train_sim_cfg = SimulationConfigGym()
train_sim_cfg.num_simulation_steps = train_eps_length + 1
# Register , how your env should be constructed (always with 5, or you can take values from the `config` EnvContext object):
env_kwargs = {'env_config_path': env_config_path, 'use_kinematic': True, 'sim_cfg': train_sim_cfg}

tune.register_env("L5-CLE-V0", lambda config: L5Env(**env_kwargs))
# tune.register_env("L5-CLE-V1", lambda config: L5EnvWrapper(env = L5Env(**env_kwargs), \
#                                                            raster_size= cfg['raster_params']['raster_size'][0], \
#                                                            n_channels = 7))
tune.register_env("L5-CLE-V1", lambda config: L5EnvWrapperTorch(env = L5Env(**env_kwargs), \
                                                           raster_size= cfg['raster_params']['raster_size'][0], \
                                                           n_channels = 7))

#################### Wandb ####################

# import numpy as np
# import ray
# from ray import air, tune
# from ray.air import session
# from ray.air.integrations.wandb import setup_wandb
# from ray.air.integrations.wandb import WandbLoggerCallback
# os.environ['WANDB_NOTEBOOK_NAME'] = '/workspace/source/rllib_sac.py'
# os.environ["WANDB_API_KEY"] = '083592c84134c040dcca598c644c348d32540a08'

# import wandb
# wandb.init(project="l5kit2", reinit = True)

#################### Train ####################
import ray
from ray import air, tune
train_envs = 4

hcmTz = pytz.timezone("Asia/Ho_Chi_Minh") 
date = datetime.datetime.now(hcmTz).strftime("%d-%m-%Y_%H-%M-%S")
ray_result_logdir = '/home/pronton/ray_results/debug_sac' + date

lr = 3e-3
lr_start = 3e-4
lr_end = 3e-5
config_param_space = {
    "env": "L5-CLE-V1",
    "framework": "torch",
    "num_gpus": 1,
    "num_workers": 2, # 63
    "num_envs_per_worker": train_envs,
    'q_model_config' : {
            # "dim": 112,
            # "conv_filters" : [[64, [7,7], 3], [32, [11,11], 3], [32, [11,11], 3]],
            # "conv_activation": "relu",
            "post_fcnet_hiddens": [256],
            "post_fcnet_activation": "relu",
        },
    'policy_model_config' : {
            # "dim": 112,
            # "conv_filters" : [[64, [7,7], 3], [32, [11,11], 3], [32, [11,11], 3]],
            # "conv_activation": "relu",
            "post_fcnet_hiddens": [256],
            "post_fcnet_activation": "relu",
        },
    'tau': 0.005,
    'target_network_update_freq': 1,
    'replay_buffer_config':{
        'type': 'MultiAgentPrioritizedReplayBuffer',
        'capacity': int(1e5), #int(1e5)
        "worker_side_prioritization": True,
    },
    'num_steps_sampled_before_learning_starts': 1024, #8000
    
    'target_entropy': 'auto',
#     "model": {
#         "custom_model": "GN_CNN_torch_model",
#         "custom_model_config": {'feature_dim':128},
#     },
#     'store_buffer_in_checkpoints': True,
#     'num_steps_sampled_before_learning_starts': 1024, # 8000,
    
#     'target_entropy': 'auto',
# #     "model": {
# #         "custom_model": "GN_CNN_torch_model",
# #         "custom_model_config": {'feature_dim':128},
# #     },
#     '_disable_preprocessor_api': True,
#      "eager_tracing": True,
#      "restart_failed_sub_environments": True,
 
    # 'train_batch_size': 4000,
    # 'sgd_minibatch_size': 256,
    # 'num_sgd_iter': 16,
    # 'store_buffer_in_checkpoints' : False,
    'seed': 42,
    'batch_mode': 'truncate_episodes',
    "rollout_fragment_length": 1,
    'train_batch_size': 256, # 2048
    'training_intensity' : 32, # (4x 'natural' value = 8) 'natural value = train_batch_size / (rollout_fragment_length x num_workers x num_envs_per_worker) = 256 / 1x 8 x 4 = 8
    'gamma': 0.8,
    'twin_q' : True,
    "lr": 3e-4,
    "min_sample_timesteps_per_iteration": 1024, # 8000
}

result_grid = tune.Tuner(
    "SAC",
    run_config=air.RunConfig(
        stop={"episode_reward_mean": 0, 'timesteps_total': int(4e6)},
        local_dir=ray_result_logdir,
        checkpoint_config=air.CheckpointConfig(num_to_keep=2, checkpoint_frequency = 10, checkpoint_score_attribute = 'episode_reward_mean'),
        # callbacks=[WandbLoggerCallback( project="l5kit2", save_checkpoints=False),],
    ),
        
    param_space=config_param_space).fit()
#################### Retrain ####################
# config_param_space['stop']['timesteps_total'] = 3e-5
# path_to_trained_agent_checkpoint = 'l5kit/ray_results/29-12-2022_07-47-22/SAC/SAC_L5-CLE-V1_5af7a_00000_0_2022-12-29_00-47-23/checkpoint_000249'
# from ray.rllib.algorithms.sac import SAC
# ray.tune.run(SAC, config=config_param_space, restore=path_to_trained_agent_checkpoint)
from ray.rllib.algorithms.sac import SAC
    # checkpoint_path = 'l5kit/ray_results/01-01-2023_15-53-49/SAC/SAC_L5-CLE-V1_cf7bb_00000_0_2023-01-01_08-53-50/checkpoint_000170'
    # checkpoint_path = '/home/pronton/ray_results/31-12-2022_07-53-04/SAC/SAC_L5-CLE-V1_7bae1_00000_0_2022-12-31_00-53-04/checkpoint_000360'
checkpoint_path = '/home/pronton/ray_results/debug_sac16-04-2023_14-35-28/SAC/SAC_L5-CLE-V1_428d6_00000_0_2023-04-16_07-35-28/checkpoint_000100'
model = SAC(config=config_param_space, env='L5-CLE-V2')
model.restore(checkpoint_path)

# ray_result_logdir = '/workspace/datasets/ray_results/debug02-04-2023_11-30-49/SAC'

# tuner = tune.Tuner.restore(
#     path=ray_result_logdir, resume_errored = True
# )
# tuner.fit()