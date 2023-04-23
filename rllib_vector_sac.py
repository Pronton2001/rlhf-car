import os
from src.customModel.customModel import TorchAttentionModel3, TorchAttentionModel4SAC, TorchVectorQNet, TorchVectorPolicyNet

from src.constant import SRC_PATH
os.environ["L5KIT_DATA_FOLDER"] = '/workspace/datasets'
# os.environ['CUDA_VISIBLE_DEVICES']= '0'
# os.environ["TUNE_RESULT_DIR"] =  '/DATA/l5kit/rllib_tb_logs'
import gym
from l5kit.configs import load_config_data
from l5kit.environment.envs.l5_env import SimulationConfigGym, GymStepOutput, L5Env
from l5kit.environment.envs.l5_env2 import SimulationConfigGym, GymStepOutput, L5Env2
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

ray.init(num_cpus=64, ignore_reinit_error=True, log_to_driver=False, object_store_memory = 5*10**9, local_mode=False)


from l5kit.configs import load_config_data

# get environment config
# env_config_path = '/workspace/source/configs/gym_config_history3.yaml'
# env_config_path = '/workspace/source/configs/gym_config84.yaml'
env_config_path = SRC_PATH + 'src/configs/gym_vectorizer_config.yaml'
env_config_path = SRC_PATH + 'src/configs/gym_vectorizer_config_hist3.yaml'
# env_config_path = '/workspace/source/src/configs/gym_vectorizer_config.yaml'
cfg = load_config_data(env_config_path)


#################### Define Training and Evaluation Environments ####################
# n_channels = (cfg['model_params']['future_num_frames'] + 1)* 2 + 3
# print(cfg['model_params']['future_num_frames'], cfg['model_params']['history_num_frames'], n_channels)
from ray import tune
from src.customEnv.wrapper import L5EnvWrapper, L5EnvWrapperTorch
train_eps_length = 32
train_sim_cfg = SimulationConfigGym()
train_sim_cfg.num_simulation_steps = train_eps_length + 1


# Register , how your env should be constructed (always with 5, or you can take values from the `config` EnvContext object):
env_kwargs = {'env_config_path': env_config_path, 'use_kinematic': True, 'sim_cfg': train_sim_cfg}

tune.register_env("L5-CLE-V2", lambda config: L5Env2(**env_kwargs))
ModelCatalog.register_custom_model( "TorchAttentionModel3", TorchAttentionModel3)
ModelCatalog.register_custom_model( "TorchAttentionModel4", TorchAttentionModel4SAC)
ModelCatalog.register_custom_model( "TorchVectorQNet", TorchVectorQNet)
ModelCatalog.register_custom_model( "TorchVectorPolicyNet", TorchVectorPolicyNet)
# tune.register_env("L5-CLE-V1", lambda config: L5EnvWrapper(env = L5Env(**env_kwargs), \
#                                                            raster_size= cfg['raster_params']['raster_size'][0], \
# #                                                            n_channels = n_channels))
# tune.register_env("L5-CLE-V1", lambda config: L5EnvWrapperTorch(env = L5Env(**env_kwargs), \
#                                                            raster_size= cfg['raster_params']['raster_size'][0], \
#                                                            n_channels = n_channels))

#################### Wandb ####################

# import numpy as np
# import ray
# from ray import air, tune
# from ray.air import session
# from ray.air.integrations.wandb import setup_wandb
# from ray.air.integrations.wandb import WandbLoggerCallback
# os.environ['WANDB_NOTEBOOK_NAME'] = '/workspace/source/rllib_ppo.py'
# os.environ["WANDB_API_KEY"] = '083592c84134c040dcca598c644c348d32540a08'


# import wandb
# wandb.init(project="l5kit2", reinit = True)

#################### Train ####################
import ray
from ray import air, tune
hcmTz = pytz.timezone("Asia/Ho_Chi_Minh") 
date = datetime.datetime.now(hcmTz).strftime("%d-%m-%Y_%H-%M-%S")
ray_result_logdir = '/home/pronton/ray_results/debug_vector_sac_test_q_model' + date

train_envs = 4
lr = 3e-4
lr_start = 3e-5
lr_end = 3e-6
# config_param_space = {
#     "env": "L5-CLE-V2",
#     "framework": "torch",
#     "num_gpus": 1,
#     "num_workers": 8, # 63
#     "num_envs_per_worker": train_envs,
#     'q_model_config':{
#         'custom_model': 'TorchVectorQNet',
#         'custom_model_config': {'cfg': cfg,}
#     },
#     'policy_model_config':{
#         'custom_model': 'TorchVectorPolicyNet',
#         'custom_model_config': {'cfg': cfg,}
#     },

#     # 'q_model_config' : {
#     #         # "dim": 112,
#     #         # "conv_filters" : [[64, [7,7], 3], [32, [11,11], 3], [32, [11,11], 3]],
#     #         # "conv_activation": "relu",
#     #         "post_fcnet_hiddens": [256],
#     #         "post_fcnet_activation": "relu",
#     #     },
#     # 'policy_model_config' : {
#     #         # "dim": 112,
#     #         # "conv_filters" : [[64, [7,7], 3], [32, [11,11], 3], [32, [11,11], 3]],
#     #         # "conv_activation": "relu",
#     #         "post_fcnet_hiddens": [256],
#     #         "post_fcnet_activation": "relu",
#     #     },
#     'disable_env_checking': True,
#     'tau': 0.005,
#     'target_network_update_freq': 1,
#     'replay_buffer_config':{
#         'type': 'MultiAgentPrioritizedReplayBuffer',
#         'capacity': int(1e5), #int(1e5)
#         "worker_side_prioritization": True,
#     },
#     'num_steps_sampled_before_learning_starts': 1024, #8000
    
#     'target_entropy': 'auto',
# #     "model": {
# #         "custom_model": "GN_CNN_torch_model",
# #         "custom_model_config": {'feature_dim':128},
# #     },
# #     'store_buffer_in_checkpoints': True,
# #     'num_steps_sampled_before_learning_starts': 1024, # 8000,
    
# #     'target_entropy': 'auto',
# # #     "model": {
# # #         "custom_model": "GN_CNN_torch_model",
# # #         "custom_model_config": {'feature_dim':128},
# # #     },
#     '_disable_preprocessor_api': True,
# #      "eager_tracing": True,
# #      "restart_failed_sub_environments": True,
 
#     # 'train_batch_size': 4000,
#     # 'sgd_minibatch_size': 256,
#     # 'num_sgd_iter': 16,
#     # 'store_buffer_in_checkpoints' : False,
#     'seed': 42,
#     'batch_mode': 'truncate_episodes',
#     "rollout_fragment_length": 1,
#     'train_batch_size': 256, # 2048
#     'training_intensity' : 32, # (4x 'natural' value = 8) 'natural value = train_batch_size / (rollout_fragment_length x num_workers x num_envs_per_worker) = 256 / 1x 8 x 4 = 8
#     'gamma': 0.8,
#     'twin_q' : True,
#     "lr": 3e-4,
#     "min_sample_timesteps_per_iteration": 1024, # 8000
# }

lr = 3e-4
lr_start = 3e-5
lr_end = 3e-6
config_param_space = {
    "env": "L5-CLE-V2",
    "framework": "torch",
    "num_gpus": 1,
    "num_workers": 63,
    "num_envs_per_worker": train_envs,
    'q_model_config':{
        'custom_model': 'TorchVectorQNet',
        'custom_model_config': {'cfg': cfg,}
    },
    'policy_model_config':{
        'custom_model': 'TorchVectorPolicyNet',
        'custom_model_config': {'cfg': cfg,}
    },
    'tau': 0.005,
    'target_network_update_freq': 1,
    'replay_buffer_config':{
        'type': 'MultiAgentPrioritizedReplayBuffer',
        'capacity': int(4e5),
        "worker_side_prioritization": True,
    },
    'num_steps_sampled_before_learning_starts': 8000,
    
    'target_entropy': 'auto',
#     "model": {
#         "custom_model": "GN_CNN_torch_model",
#         "custom_model_config": {'feature_dim':128},
#     },
    '_disable_preprocessor_api': True,
     "eager_tracing": True,
     "restart_failed_sub_environments": True,
 
    # 'train_batch_size': 4000,
    # 'sgd_minibatch_size': 256,
    # 'num_sgd_iter': 16,
    # 'store_buffer_in_checkpoints' : False,
    'seed': 42,
    'batch_mode': 'truncate_episodes',
    "rollout_fragment_length": 1,
    'train_batch_size': 256, #512, #1024,#2048,
    'training_intensity' : 32, # (4x 'natural' value = 8) train_batch_size / (rollout_fragment_length x num_workers x num_envs_per_worker).
    'gamma': 0.8,
    'twin_q' : True,
    "lr": 3e-4,
    "min_sample_timesteps_per_iteration": 8000,
}

result_grid = tune.Tuner(
    "SAC",
    run_config=air.RunConfig(
        stop={"episode_reward_mean": 0, 'timesteps_total': int(6e6)},
        local_dir=ray_result_logdir,
        checkpoint_config=air.CheckpointConfig(num_to_keep=2, 
                                               checkpoint_frequency = 10, 
                                               checkpoint_score_attribute = 'episode_reward_mean'),
        # callbacks=[WandbLoggerCallback(project="l5kit2", save_code = True, save_checkpoints = False),],
        ),
    param_space=config_param_space).fit()
    

#################### Retrain ####################
# ray_result_logdir = '/workspace/datasets/ray_results/01-04-2023_19-55-37_(PPO~-70)/PPO'

# tuner = tune.Tuner.restore(
#     path=ray_result_logdir, resume_errored = True,
# )
# tuner.fit()