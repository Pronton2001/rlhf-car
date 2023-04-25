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

ray.init(num_cpus=1, ignore_reinit_error=True, log_to_driver=False, local_mode=False)



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
eval_eps_length = 32
eval_sim_cfg = SimulationConfigGym()
eval_sim_cfg.num_simulation_steps = eval_eps_length + 1

# Register , how your env should be constructed (always with 5, or you can take values from the `config` EnvContext object):
env_kwargs = {'env_config_path': env_config_path, 'use_kinematic': True, 'sim_cfg': eval_sim_cfg}

tune.register_env("L5-CLE-V2", lambda config: L5Env2(**env_kwargs))
ModelCatalog.register_custom_model( "TorchAttentionModel3", TorchAttentionModel3)
ModelCatalog.register_custom_model( "TorchAttentionModel4", TorchAttentionModel4SAC)
ModelCatalog.register_custom_model( "TorchVectorQNet", TorchVectorQNet)
ModelCatalog.register_custom_model( "TorchVectorPolicyNet", TorchVectorPolicyNet)


train_envs = 4
lr = 3e-4
lr_start = 3e-5
lr_end = 3e-6
lr = 3e-4
lr_start = 3e-5
lr_end = 3e-6
eval_config_param_space = {
    # "env": "L5-CLE-EVAL-V2",
    "framework": "torch",
    "num_gpus": 0,
    "num_workers": 1,
    "num_envs_per_worker": train_envs,
    'q_model_config':{
        'custom_model': 'TorchVectorQNet',
        'custom_model_config': {'cfg': cfg,},
        "post_fcnet_hiddens": [256],
        "post_fcnet_activation": "relu",
    },
    'policy_model_config':{
        'custom_model': 'TorchVectorPolicyNet',
        'custom_model_config': {'cfg': cfg,},
        "post_fcnet_hiddens": [256],
        "post_fcnet_activation": "relu",
    },
    'tau': 0.005,
    'target_network_update_freq': 1,
    'replay_buffer_config':{
        'type': 'MultiAgentPrioritizedReplayBuffer',
        'capacity': int(1e6),
        "worker_side_prioritization": True,
    },
    'num_steps_sampled_before_learning_starts': 1024,
    
    'target_entropy': 'auto',
#     "model": {
#         "custom_model": "GN_CNN_torch_model",
#         "custom_model_config": {'feature_dim':128},
#     },
    '_disable_preprocessor_api': True,
     "eager_tracing": True,
     "restart_failed_sub_environments": True,
     'disable_env_checking': True,
 
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
    "min_sample_timesteps_per_iteration": 1024,
}

rollout_sim_cfg = SimulationConfigGym()
rollout_sim_cfg.num_simulation_steps = 50

# Register , how your env should be constructed (always with 5, or you can take values from the `config` EnvContext object):
env_kwargs = {'env_config_path': env_config_path, 
              'use_kinematic': True, 
              'sim_cfg': rollout_sim_cfg,  
              'train': True,
              'return_info': True}

rollout_env = L5Env2(**env_kwargs)
tune.register_env("L5-CLE-EVAL-V2", lambda config: L5Env2(**env_kwargs))

from ray.rllib.algorithms.sac import SAC
checkpoint_path = '/home/pronton/ray_results/vector_sac(-25)/SAC/SAC_L5-CLE-V2_a638f_00000_0_2023-04-24_14-52-35/checkpoint_000540'
model = SAC(config=eval_config_param_space, env='L5-CLE-EVAL-V2')
model.restore(checkpoint_path)

rollout_env.set_reset_id(1)
obs = rollout_env.reset()
action = model.compute_single_action(input_dict= {'obs': obs}, explore= False)
# action = model.compute_single_action(obs, full_fetch = False)
print(action)

# print(model.compute_single_action(obs2, deterministic = True))