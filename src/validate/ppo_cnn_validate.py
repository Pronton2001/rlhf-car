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
# ray.init(num_cpus=1, ignore_reinit_error=True, log_to_driver=False, local_mode= False)


from l5kit.configs import load_config_data

env_config_path = SRC_PATH +'src/configs/gym_config84.yaml'
# env_config_path = SRC_PATH +'src/configs/gym_config.yaml'
cfg = load_config_data(env_config_path)
# os.environ['CUDA_VISIBLE_DEVICES']= '1'
n_channels = (cfg['model_params']['history_num_frames'] + 1)* 2 + 3
print('num channels:', n_channels)
from ray import tune
from src.customEnv.wrapper import L5EnvRasterizerTorch, L5EnvWrapperTorch, L5EnvWrapper
train_eps_length = 32
train_sim_cfg = SimulationConfigGym()
train_sim_cfg.num_simulation_steps = train_eps_length + 1


import ray
from ray import air, tune
train_envs = 4

LOADED = False
if not LOADED:
    train_envs = 4
    lr = 3e-4
    lr_start = 3e-5
    lr_end = 3e-6
    lr_time = int(4e6)
    config_param_space = {
        "env": "L5-CLE-V1",
        "framework": "torch",
        "num_gpus": 1,
        "num_workers": 8,
        "num_envs_per_worker": train_envs,
        "model": {
            "custom_model": "GN_CNN_torch_model",
            "custom_model_config": {'feature_dim':128},
        },
        
        'model' : {
                # "dim": 84,
                # "conv_filters" : [[64, [7,7], 3], [32, [11,11], 3], [32, [11,11], 3]],
                # "conv_activation": "relu",
                "post_fcnet_hiddens": [256],
                "post_fcnet_activation": "relu",
                "vf_share_layers": False,   
                # "use_attention": True,
                # "attention_num_transformer_units": 1,
                # "attention_dim": 64,
                # "attention_num_heads": 2,
                # "attention_memory_inference": 100,
                # "attention_memory_training": 50,
                # "attention_use_n_prev_actions": 0,
                # "attention_use_n_prev_rewards": 0,
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
        'train_batch_size': 1024, # 8000 
        'sgd_minibatch_size': 512, #2048
        'num_sgd_iter': 10,#16,
        'batch_mode': 'truncate_episodes',
        # "rollout_fragment_length": 32,
        'gamma': 0.8,    
    }


    from src.customEnv.wrapper import  L5EnvWrapperTorchCLEReward
    from ray import tune
    rollout_sim_cfg = SimulationConfigGym()
    rollout_sim_cfg.num_simulation_steps = None
    eval_env_kwargs = {'env_config_path': env_config_path, 'use_kinematic': True, 'sim_cfg': rollout_sim_cfg, 'train': False, 'return_info': True}
    rollout_env = L5EnvWrapperTorch(env = L5Env(**eval_env_kwargs), \
                                            raster_size= cfg['raster_params']['raster_size'][0], \
                                            n_channels = 7)
    tune.register_env("L5-CLE-EVAL-V1", lambda config: L5EnvWrapperTorchCLEReward(env = L5Env(**eval_env_kwargs), \
                                                            raster_size= cfg['raster_params']['raster_size'][0], \
                                                            n_channels = 7))  

    checkpoint_path = '/home/pronton/ray_results/luanvan/PPO-CNN_CLEreward_06-05-2023_10-22-27/PPO/PPO_L5-CLE-V1_39fe3_00000_0_2023-05-06_03-22-27/checkpoint_000550'
    checkpoint_path = '/home/pronton/ray_results/luanvan/PPO-CNN_CLEreward_06-05-2023_10-22-27/PPO/PPO_L5-CLE-V1_39fe3_00000_0_2023-05-06_03-22-27/checkpoint_000640'
    checkpoint_path = '/home/pronton/ray_results/luanvan/PPO_CNN_8411-06-2023_16-25-36/PPO/PPO_L5-CLE-V1_ec5a5_00000_0_2023-06-11_09-25-36/checkpoint_000590'

    from ray.rllib.algorithms.ppo import PPO
    model = PPO(config=config_param_space, env='L5-CLE-EVAL-V1')
    model.restore(checkpoint_path)

    from torchsummary import summary
    summary(model.get_policy().model)

    from src.simulation.unrollGym import unroll_to_quantitative
    sim_outs = unroll_to_quantitative(model, rollout_env, 100, cfg['gym_params']['max_val_scene_id'])
    from src.validate.validator import quantify_outputs, save_data, CLEValidator
    quantify_outputs(sim_outs)
    CLEValidator(sim_outs)
    save_data(sim_outs, f'{SRC_PATH}/src/validate/ppo_cnn_cp590(-77).obj')
else:
    import pickle
    file = open(f'{SRC_PATH}/src/validate/ppo_cnn_checkpoint640(-84).obj', 'rb')
    from src.validate.validator import quantify_outputs, CLEValidator
    sim_outs = pickle.load(file)
    quantify_outputs(sim_outs)
    CLEValidator(sim_outs)