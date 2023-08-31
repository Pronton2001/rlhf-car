import os

os.environ["L5KIT_DATA_FOLDER"] = '/workspace/datasets'
# os.environ['CUDA_VISIBLE_DEVICES']= '1'
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
from src.constant import SRC_PATH


LOADED = False

if not LOADED:
    env_config_path = SRC_PATH + 'src/configs/gym_config84.yaml'
    cfg = load_config_data(env_config_path)
    import ray
    train_envs = 4

    lr = 3e-3
    lr_start = 3e-4
    lr_end = 3e-5
    config_param_space = {
        # "env": "L5-CLE-V1",
        "framework": "torch",
        "num_gpus": 0,
        "num_workers": 0, # 63
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
        'train_batch_size': 256, # 2048
        'training_intensity' : 32, # (4x 'natural' value = 8) 'natural value = train_batch_size / (rollout_fragment_length x num_workers x num_envs_per_worker) = 256 / 1x 8 x 4 = 8
        'gamma': 0.8,
        'twin_q' : True,
        "lr": 3e-4,
        "min_sample_timesteps_per_iteration": 1024, # 8000
    }

    from src.customEnv.wrapper import L5EnvWrapperTorchCLEReward
    from ray import tune
    rollout_sim_cfg = SimulationConfigGym()
    rollout_sim_cfg.num_simulation_steps = None

    eval_env_kwargs = {'env_config_path': env_config_path, 'use_kinematic': True, 'sim_cfg': rollout_sim_cfg, 'train': False, 'return_info': True}
    # env_kwargs = {'env_config_path': env_config_path, 'use_kinematic': True, 'sim_cfg': train_sim_cfg}
    # rollout_env = L5Env2(**eval_env_kwargs)
    rollout_env = L5EnvWrapperTorchCLEReward(env = L5Env(**eval_env_kwargs), \
                                            raster_size= cfg['raster_params']['raster_size'][0], \
                                            n_channels = 7)
    tune.register_env("L5-CLE-EVAL-V1", lambda config: L5EnvWrapperTorchCLEReward(env = L5Env(**eval_env_kwargs), \
                                                            raster_size= cfg['raster_params']['raster_size'][0], \
                                                            n_channels = 7))
    # tune.register_env("L5-CLE-EVAL-V2", lambda config: L5Env2(**eval_env_kwargs))
    
    checkpoint_path = '/home/pronton/ray_results/luanvan/SAC_CNN_CLEreward05-05-2023_20-13-20/SAC/SAC_L5-CLE-V1_9b487_00000_0_2023-05-05_13-13-20/checkpoint_000150'

    from ray.rllib.algorithms.sac import SAC
    model = SAC(config=config_param_space, env='L5-CLE-EVAL-V1')
    model.restore(checkpoint_path)

    from torchsummary import summary
    summary(model.get_policy().model)

    # from src.simulation.unrollGym import unroll
    # sim_outs = unroll(model, rollout_env, 100, cfg['gym_params']['max_val_scene_id'])
    # from src.validate.validator import quantify_outputs, save_data
    # quantify_outputs(sim_outs)
    # save_data(sim_outs, f'{SRC_PATH}/src/validate/sac_cnn_checkpoint150(-73).obj')
else:
    import pickle
    file = open(f'{SRC_PATH}/src/validate/sac_cnn_checkpoint150(-73).obj', 'rb')
    from src.validate.validator import quantify_outputs
    sim_outs = pickle.load(file)
    quantify_outputs(sim_outs)
