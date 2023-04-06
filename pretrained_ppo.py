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
from src.customModel.customModel import TorchAttentionModel3

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

ray.init(num_cpus=9, ignore_reinit_error=True, log_to_driver=False, object_store_memory = 5*10**9)


from l5kit.configs import load_config_data

# get environment config
# env_config_path = '/workspace/source/configs/gym_config_history3.yaml'
# env_config_path = '/workspace/source/configs/gym_config84.yaml'
# env_config_path = '/workspace/source/src/configs/gym_config84.yaml'
env_config_path = '/workspace/source/src/configs/gym_vectorizer_config.yaml'
cfg = load_config_data(env_config_path)


#################### Define Training and Evaluation Environments ####################
n_channels = (cfg['model_params']['future_num_frames'] + 1)* 2 + 3
print(cfg['model_params']['future_num_frames'], cfg['model_params']['history_num_frames'], n_channels)
from ray import tune
from src.customEnv.wrapper import L5EnvWrapper, L5EnvWrapperTorch
train_eps_length = 32
train_sim_cfg = SimulationConfigGym()
train_sim_cfg.num_simulation_steps = train_eps_length + 1


# Register , how your env should be constructed (always with 5, or you can take values from the `config` EnvContext object):
# env_kwargs = {'env_config_path': env_config_path, 'use_kinematic': True, 'sim_cfg': train_sim_cfg}

# tune.register_env("L5-CLE-V0", lambda config: L5Env(**env_kwargs))
# # tune.register_env("L5-CLE-V1", lambda config: L5EnvWrapper(env = L5Env(**env_kwargs), \
# #                                                            raster_size= cfg['raster_params']['raster_size'][0], \
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
import ray.rllib.algorithms.ppo as ppo
from pprint import PrettyPrinter
import numpy as np
from l5kit.configs.config import load_config_data
from l5kit.data.local_data_manager import LocalDataManager
from l5kit.environment.envs.l5_env2 import GymStepOutput, SimulationConfigGym, L5Env2
import os
from ray.rllib.models import ModelCatalog
# model = TorchGNCNN(np.zeros((112,112,7)), np.array((3,)),3, model_config= {'custom_model_config': {'feature_dim': 128}}, name='')

# # In L5env
# batch_data = {'obs': torch.ones((32,7, 112, 112))}
# print('batch', batch_data['obs'].shape)

# # After process in L5envWrapper
# batch_data = {'obs': torch.ones((32, 112, 112, 7))}


# # obs_transformed = obs_batch.permute(0, 3, 1, 2) # 32 x 112 x 112 x 7 [B, size, size, channels]
# # print('transformed', obs_transformed.shape)
# # print(obs_transformed.shape)
# model(input_dict=batch_data)

# print(rollout_env.action_space)
# model = TorchAttentionModel3(np.zeros((112,112,7)), np.array((3,)),3, model_config= {"custom_model_config": {'cfg':cfg}}, name='')
# for param_tensor in model.state_dict():
#     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
###################### TRAINING ######################
ModelCatalog.register_custom_model( "TorchSeparatedAttentionModel", TorchAttentionModel3)
from ray import tune
import ray
train_eps_length = 32
train_sim_cfg = SimulationConfigGym()
train_sim_cfg.num_simulation_steps = train_eps_length + 1
env_kwargs = {'env_config_path': env_config_path, 'use_kinematic': False, 'sim_cfg': train_sim_cfg, 'rescale_action': False}
tune.register_env("L5-CLE-V2", lambda config: L5Env2(**env_kwargs))
ray.init(num_cpus=9, ignore_reinit_error=True, log_to_driver=False, local_mode=False)
# algo = ppo.PPO(
#         env="L5-CLE-V2",
#         config={
#             'disable_env_checking':True,
#             "framework": "torch",
#             'log_level': 'INFO',
#             'num_gpu': 0,
#             'train_batch_size': 1,
#             'sgd_minibatch_size': 1,
#             'num_sgd_iter': 1,
#             'seed': 42,
#             'batch_mode': 'truncate_episodes',
#             # "rollout_fragment_length": 32,
#             "model": {
#                 "custom_model": "TorchSeparatedAttentionModel",
#                 # Extra kwargs to be passed to your model's c'tor.
#                 "custom_model_config": {'cfg':cfg},
#             },
#             # "output": "/home/pronton/rl/l5kit/examples/RL/notebooks/logs/l5env2-out", 
#             # "output_max_file_size": 5000000,
#             '_disable_preprocessor_api': True,
#         },
#     )

# for i in range(1):
#     result = algo.train()
#     print(PrettyPrinter(result))
import ray
from ray import air, tune
import pytz
import datetime
hcmTz = pytz.timezone("Asia/Ho_Chi_Minh") 
date = datetime.datetime.now(hcmTz).strftime("%d-%m-%Y_%H-%M-%S")
ray_result_logdir = '~/ray_results/debug' + date

train_envs = 4
lr = 3e-3
# lr_start = 3e-4
# lr_end = 3e-5
# lr_time = int(4e6)

config_param_space = {
    "env": "L5-CLE-V2",
    "framework": "torch",
    "num_gpus": 1,
    "num_workers": 8,
    "num_envs_per_worker": train_envs,
    'disable_env_checking':True,
    "model": {
            "custom_model": "TorchSeparatedAttentionModel",
            # Extra kwargs to be passed to your model's c'tor.
            "custom_model_config": {'cfg':cfg},
            },

    '_disable_preprocessor_api': True,
    "eager_tracing": True,
    "restart_failed_sub_environments": True,
    "lr": lr,
    'seed': 42,
    # "lr_schedule": [
    #     [1e6, lr_start],
    #     [2e6, lr_end],
    # ],
    'train_batch_size': 2048, # 8000 
    'sgd_minibatch_size': 512, #2048
    'num_sgd_iter': 10,#16,
    'seed': 42,
    # 'batch_mode': 'truncate_episodes',
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
        # callbacks=[WandbLoggerCallback(project="l5kit2", save_code = True, save_checkpoints = False),],
        ),
    param_space=config_param_space).fit()


#################### Retrain ####################
# ray_result_logdir = '/workspace/datasets/ray_results/01-04-2023_19-55-37_(PPO~-70)/PPO'

# tuner = tune.Tuner.restore(
#     path=ray_result_logdir, resume_errored = True,
# )
# tuner.fit()
