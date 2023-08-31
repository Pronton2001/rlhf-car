import os
from src.customModel.customModel import TorchAttentionModel3, TorchVectorSharedSAC, TorchVectorQNet, TorchVectorPolicyNet

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



ray.init(num_cpus=9, ignore_reinit_error=True, log_to_driver=False, local_mode=True)
# ray.init(num_cpus=9, ignore_reinit_error=True, log_to_driver=False, local_mode=False)


from l5kit.configs import load_config_data

# get environment config
# env_config_path = '/workspace/source/configs/gym_config_history3.yaml'
# env_config_path = '/workspace/source/configs/gym_config84.yaml'
env_config_path = SRC_PATH + 'src/configs/gym_vectorizer_config.yaml'
# env_config_path = SRC_PATH + 'src/configs/gym_vectorizer_config_hist3.yaml'
cfg = load_config_data(env_config_path)


#################### Define Training and Evaluation Environments ####################
# n_channels = (cfg['model_params']['future_num_frames'] + 1)* 2 + 3
# print(cfg['model_params']['future_num_frames'], cfg['model_params']['history_num_frames'], n_channels)
from ray import tune
from src.customEnv.wrapper import L5Env2WrapperTorchCLEReward
train_eps_length = 32
train_sim_cfg = SimulationConfigGym()
train_sim_cfg.num_simulation_steps = train_eps_length + 1


# Register , how your env should be constructed (always with 5, or you can take values from the `config` EnvContext object):
env_kwargs = {'env_config_path': env_config_path, 'use_kinematic': True, 'sim_cfg': train_sim_cfg}

# tune.register_env("L5-CLE-V2", lambda config: L5Env2(**env_kwargs))
reward_kwargs = {
    'yaw_weight': 1.0,
    'dist_weight': 1.0,
    # 'd2r_weight': 0.0,
    'cf_weight': 20.0,
    'cr_weight': 20.0,
    'cs_weight': 20.0,
}
tune.register_env("L5-CLE-V2", lambda config: L5Env2WrapperTorchCLEReward(L5Env2(**env_kwargs), reward_kwargs=reward_kwargs))

# ModelCatalog.register_custom_model( "TorchAttentionModel3", TorchAttentionModel3)
# ModelCatalog.register_custom_model( "TorchAttentionModel4", TorchAttentionModel4SAC)
ModelCatalog.register_custom_model( "TorchVectorQNet", TorchVectorQNet)
ModelCatalog.register_custom_model( "TorchVectorPolicyNet", TorchVectorPolicyNet)

#################### Retrain ####################
# ray_result_logdir = '/home/pronton/ray_results/luanvan/KL_debug/SAC-T_RLFT_fixedKLKin_fixedConfig08-06-2023_17-22-05/KLSAC_2023-06-08_10-22-07'
# ray_result_logdir = '/home/pronton/ray_results/debug/luanvan/KLweight/KLRewardSAC-T_load_KLweight=1_decay_trainset_1e6_01-07-2023_09-40-32/KLRewardSAC_2023-07-01_02-40-35'
# ray_result_logdir = '/home/pronton/ray_results/debug/luanvan/KLweight/COPY-KLRewardSAC-T_load_KLweight=1_trainset_5e5_05-07-2023_14-38-09/KLRewardSAC_2023-07-05_07-38-11'
# # ray_result_logdir = '/home/pronton/ray_results/debug/KLSAC_25-07-2023_13-23-34/KLSAC_2023-07-25_06-23-36'
# ray_result_logdir = '/home/pronton/ray_results/debug/luanvan/KLweight/COPY-KLRewardSAC-T_load_KLweight=1_trainset_5e5_05-07-2023_14-38-09/KLRewardSAC_2023-07-05_07-38-11'

# tuner = tune.Tuner.restore(
#     path=ray_result_logdir, resume_errored=True, 
# )
# tuner.fit()
# exit()
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
ray_result_logdir = '/home/pronton/ray_results/debug/KLRewardSAC-T_load_KLweight=.3_trainset_1e5_steer=-1.5_' + date
# ray_result_logdir = '/home/pronton/ray_results/debug/KLSAC_' + date

train_envs = 4
lr = 3e-4
lr_start = 3e-5
lr_end = 3e-6

lr = 3e-4
lr_start = 3e-5
lr_end = 3e-6
config_param_space = {
    "env": "L5-CLE-V2",
    "framework": "torch",
    "num_gpus": 1,
    "num_workers": 8,
    "num_envs_per_worker": train_envs,
    'q_model_config':{
        'custom_model': 'TorchVectorQNet',
        "custom_model_config": {
            'cfg':cfg,
            'freeze_for_RLtuning':  False,
            'load_pretrained': True,
            'share_feature_extractor': False, # policy, q and twin-q use 1 shared feature extractor -> more efficiency
        },
    },
    'policy_model_config':{
        'custom_model': 'TorchVectorPolicyNet',
        "custom_model_config": {
            'cfg':cfg,
            'freeze_for_RLtuning': False,
            'load_pretrained': True,
            'share_feature_extractor': False,
            'kl_div_weight': .3,
            'log_std_acc': -1,
            'log_std_steer': -1.5,
            'reward_kwargs': reward_kwargs,
        },
    },
    'tau': 0.005,
    'target_network_update_freq': 1,
    'replay_buffer_config':{
        'type': 'MultiAgentPrioritizedReplayBuffer',
        'capacity': int(1e5),
        "worker_side_prioritization": True,
    },
    'num_steps_sampled_before_learning_starts': 32,
    
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
    'train_batch_size': 32, #512, #1024,#2048,
    # 'training_intensity' : 32, # (4x 'natural' value = 8) train_batch_size / (rollout_fragment_length x num_workers x num_envs_per_worker).
    'gamma': 0.8,
    'twin_q' : True,
    "lr": 3e-4,
    "min_sample_timesteps_per_iteration": 32,
}

# from src.customModel.customKLActorCriticTrainer import KLEntropyActorCritic
from src.customModel.customKLRewardSACTrainer import KLRewardSAC
# from src.customModel.customKLSACTrainer import KLSAC
result_grid = tune.Tuner(
    KLRewardSAC,
    run_config=air.RunConfig(
        stop={"episode_reward_mean": 0, 'timesteps_total': int(2048)}, 
        local_dir=ray_result_logdir,
        checkpoint_config=air.CheckpointConfig(checkpoint_frequency = 1)
        # checkpoint_config=air.CheckpointConfig(num_to_keep=2, 
        #                                        checkpoint_frequency = 10,
        #                                        checkpoint_score_attribute = 'episode_reward_mean'),
        # callbacks=[WandbLoggerCallback(project="l5kit2", save_code = True, save_checkpoints = False),],
        ),
    param_space=config_param_space).fit()
    


# from src.validate.validator import save_data
# from src.customModel.customKLRewardSACPolicy import actions
# print(actions)
# save_data(actions, f'{SRC_PATH}src/validate/testset/sac_500_actions.obj')