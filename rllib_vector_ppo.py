import os
from src.customModel.customModel import TorchAttentionModel3, TorchVectorPPO

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

ray.init(num_cpus=9, ignore_reinit_error=True, log_to_driver=False, object_store_memory = 5*10**9, local_mode=True)


from l5kit.configs import load_config_data

# get environment config
# env_config_path = '/workspace/source/configs/gym_config_history3.yaml'
# env_config_path = '/workspace/source/configs/gym_config84.yaml'
env_config_path = SRC_PATH + 'src/configs/gym_vectorizer_config.yaml'
# env_config_path = SRC_PATH + 'src/configs/gym_vectorizer_config_hist3.yaml'
cfg = load_config_data(env_config_path)


#################### Define Training and Evaluation Environments ####################
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
ModelCatalog.register_custom_model( "TorchVectorPPO", TorchVectorPPO)
# ModelCatalog.register_custom_model( "TorchAttentionModel4Pretrained", TorchAttentionModel4Pretrained)
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
# ray_result_logdir = '/home/pronton/ray_results/debug_vector_ppo_newModel/separated_kin_hist3_updated2_CLErewardNew_dropoutMHAhead=0.0_2actions_loadpretrained_3fclayers_hist3' + date
ray_result_logdir = '/home/pronton/ray_results/luanvan/PPO-T_nonfreeze_nonload' + date
ray_result_logdir = '/home/pronton/ray_results/debug/KLRewardPPO-T_train9Gset' + date

train_envs = 4
lr = 3e-3
lr_start = 3e-4
lr_end = 3e-5
lr_time = int(4e6)

config_param_space = {
    "env": "L5-CLE-V2",
    "framework": "torch",
    "num_gpus": 1,
    "num_workers": 8,
    "num_envs_per_worker": train_envs,
    "model": {
        "custom_model": "TorchVectorPPO",
        "custom_model_config": {
            'cfg':cfg,
            'freeze_for_RLtuning':  False,
            'load_pretrained': False,
            'shared_feature_extractor': False,
            'kl_div_weight': 1,
            'log_std_acc': -1,
            'log_std_steer': -1,
            'reward_kwargs': reward_kwargs,
        },
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

from src.customModel.customKLRewardPPOTrainer import KLRewardPPO

result_grid = tune.Tuner(
        KLRewardPPO, 
        run_config=air.RunConfig(
        stop={"episode_reward_mean": 0, 'timesteps_total': int(500)},
        local_dir=ray_result_logdir,
        checkpoint_config=air.CheckpointConfig(num_to_keep=2, 
                                               checkpoint_frequency = 10, 
                                               checkpoint_score_attribute = 'episode_reward_mean'),
        # callbacks=[WandbLoggerCallback(project="l5kit2", save_code = True, save_checkpoints = False),],
        ),
    param_space=config_param_space).fit()
    
from src.validate.validator import save_data
from src.customModel.customKLRewardPPOPolicy import actions
print(actions)
save_data(actions, f'{SRC_PATH}src/validate/testset/500_actions.obj')

#################### Retrain ####################
# ray_result_logdir = '/workspace/datasets/ray_results/01-04-2023_19-55-37_(PPO~-70)/PPO'

# tuner = tune.Tuner.restore(
#     path=ray_result_logdir, resume_errored = True,
# )
# tuner.fit()