import time
start = time.time()
import os
os.environ["L5KIT_DATA_FOLDER"] = '/workspace/datasets'
os.environ['CUDA_VISIBLE_DEVICES']= '0'
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
from src.customModel.customModel import TorchRasterNet

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
ray.init(num_cpus=9, ignore_reinit_error=True, log_to_driver=False, object_store_memory = 5*10**9, local_mode= False)


from l5kit.configs import load_config_data

env_config_path = '/workspace/source/src/configs/gym_rasterizer_config.yaml'
cfg = load_config_data(env_config_path)

ModelCatalog.register_custom_model( "TorchSeparatedRasterModel", TorchRasterNet)

n_channels = (cfg['model_params']['history_num_frames'] + 1)* 2 + 3
print('num channels:', n_channels)
from ray import tune
from src.customEnv.wrapper import L5EnvRasterizerTorch
train_eps_length = 32
train_sim_cfg = SimulationConfigGym()
train_sim_cfg.num_simulation_steps = train_eps_length + 1


env_kwargs = {'env_config_path': env_config_path, 'use_kinematic': False, 'sim_cfg': train_sim_cfg, 'rescale_action': False}
tune.register_env("L5-CLE-V1", lambda config: L5EnvRasterizerTorch(env = L5Env(**env_kwargs), \
                                                           raster_size= cfg['raster_params']['raster_size'][0], \
                                                           n_channels = n_channels))

#################### Train ####################

import ray
from ray import air, tune
train_envs = 4

hcmTz = pytz.timezone("Asia/Ho_Chi_Minh") 
date = datetime.datetime.now(hcmTz).strftime("%d-%m-%Y_%H-%M-%S")
ray_result_logdir = '/workspace/datasets/ray_results/debug_unfreeze_actorNet' + date

# lr = 3e-3
lr_start = 3e-5
lr_end = 3e-6

pretrained_policy = RasterizedPlanningModelFeature(
                model_arch="resnet50",
                num_input_channels=5,
                num_targets=3 * cfg["model_params"]["future_num_frames"],  # X, Y, Yaw * number of future states
                weights_scaling=[1., 1., 1.],
                criterion=nn.MSELoss(reduction="none"),)

model_path = "/workspace/source/src/model/planning_model_20201208.pt"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pretrained_policy.load_state_dict(torch.load(model_path).state_dict())
# pretrained_policy.to(device)
config_param_space = {
    "env": "L5-CLE-V1",
    "framework": "torch",
    "num_gpus": 1,
    "num_workers": 8,
    "num_envs_per_worker": train_envs, #8 * 32
    'disable_env_checking':True,
    "model": {
            "custom_model": "TorchSeparatedRasterModel",
            # Extra kwargs to be passed to your model's c'tor.
            "custom_model_config": {
                'future_num_frames':cfg["model_params"]["future_num_frames"],
                'freeze_actor': False,
                },
            },
    "pretrained_policy": pretrained_policy,
    '_disable_preprocessor_api': True,
    "eager_tracing": True,
    "restart_failed_sub_environments": True,
    # "lr": lr,
    'seed': 42,
    "lr_schedule": [
         [1e6, lr_start],
         [2e6, lr_end],
     ],
    'train_batch_size': 1024, # 8000 
    'sgd_minibatch_size': 64, #2048
    'num_sgd_iter': 10,#10,#16,
    'seed': 42,
    # 'batch_mode': 'truncate_episodes',
    # "rollout_fragment_length": 32,
    'gamma': 0.8,    
}


from src.customModel.customPPOTrainer import KLPPO
# ray.tune.run(KLPPO, config=config_param_space, restore=path_to_trained_agent_checkpoint)
# checkpoint_path = '/workspace/datasets/ray_results/08-04-2023_14-17-36(RasterPPO_vf~2)/KLPPO_2023-04-08_07-17-36/KLPPO_L5-CLE-V1_70625_00000_0_2023-04-08_07-17-37/checkpoint_000030'
model = KLPPO(config=config_param_space, env='L5-CLE-V1')
# model.restore(checkpoint_path)
from ray.tune.logger import pretty_print
for i in range(10000):
    print('alo')
    result = model.train()
    print(pretty_print(result))
# result_grid = tune.Tuner(
#     KLPPO,
#     run_config=air.RunConfig(
#         stop={"episode_reward_mean": 0, 'timesteps_total': int(6e6)},
#         local_dir=ray_result_logdir,
#         checkpoint_config=air.CheckpointConfig(num_to_keep=2, 
#                                             checkpoint_frequency = 10, 
#                                             checkpoint_score_attribute = 'episode_reward_mean'),
#         # callbacks=[WandbLoggerCallback(project="l5kit2", save_code = True, save_checkpoints = False),],
#         ),
#     param_space=config_param_space).fit()
