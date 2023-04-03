import os
os.environ["L5KIT_DATA_FOLDER"] = '/workspace/datasets'
os.environ['CUDA_VISIBLE_DEVICES']= '0'


import gym

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.vec_env import SubprocVecEnv

from l5kit.configs import load_config_data
from l5kit.environment.feature_extractor import CustomFeatureExtractor
from l5kit.environment.callbacks import L5KitEvalCallback
from l5kit.environment.envs.l5_env import SimulationConfigGym

from l5kit.visualization.visualizer.zarr_utils import episode_out_to_visualizer_scene_gym_cle
from l5kit.visualization.visualizer.visualizer import visualize
from bokeh.io import output_notebook, show
from l5kit.environment.gym_metric_set import L2DisplacementYawMetricSet, CLEMetricSet
from prettytable import PrettyTable
import datetime
import pytz
# Dataset is assumed to be on the folder specified
# in the L5KIT_DATA_FOLDER environment variable

# get environment config
env_config_path = '/workspace/source/src/configs/gym_config.yaml'
cfg = load_config_data(env_config_path)

# Train on episodes of length 32 time steps
train_eps_length = 32
train_envs = 4

# Evaluate on entire scene (~248 time steps)
# eval_eps_length = None
# eval_envs = 1

# make train env
train_sim_cfg = SimulationConfigGym()
train_sim_cfg.num_simulation_steps = train_eps_length + 1
env_kwargs = {'env_config_path': env_config_path, 'use_kinematic': True, 'sim_cfg': train_sim_cfg}
env = make_vec_env("L5-CLE-v0", env_kwargs=env_kwargs, n_envs=train_envs,
                   vec_env_cls=SubprocVecEnv, vec_env_kwargs={"start_method": "fork"})

# make eval env
validation_sim_cfg = SimulationConfigGym()
validation_sim_cfg.num_simulation_steps = None
eval_env_kwargs = {'env_config_path': env_config_path, 'use_kinematic': True, \
                   'return_info': True, 'train': False, 'sim_cfg': validation_sim_cfg}
eval_env = make_vec_env("L5-CLE-v0", env_kwargs=eval_env_kwargs, n_envs=eval_envs,
                        vec_env_cls=SubprocVecEnv, vec_env_kwargs={"start_method": "fork"})

# A simple 2 Layer CNN architecture with group normalization
model_arch = 'simple_gn'
features_dim = 128

# Custom Feature Extractor backbone
policy_kwargs = {
    "features_extractor_class": CustomFeatureExtractor,
    "features_extractor_kwargs": {"features_dim": features_dim, "model_arch": model_arch},
    "normalize_images": False
}

# Clipping schedule of PPO epsilon parameter
start_val = 0.1
end_val = 0.01
training_progress_ratio = 1.0
clip_schedule = get_linear_fn(start_val, end_val, training_progress_ratio)

lr = 3e-4
num_rollout_steps = 256
gamma = 0.8
gae_lambda = 0.9
n_epochs = 10
seed = 42
batch_size = 512
tensorboard_log = '/workspace/datasets/sb3_tb_logs/' + str(datetime.date.today()) + '/'

################
import wandb
from wandb.integration.sb3 import WandbCallback
import os
# os.environ["WANDB_API_KEY"] = '083592c84134c040dcca598c644c348d32540a08'
# os.environ['WANDB_NOTEBOOK_NAME'] = 'sb3_ppo.py'
# config = {
#     'lr' : 3e-4,
#     'num_rollout_steps' :  256, # 2048
#     'gamma' : 0.8,
#     'gae_lambda' : 0.9,
#     'n_epochs' : 10, # 16, #10
#     'seed' : 42,
#     'batch_size' : 512, # 512
#     }
# hcmTz = pytz.timezone("Asia/Ho_Chi_Minh") 
# date = datetime.datetime.now(hcmTz).strftime("%d-%m-%Y_%H-%M-%S")

# experiment_name = f"PPO_sb3_{date}"

# run = wandb.init(
#     project="l5kit2",
#     sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
#     monitor_gym=False,  # auto-upload the videos of agents playing the game
#     save_code=True,  # optional
#     reinit = True,
#     group = 'PPO',
#     config = config,
#     name = experiment_name,
#     # dir = '/workspace/datasets/'
# #     name = 'PPO_sb3_05-01-2023_15-12-56(3)',
# )
################
# define model
# model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=0, n_steps=num_rollout_steps,
#             learning_rate=lr, gamma=gamma, tensorboard_log=f"runs/{run.id}", n_epochs=n_epochs,
#             clip_range=clip_schedule, batch_size=batch_size, seed=seed, gae_lambda=gae_lambda,)
# define model
model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1, n_steps=num_rollout_steps,
            learning_rate=lr, gamma=gamma, tensorboard_log=tensorboard_log, n_epochs=n_epochs,
            clip_range=clip_schedule, batch_size=batch_size, seed=seed, gae_lambda=gae_lambda)

#######################
# callback_list = []
# hcmTz = pytz.timezone("Asia/Ho_Chi_Minh") 
# date = datetime.datetime.now(hcmTz).strftime("%d-%m-%Y_%H-%M-%S")
# # Save Model Periodically
# save_freq = 10000
# save_path = '/workspace/datasets/sb3_results/debug'+ date
# # save_path = 'l5kit/logs/05-01-2023_15-12-56'
# output = 'PPO'
# checkpoint_callback = CheckpointCallback(save_freq=(save_freq // train_envs), save_path=save_path, \
#                                          name_prefix=output)
# callback_list.append(checkpoint_callback)

# # Eval Model Periodically
# eval_freq = 1000#10000# NOTE: CHANGE THIS
# n_eval_episodes = 1
# val_eval_callback = L5KitEvalCallback(eval_env, eval_freq=(eval_freq // train_envs), \
#                                       n_eval_episodes=n_eval_episodes, n_eval_envs=eval_envs)
# callback_list.append(val_eval_callback)
# callback_list.append(WandbCallback(
#         # gradient_save_freq=100,
#         # model_save_path=f"models/{run.id}",
#         # verbose=2,
#         # model_save_freq=10,
#     ))
#############
callback_list = []

# Save Model Periodically
hcmTz = pytz.timezone("Asia/Ho_Chi_Minh") 
date = datetime.datetime.now(hcmTz).strftime("%d-%m-%Y_%H-%M-%S")
save_freq = 10000
save_path = '/workspace/datasets/sb3_results/debug'+ date

output = 'PPO'
checkpoint_callback = CheckpointCallback(save_freq=(save_freq // train_envs), save_path=save_path, \
                                         name_prefix=output)
callback_list.append(checkpoint_callback)

# Eval Model Periodically
eval_freq = 10000
n_eval_episodes = 1
val_eval_callback = L5KitEvalCallback(eval_env, eval_freq=(eval_freq // train_envs), \
                                      n_eval_episodes=n_eval_episodes, n_eval_envs=eval_envs)
callback_list.append(val_eval_callback)

#############

# model = PPO.load('/workspace/datasets/sb3_results/23-03-2023_14-51-50/PPO_1850000_steps.zip', env = env)
n_steps = 6e6
model.learn(n_steps, callback=callback_list)
# model = PPO.load('./PPO_100000_steps.zip', env = env)
# model.learn(n_steps, callback=callback_list, reset_num_timesteps=False)

run.finish()