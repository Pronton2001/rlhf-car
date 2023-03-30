import json
from multiprocessing import Process
import threading
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from prettytable import PrettyTable

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset

from l5kit.dataset import EgoDatasetVectorized, EgoDataset
from l5kit.vectorization.vectorizer_builder import build_vectorizer
from l5kit.rasterization.rasterizer_builder import build_rasterizer
from l5kit.simulation.dataset import SimulationConfig
from l5kit.simulation.unroll import ClosedLoopSimulator

from l5kit.visualization.visualizer.visualizer import visualize, visualize2, visualize3, visualize4
from bokeh.io import output_notebook, show
from l5kit.data import MapAPI
from bokeh.models import Button
from l5kit.environment.envs.l5_env import GymStepOutput, SimulationConfigGym, L5Env

from src.customEnv.wrapper import L5EnvWrapper, L5EnvWrapperHFreward, L5EnvWrapperWithoutReshape

import os
from pref_db import PrefDB
from numpy.testing import assert_equal


# set env variable for data

dataset_path = "/workspace/datasets/"
source_path = "/workspace/source/"
dataset_path = '/mnt/datasets/'
source_path = "/home/pronton/rl/rlhf-car/"
os.environ["L5KIT_DATA_FOLDER"] = dataset_path
dm = LocalDataManager(None)
# get config
cfg = load_config_data(source_path + 'src/configs/gym_config84.yaml')

####################################################
import gym

from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.vec_env import SubprocVecEnv

from l5kit.configs import load_config_data
from l5kit.environment.feature_extractor import CustomFeatureExtractor
from l5kit.environment.callbacks import L5KitEvalCallback
from l5kit.environment.envs.l5_env import SimulationConfigGym

from l5kit.visualization.visualizer.zarr_utils import episode_out_to_visualizer_scene_gym_cle, zarr_to_visualizer_scene
from l5kit.visualization.visualizer.visualizer import visualize
from bokeh.layouts import column, LayoutDOM, row, gridplot
from bokeh.io import curdoc

MODEL= 'SB3 MODEL'
MODEL= 'RLLIB MODEL'

# Dataset is assumed to be on the folder specified
# in the L5KIT_DATA_FOLDER environment variable

# get environment config
env_config_path = 'src/configs/gym_config84.yaml'
cfg = load_config_data(env_config_path)
# Train on episodes of length 32 time steps
train_eps_length = 32
train_envs = 4

# make train env
# Stable baselines 3
def sb3_model():#FIXME - AttributeError: 'Box' object has no attribute 'low_repr' -> install stable-baselines-3 1.7
    train_sim_cfg = SimulationConfigGym()
    train_sim_cfg.num_simulation_steps = train_eps_length + 1
    env_kwargs = {'env_config_path': env_config_path, 'use_kinematic': True, 'sim_cfg': train_sim_cfg, 'train': False, 'return_info': True,}
    env = make_vec_env("L5-CLE-v0", env_kwargs=env_kwargs, n_envs=train_envs,
                    vec_env_cls=SubprocVecEnv, vec_env_kwargs={"start_method": "fork"})

    # make train env
    modelA = SAC.load(dataset_path +'logs/SAC_640000_steps.zip', env = env
                    , custom_objects = {
                "learning_rate": 0.0,
                "lr_schedule": lambda _: 0.0,
                "clip_range": lambda _: 0.0,
            }
            )
    rollout_sim_cfg = SimulationConfigGym()
    rollout_sim_cfg.num_simulation_steps = 20
    rollout_env = gym.make("L5-CLE-v0", env_config_path=env_config_path, sim_cfg=rollout_sim_cfg, \
                        use_kinematic=True, train=True, return_info=True)
    return rollout_env, modelA
# rollout_env, modelA = sb3_model()
def rllib_model():
    train_envs = 4
    lr = 3e-3
    lr_start = 3e-4
    lr_end = 3e-5
    config_param_space = {
        "env": "L5-CLE-V1",
        "framework": "torch",
        "num_gpus": 0,
        # "num_workers": 63,
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
            'capacity': int(1e5),
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
        'train_batch_size': 2048,
        'training_intensity' : 32, # (4x 'natural' value = 8)
        'gamma': 0.8,
        'twin_q' : True,
        "lr": 3e-4,
        "min_sample_timesteps_per_iteration": 8000,
    }
    from ray import tune
    rollout_sim_cfg = SimulationConfigGym()
    rollout_sim_cfg.num_simulation_steps = None

    env_kwargs = {'env_config_path': env_config_path, 
                'use_kinematic': True, 
                'sim_cfg': rollout_sim_cfg,  
                'train': True, 
                'return_info': True}

    rollout_env = L5EnvWrapper(env = L5Env(**env_kwargs), \
                            raster_size= cfg['raster_params']['raster_size'][0], \
                            n_channels = 7,)
    tune.register_env("L5-CLE-V2", 
                    lambda config: L5EnvWrapper(env = L5Env(**env_kwargs), \
                                                raster_size= cfg['raster_params']['raster_size'][0], \
                                                n_channels = 7))
    from ray.rllib.algorithms.sac import SAC
    # checkpoint_path = 'l5kit/ray_results/01-01-2023_15-53-49/SAC/SAC_L5-CLE-V1_cf7bb_00000_0_2023-01-01_08-53-50/checkpoint_000170'
    checkpoint_path = dataset_path + 'ray_results/31-12-2022_07-53-04/SAC/SAC_L5-CLE-V1_7bae1_00000_0_2022-12-31_00-53-04/checkpoint_000360'
    algo = SAC(config=config_param_space, env='L5-CLE-V2')
    algo.restore(checkpoint_path)
    return rollout_env, algo
traj1 = []
raster_size = cfg['raster_params']['raster_size'][0]
def rollout_episode(model, env, idx = 0, jump = 10):
    """Rollout a particular scene index and return the simulation output.

    :param model: the RL policy
    :param env: the gym environment
    :param idx: the scene index to be rolled out
    :return: the episode output of the rolled out scene
    """

    # Set the reset_scene_id to 'idx'
    env.set_reset_id(idx)
    
    # Rollout step-by-step
    obs = env.reset()
    done = False
    i = 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        if i % jump == 0:
            assert obs.shape == (raster_size, raster_size, 7),f'{(raster_size,raster_size,7)} != + {obs.shape})'
            if type(env)== L5EnvWrapper:
                im = obs.reshape(7, raster_size, raster_size) # reshape again
                # plt.imshow(im[2])
                # plt.show()
            elif type(env) == L5EnvWrapperWithoutReshape:
                im = obs.transpose(2,0,1) # 
                # plt.imshow(im[2])
                # plt.show()
            traj1.append([im, action])
        i+=1
        obs, _, done, info = env.step(action)
        if done:
            break

    # The episode outputs are present in the key "sim_outs"
    sim_out = info["sim_outs"][0]
    return sim_out

def rollout_episode_rllib(model, env, idx = 0, jump = 10):
    """Rollout a particular scene index and return the simulation output.

    :param model: the RL policy
    :param env: the gym environment
    :param idx: the scene index to be rolled out
    :return: the episode output of the rolled out scene
    """

    # Set the reset_scene_id to 'idx'
    env.set_reset_id(idx)
    
    # Rollout step-by-step
    obs = env.reset()
    done = False
    i = 0
    while True:
        action = model.compute_single_action(obs, deterministic=True)
        assert obs.shape == (84,84,7), f'error shape {obs.shape}'  # SAC: use  per
        if i % jump == 0: # 0, jump, 2*jump,...
            assert obs.shape == (raster_size, raster_size, 7),f'{(raster_size,raster_size,7)} != + {obs.shape})'
            if type(env) in [L5EnvWrapper, L5EnvWrapperHFreward]:
                im = obs.reshape(7, raster_size, raster_size) # reshape again
                # plt.imshow(im[2])
                # plt.show()
                obs, _, done, info = env.step(action) # action -> unscale action
                # print('output:', env.ego_output_dict)
                action_dict = env.ego_output_dict
                actions = np.concatenate((action_dict['positions'][0][0], action_dict['yaws'][0][0]))
                assert_equal(len(actions), 3) # x, y, yaw
            elif type(env) == L5EnvWrapperWithoutReshape:
                im = obs.transpose(2,0,1) # 
                # plt.imshow(im[2])
                # plt.show()
                obs, _, done, info = env.step(action) # action -> unscale action
                # print('output:', env.ego_output_dict)
                action_dict = env.ego_output_dict
                actions = np.concatenate((action_dict['positions'][0][0], action_dict['yaws'][0][0]))
                assert len(actions) == 3, f'len(action_list) != 3, action_list = {actions}' # x, y, yaw
            traj1.append([im, actions])
            # print(actions)
        else:
            obs, _, done, info = env.step(action)
            # print('output:', rollout_env.ego_output_dict)
        i+=1
        if done:
            break

    # The episode outputs are present in the key "sim_outs"
    sim_out = info["sim_outs"][0]
    return sim_out

rast = build_rasterizer(cfg, dm)
zarr_dataset = ChunkedDataset(dm.require(cfg["train_data_loader"]["key"])).open() #TODO: should load 1 time only
dataset = EgoDataset(cfg, zarr_dataset, rast)
traj2 = []
def getImg(scene_idx, jump= 10):
    global traj2
    indexes = dataset.get_scene_indices(scene_idx)
    # print(len(indexes), scene_idx, indexes[::jump], jump)
    # start_idx = indexes[0]
    for i in indexes[::jump]: #0, jump, 2*jump,...
        # assert (i-start_idx) % jump == 0, f'wrong idx: {i}'
        data = dataset[i]
        # print(data.keys())
        target_positions, target_yaws = data['target_positions'], data['target_yaws']
        # just use future target pos for 
        # print(target_positions[0])
        # print(target_yaws[0])
        # im = data["image"].transpose(1, 2, 0) #  num_channels, size, size -> size, size, num_channels
        im = data["image"]
        assert im.shape == (7, raster_size, raster_size), f'shape is wrong, {im.shape} != {(7, raster_size, raster_size)}'
        # im = dataset.rasterizer.to_rgb(im)
        # plt.imshow(data["image"][2])
        # plt.show()
        actions = np.concatenate((data['target_positions'][0], data['target_yaws'][0]))
        # print(action_list, data['target_positions'], data['target_yaws'], data['target_positions'][0], data['target_yaws'][0])
        assert len(actions) == 3, f'len(action_list) != 3, action_list = {actions} ' # x, y, yaw
        traj2.append([im, actions])
        # print(actions)
        # traj must scale by rescale 
# define the callback function
def button_callback(button):
    button_name = button.label
    print(f"The '{button_name}' button was clicked")
    if button_name == 'Left':
        pref = [1.0, 0.0]
    elif button_name == 'Right':
        pref = [0.0, 1.0]
    elif button_name == 'Same':
        pref = [0.5, 0.5]
    else:
        pref = None

    wait_function(pref)

pref_db = PrefDB(maxlen=5)
PREFLOGDIR = 'src/pref/preferences/'
directory = 'src/pref/preferences/'

scene_indices = []
idx = 0 
for filename in os.listdir(directory):
    num = filename.split('.')[0]
    try:
        scene_indices.append(int(num))
    except Exception as e:
        print(e)
        continue
if len(scene_indices) == 0:
    idx = 0 # Original index
else:
    print('labeled scene indices:', scene_indices)
    idx = max(scene_indices) + 1 # Next index    
#idx = 83


doc_demo = curdoc()

if MODEL == 'RLLIB MODEL':
    rollout_env, modelA = rllib_model()
else:
    rollout_env, modelA = sb3_model()

# define the wait function
def wait_function(pref):
    global pref_db, idx, traj1, traj2
    '''this function store pref.json (disk storage)
    pref.json:
    t1: [(s0,a0), (s1,a1),...] , t2: [(s0,a0),(s1,a1),...] pref
    '''
    if not pref:
        doc_demo.clear()
        print(doc_demo.session_callbacks)
        for cb in doc_demo.session_callbacks:
            doc_demo.remove_periodic_callback(cb)
        print(doc_demo.session_callbacks)
        idx = idx + 1
        traj1, traj2 = [], []
        PrefInterface(idx)
        return
    print('traj1', len(traj1))
    print('traj2', len(traj2))
    pref_db.append(traj1, traj2, pref)
    traj1, traj2 = [], []
    if len(pref_db) >= pref_db.maxlen:
        pref_db.save(PREFLOGDIR + str(idx + 1) + '.pkl.gz')
        print('saved')
        for _ in range(len(pref_db)): # del all
            pref_db.del_first()
    doc_demo.clear()
    print(doc_demo.session_callbacks)
    for cb in doc_demo.session_callbacks:
        doc_demo.remove_periodic_callback(cb)
    print(doc_demo.session_callbacks)
    idx = idx + 1
    PrefInterface(idx)

# def save_prefs(pref_db_train, pref_db_val, log_dir = 'src/pref/preferences'):
#     train_path = os.path.join(log_dir, 'train.pkl.gz')
#     pref_db_train.save(train_path)
#     print("Saved training preferences to '{}'".format(train_path))
#     val_path = os.path.join(log_dir, 'val.pkl.gz')
#     pref_db_val.save(val_path)
#     print("Saved validation preferences to '{}'".format(val_path))

# Define the buttons
left_button = Button(label="Left", button_type="success")
right_button = Button(label="Right", button_type="success")
cannot_tell_button = Button(label="Can't tell", button_type="warning")
same_button = Button(label="Same", button_type="danger")


# Attach the callbacks to the buttons
left_button.on_click(lambda: button_callback(left_button))
right_button.on_click(lambda: button_callback(right_button))
cannot_tell_button.on_click(lambda: button_callback(cannot_tell_button))
same_button.on_click(lambda: button_callback(same_button))
pref_buttons = row(left_button, column(same_button, cannot_tell_button), right_button)

doc_buttons = curdoc()

mapAPI = MapAPI.from_cfg(dm, cfg)

jump = 10
def PrefInterface(scene_idx):
    start_time = time.time()
    # sac_out = rollout_episode(modelA, rollout_env, scene_idx)
    if MODEL=='RLLIB MODEL':
        sac_out = rollout_episode_rllib(modelA, rollout_env, scene_idx, jump)
    else:
        sac_out = rollout_episode(modelA, rollout_env, scene_idx, jump)
    vis_in = episode_out_to_visualizer_scene_gym_cle(sac_out, mapAPI)[::2]
    v1 = visualize4(scene_idx, vis_in, doc_demo, 'left')
    print(time.time() - start_time)
    start_time = time.time()
    vis_human_in = zarr_to_visualizer_scene(zarr_dataset.get_scene_dataset(scene_idx), mapAPI)[::2]
    getImg(scene_idx, jump)
    v2 = visualize4(scene_idx, vis_human_in, doc_demo, 'right')
    print(time.time() - start_time)
    # doc_demo.add_root(column(row(v1,v2), pref_buttons))

    doc_demo.add_root(row(v1, v2))
    doc_buttons.add_root(pref_buttons)

PrefInterface(idx)