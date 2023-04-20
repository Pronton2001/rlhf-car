from typing import Dict
from l5kit.data.map_api import MapAPI
from l5kit.environment.envs.l5_env import GymStepOutput, SimulationConfigGym, L5Env
import gym
from src.constant import SRC_PATH
from src.customEnv import wrapper
import os
from matplotlib import pyplot as plt
import numpy as np
import torch
from l5kit.configs import load_config_data
from l5kit.data import ChunkedDataset, LocalDataManager

os.environ['L5KIT_DATA_FOLDER'] = '/workspace/datasets/'


actions = [np.array([0.33075115, 0.9822087 , 0.35992876], dtype=np.float32), np.array([-0.47357458,  0.45461076,  0.57742184], dtype=np.float32), np.array([ 0.5737099 ,  0.90265733, -0.6825135 ], dtype=np.float32), np.array([-0.6482308 ,  0.1528697 ,  0.22510704], dtype=np.float32), np.array([0.32743016, 0.55001056, 0.2772629 ], dtype=np.float32), np.array([ 0.09946773, -0.8968407 , -0.7836492 ], dtype=np.float32), np.array([-0.73099047, -0.2190136 ,  0.3157143 ], dtype=np.float32), np.array([-0.37622538, -0.98760587,  0.23697242], dtype=np.float32)]

i = 0
def rollout_episode(env, idx = 0):
    """Rollout a particular scene index and return the simulation output.

    :param model: the RL policy
    :param env: the gym environment
    :param idx: the scene index to be rolled out
    :return: the episode output of the rolled out scene
    """

    # Set the reset_scene_id to 'idx'
    env.set_reset_id(idx)
    global i
    
    
    # Rollout step-by-step
    obs = env.reset()
    done = False
    while True:
        #action = np.array(env.action_space.sample())
        #actions.append(action)
        action = actions[i]
        i+=1
        obs, _, done, info = env.step(action)
        if done:
            break

    # The episode outputs are present in the key "sim_outs"
    sim_out = info["sim_outs"][0]
    return sim_out

# from l5kit.visualization.visualizer.zarr_utils import episode_out_to_visualizer_scene_gym_cle, simulation_out_to_visualizer_scene
# from l5kit.visualization.visualizer.visualizer import visualize
# from bokeh.io import output_notebook, show
# python src/test/test_rescale_action.py
# export PYTHONPATH=/workspace/source
sim_outs =[]
rollout_sim_cfg = SimulationConfigGym()
rollout_sim_cfg.num_simulation_steps = 10
env_config_path = SRC_PATH + 'src/configs/gym_config.yaml'

# env_config_path = 
env_kwargs = {'env_config_path': env_config_path, 'use_kinematic': True, 'sim_cfg': rollout_sim_cfg,  'train': True, 'return_info': True}

rollout_env = L5Env(**env_kwargs)

for i in range(1):
    rollout_episode(rollout_env, i)

#print(actions)
# might change with different rasterizer
# mapAPI = MapAPI.from_cfg(dmg, cfg)

# def visualize_outputs(sim_outs, map_API):
#     for sim_out in sim_outs: # for each scene
#         vis_in = episode_out_to_visualizer_scene_gym_cle(sim_out, map_API)
#         show(visualize(sim_out.scene_id, vis_in))

# visualize_outputs(sim_outs, mapAPI)
