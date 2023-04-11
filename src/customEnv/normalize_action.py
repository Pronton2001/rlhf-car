from typing import List, Union
from l5kit.environment.utils import (calculate_non_kinematic_rescale_params, KinematicActionRescaleParams,
                                     NonKinematicActionRescaleParams)

from l5kit.environment.envs.l5_env import SimulationConfigGym, GymStepOutput, L5Env
from l5kit.environment.envs.l5_env2 import L5Env2


import torch
def standard_normalizer(non_kin_rescale , action):

    if type(action) == torch.Tensor:
        scaled_action = action.view(-1,3).clone()
    elif type(action) == np.ndarray:
        scaled_action = action.reshape(-1,3).copy()
    elif type(action) == List:
        scaled_action = np.asarray(action).reshape(-1,3).copy()

    # non_kin_rescale= env.non_kin_rescale
    scaled_action[:,0] = (action[:,0]-non_kin_rescale.x_mu)/non_kin_rescale.x_scale
    scaled_action[:,1] = (action[:,1]-non_kin_rescale.y_mu)/non_kin_rescale.y_scale 
    scaled_action[:,2] = (action[:,2]-non_kin_rescale.yaw_mu)/non_kin_rescale.yaw_scale
    return scaled_action

if __name__ == '__main__':
    import numpy as np
    import os
    os.environ["L5KIT_DATA_FOLDER"] = '/workspace/datasets'

    from l5kit.configs import load_config_data

    env_config_path = '/workspace/source/src/configs/gym_rasterizer_config.yaml'
    cfg = load_config_data(env_config_path)

    n_channels = (cfg['model_params']['history_num_frames'] + 1)* 2 + 3
    print('num channels:', n_channels)
    from ray import tune
    from src.customEnv.wrapper import L5EnvRasterizerTorch
    train_eps_length = 32
    train_sim_cfg = SimulationConfigGym()
    train_sim_cfg.num_simulation_steps = train_eps_length + 1

    env_kwargs = {'env_config_path': env_config_path, 'use_kinematic': False, 'sim_cfg': train_sim_cfg}#, 'rescale_action': False}
    env = L5Env(**env_kwargs)
    print(env.non_kin_rescale)
    l5envRaster = L5EnvRasterizerTorch(env = L5Env(**env_kwargs), raster_size= cfg['raster_params']['raster_size'][0], \
                                                           n_channels = n_channels)
    # action = np.array([1.4,0.02,0.0004])
    # l5envRaster.step
    l5envRaster.reset()
    print(l5envRaster.non_kin_rescale)
    # standard_normalizer(env, action)
    # action = l5envRaster.action_space.sample()
    # print('before action:', action)
    pred_x = torch.ones((32,1)) * 1.4
    pred_y = torch.ones((32,1)) * 0.04
    pred_yaw = torch.ones((32,1)) * 0.0004

    action = torch.cat((pred_x,pred_y, pred_yaw), dim = -1)
    action = standard_normalizer(l5envRaster.non_kin_rescale, action)
    print('before step:', action[0].numpy())
    print(l5envRaster.step(action[0].numpy()))