from typing import List, Union
from l5kit.environment.kinematic_model import UnicycleModel
from l5kit.environment.utils import (calculate_non_kinematic_rescale_params, KinematicActionRescaleParams,
                                     NonKinematicActionRescaleParams)

from l5kit.environment.envs.l5_env import SimulationConfigGym, GymStepOutput, L5Env
from l5kit.environment.envs.l5_env2 import L5Env2
import numpy as np


import torch
import logging

def standard_normalizer_nonKin(actions, non_kin_rescale =NonKinematicActionRescaleParams(x_mu=0.47771242, x_scale=5.093913674354553, y_mu=0.0013034682, y_scale=0.08574443869292736, yaw_mu=-0.0010775143, yaw_scale=0.03146977396681905)):
    if type(actions) == torch.Tensor:
        scaled_action = actions.view(-1,3).clone()
    elif type(actions) == np.ndarray:
        scaled_action = actions.reshape(-1,3).copy()
    elif type(actions) == List:
        scaled_action = np.asarray(actions).reshape(-1,3).copy()
    else:
        scaled_action = actions = np.asarray(actions).reshape(-1,3).copy()

    # non_kin_rescale= env.non_kin_rescale
    scaled_action[:,0] = (actions[:,0]-non_kin_rescale.x_mu)/non_kin_rescale.x_scale
    scaled_action[:,1] = (actions[:,1]-non_kin_rescale.y_mu)/non_kin_rescale.y_scale 
    scaled_action[:,2] = (actions[:,2]-non_kin_rescale.yaw_mu)/non_kin_rescale.yaw_scale
    return scaled_action

def standard_normalizer_kin(actions, kin_rescale =KinematicActionRescaleParams(steer_scale=0.07853981633974483, acc_scale=0.6000000000000001)):
    if type(actions) == torch.Tensor:
        scaled_action = actions.view(-1,2).clone()
    elif type(actions) == np.ndarray:
        scaled_action = actions.reshape(-1,2).copy()
    elif type(actions) == List:
        scaled_action = np.asarray(actions).reshape(-1,2).copy()
    # else:
    #     actions = np.asarray(actions).reshape(-1,2)
    #     scaled_action = np.empty_like(actions)

    # non_kin_rescale= env.non_kin_rescale
    scaled_action[:,0] = actions[:,0] / kin_rescale.steer_scale
    scaled_action[:,1] = actions[:,1] / kin_rescale.acc_scale
    return scaled_action

import math
def inverseUnicycle(x, y, yaw, old_v):
    min_acc = -0.6  # min acceleration: -6 mps2
    max_acc = 0.6   # max acceleration: 6 mps2
    min_steer = -math.radians(45) * 0.1  # max yaw rate: 45 degrees per second
    max_steer = math.radians(45) * 0.1   # max yaw rate: 45 degrees per second

    steer = yaw
    acc = torch.sign(x * torch.cos(yaw)) * torch.sqrt(torch.pow(x, 2) +  torch.pow(y,2)) - old_v
    # acc = torch.sqrt(torch.pow(x, 2) +  torch.pow(y,2)) - old_v
    # logging.debug(f'x, y, yaw, old_v: {x}, {y}, {yaw}, {old_v}')
    # logging.debug(f'new_v: {torch.sign(x) * torch.sign(torch.cos(yaw)) * torch.sqrt(torch.pow(x, 2) +  torch.pow(y,2)) }')
    # logging.debug(f'acc = new_v - old_v: {acc}')
    steer = torch.clip(steer, min = min_steer, max = max_steer)
    acc = torch.clip(acc, min = min_acc, max = max_acc)

    return torch.cat((steer, acc), dim = -1)



if __name__ == '__main__':
    # import numpy as np
    # import os
    # os.environ["L5KIT_DATA_FOLDER"] = '/workspace/datasets'

    # from l5kit.configs import load_config_data

    # env_config_path = '/workspace/source/src/configs/gym_rasterizer_config.yaml'
    # cfg = load_config_data(env_config_path)

    # n_channels = (cfg['model_params']['history_num_frames'] + 1)* 2 + 3
    # print('num channels:', n_channels)
    # from ray import tune
    # from src.customEnv.wrapper import L5EnvRasterizerTorch
    # train_eps_length = 32
    # train_sim_cfg = SimulationConfigGym()
    # train_sim_cfg.num_simulation_steps = train_eps_length + 1

    # env_kwargs = {'env_config_path': env_config_path, 'use_kinematic': False, 'sim_cfg': train_sim_cfg}#, 'rescale_action': False}
    # env = L5Env(**env_kwargs)
    # print(env.non_kin_rescale)
    # l5envRaster = L5EnvRasterizerTorch(env = L5Env(**env_kwargs), raster_size= cfg['raster_params']['raster_size'][0], \
    #                                                        n_channels = n_channels)
    # # action = np.array([1.4,0.02,0.0004])
    # # l5envRaster.step
    # l5envRaster.reset()
    # print(l5envRaster.non_kin_rescale)
    # # standard_normalizer(env, action)
    # # action = l5envRaster.action_space.sample()
    # # print('before action:', action)
    # pred_x = torch.ones((32,1)) * 1.4
    # pred_y = torch.ones((32,1)) * 0.04
    # pred_yaw = torch.ones((32,1)) * 0.0004

    # action = torch.cat((pred_x,pred_y, pred_yaw), dim = -1)
    # action = standard_normalizer(l5envRaster.non_kin_rescale, action)
    # print('before step:', action[0].numpy())
    # print(l5envRaster.step(action[0].numpy()))

    kin_model = UnicycleModel()
    old_v = 0.1 * 2
    init_kin_state = np.array([0.0, 0.0, 0.0, old_v])
    kin_model.reset(init_kin_state)
    steer, acc = -0.004, -0.2
    data_dict = kin_model.update([steer, acc])
    print(data_dict)
    x, y =data_dict['positions'][0][0]
    yaw =data_dict['yaws'][0][0]
    print(inverseUnicycle((x,x), [y,y], [yaw,yaw], [old_v,old_v]))
    
