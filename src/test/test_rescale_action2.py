from collections import defaultdict
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, List, NamedTuple, Optional

import gym
import numpy as np
import torch
from gym import spaces
from gym.utils import seeding

from l5kit.configs import load_config_data
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoDataset
from l5kit.environment.kinematic_model import KinematicModel, UnicycleModel
from l5kit.environment.reward import L2DisplacementYawReward, Reward
from l5kit.environment.utils import (calculate_non_kinematic_rescale_params, KinematicActionRescaleParams,
                                     NonKinematicActionRescaleParams)
from l5kit.rasterization import build_rasterizer
from l5kit.simulation.dataset import SimulationConfig, SimulationDataset
from l5kit.simulation.unroll import (ClosedLoopSimulator, ClosedLoopSimulatorModes, SimulationOutputCLE,
                                     UnrollInputOutput)


#: Maximum acceleration magnitude for kinematic model
MAX_ACC = 6
#: Maximum steer magnitude for kinematic model
MAX_STEER = math.radians(45)


_rescale_action = True
init_kin_state = np.array([0.0, 0.0, 0.0, step_time * ego_input[0]['curr_speed']])
kin_model.reset(init_kin_state)
use_kinematic = TrueFalse
kin_model =KinematicActionRescaleParams(MAX_STEER * step_time, MAX_ACC * self.step_time)

def _rescale_action(action: np.ndarray) -> np.ndarray:
    """Rescale the input action back to the un-normalized action space. PPO and related algorithms work well
    with normalized action spaces. The environment receives a normalized action and we un-normalize it back to
    the original action space for environment updates.

    :param action: the normalized action
    :return: the unnormalized action
    """
    newAction = action.copy()
    if rescale_action:
        if use_kinematic:
            newAction[0] = kin_rescale.steer_scale * action[0]
            newAction[1] = kin_rescale.acc_scale * action[1]
        else:
            newAction[0] = non_kin_rescale.x_mu + non_kin_rescale.x_scale * action[0]
            newAction[1] = non_kin_rescale.y_mu + non_kin_rescale.y_scale * action[1]
            newAction[2] = non_kin_rescale.yaw_mu + non_kin_rescale.yaw_scale * action[2]     
    # print('rescale:' + str(action) + '->' + str(newAction))
    # assert abs(newAction[0]) > 1e-12 and abs(newAction[1])> 1e-12, 'wrong action: '+ str(newAction)+ ', prev action:' + str(action)
    return newAction
def _rescale_action(action: np.ndarray) -> np.ndarray:
    """Rescale the input action back to the un-normalized action space. PPO and related algorithms work well
    with normalized action spaces. The environment receives a normalized action and we un-normalize it back to
    the original action space for environment updates.

    :param action: the normalized action
    :return: the unnormalized action
    """
    if rescale_action:
        if use_kinematic:
            newAction = [0,0]
            newAction[0] = kin_rescale.steer_scale * action[0]
            newAction[1] = kin_rescale.acc_scale * action[1]
        else:
            action[0] = non_kin_rescale.x_mu + non_kin_rescale.x_scale * action[0]
            action[1] = non_kin_rescale.y_mu + non_kin_rescale.y_scale * action[1]
            action[2] = non_kin_rescale.yaw_mu + non_kin_rescale.yaw_scale * action[2]
    return newAction