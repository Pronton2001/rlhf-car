
from l5kit.environment.envs.l5_env import SimulationConfigGym
from l5kit.environment.envs.l5_env2 import L5Env2

import os
os.environ["L5KIT_DATA_FOLDER"] = "/media/pronton/linux_files/a100code/l5kit/l5kit_dataset"

env_config_path = '/home/pronton/rl/rlhf-car/src/configs/gym_vectorizer_config.yaml'

rollout_sim_cfg = SimulationConfigGym()
rollout_sim_cfg.num_simulation_steps = None
env_kwargs = {'env_config_path': env_config_path, 'use_kinematic': True, 'sim_cfg': rollout_sim_cfg,  'train': False, 'return_info': True}
rollout_env = L5Env2(**env_kwargs)