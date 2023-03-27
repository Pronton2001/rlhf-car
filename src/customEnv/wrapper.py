from typing import Dict
from gym import Wrapper, spaces
from l5kit.environment.envs.l5_env import GymStepOutput
import numpy as np
import matplotlib.pyplot as plt
from src.pref.test_pairwise import RewardModel
from torch import load


class L5EnvWrapper(Wrapper):
    def __init__(self, env, raster_size = 112, n_channels = 7):
        super().__init__(env)
        self.env = env
        self.n_channels = n_channels
        self.raster_size = raster_size
        obs_shape = (self.raster_size, self.raster_size, self.n_channels)
        self.observation_space =spaces.Box(low=0, high=1, shape=obs_shape, dtype=np.float32)
        # self.action_space =gym.spaces.Box(low=0, high=1, shape=obs_shape, dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3, ))

    def step(self, action:  np.ndarray) -> GymStepOutput:
        output =  self.env.step(action)
        onlyImageState = output.obs['image'].reshape(self.raster_size, self.raster_size, self.n_channels)
        assert onlyImageState.shape[-1] < onlyImageState.shape[0], f'wrong shape: {onlyImageState.shape}'
        return GymStepOutput(onlyImageState, output.reward, output.done, output.info)# NOTE: For SAC,PPO ray rllib policy

    def reset(self) -> Dict[str, np.ndarray]:
        return self.env.reset()['image'].reshape(self.raster_size, self.raster_size, self.n_channels) # : For SAC,PPO ray rllib policy

class L5EnvWrapperHFreward(Wrapper):# TODO - Code Unit test for this wrapper
    '''Change l5kit reward to preferenced-based reward'''
    def __init__(self, env, raster_size = 112, n_channels = 7, kwargs = dict(state_shape=(84, 84, 7), action_shape=(3,)), RWmodel_path = 'src/pref/model/model.pt'):
        super().__init__(env)
        self.env = env
        self.n_channels = n_channels
        self.raster_size = raster_size
        obs_shape = (self.raster_size, self.raster_size, self.n_channels)
        self.observation_space =spaces.Box(low=0, high=1, shape=obs_shape, dtype=np.float32)
        # self.action_space =gym.spaces.Box(low=0, high=1, shape=obs_shape, dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3, ))
        # kwargs = dict(state_shape=(84, 84, 7), action_shape=(3,))
        self.rewardModel = RewardModel(**kwargs)
        self.rewardModel.load_state_dict(load(RWmodel_path))

    def step(self, action:  np.ndarray) -> GymStepOutput: 
        output =  self.env.step(action)
        obs_ = output.obs['image'].reshape(self.raster_size, self.raster_size, self.n_channels)
        assert obs_.shape[-1] < obs_.shape[0], f'wrong shape: {obs_.shape}'
        info = output.info
        
        action_dict = self.env.ego_output_dict
        actions = np.concatenate((action_dict['positions'][0][0], action_dict['yaws'][0][0]))
        pred = self.rewardModel(obs_.reshape(7, self.raster_size, self.raster_size), actions)
        if 'sim_outs' in info.keys(): # episode done
            info_ = {"sim_outs": info["sim_outs"], "reward_tot": pred}

        info_ = {'reward_tot': pred}
        reward_ = pred
        return GymStepOutput(obs_, reward_, output.done, info_)# NOTE: For SAC,PPO ray rllib policy

    def reset(self) -> Dict[str, np.ndarray]:
        return self.env.reset()['image'].reshape(self.raster_size, self.raster_size, self.n_channels) # : For SAC,PPO ray rllib policy

class L5EnvWrapperWithoutReshape(Wrapper): # use transpose instead of reshape
    def __init__(self, env, raster_size = 112, n_channels = 7):
        super().__init__(env)
        self.env = env
        self.n_channels = n_channels
        self.raster_size = raster_size
        obs_shape = (self.raster_size, self.raster_size, self.n_channels)
        self.observation_space =spaces.Box(low=0, high=1, shape=obs_shape, dtype=np.float32)
        # self.action_space =gym.spaces.Box(low=0, high=1, shape=obs_shape, dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3, ))

    def step(self, action:  np.ndarray) -> GymStepOutput:
        output =  self.env.step(action)
        onlyImageState = output.obs['image'].transpose(1,2,0) # C,W,H -> W, H, C (for SAC/PPO torch model)
        
        assert onlyImageState.shape[-1] < onlyImageState.shape[1], f'wrong shape: {onlyImageState.shape}'
        return GymStepOutput(onlyImageState, output.reward, output.done, output.info)

    def reset(self) -> Dict[str, np.ndarray]:
        return self.env.reset()['image'].transpose(1,2,0) # C,W,H -> W, H, C (for SAC/PPO torch model)

if __name__ == '__main__':
    import os
    dataset_path = "/workspace/datasets/"
    source_path = "/workspace/source/"
    dataset_path = '/media/pronton/linux_files/a100code/l5kit/l5kit_dataset/'
    source_path = "/home/pronton/rl/rlhf-car/"
    os.environ["L5KIT_DATA_FOLDER"] = dataset_path
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
    from l5kit.environment.envs.l5_env import SimulationConfigGym, L5Env
    from l5kit.configs import load_config_data

    env_config_path = 'src/configs/gym_config_cpu.yaml'
    cfg = load_config_data(env_config_path)
    from ray import tune
    rollout_sim_cfg = SimulationConfigGym()
    rollout_sim_cfg.num_simulation_steps = None

    env_kwargs = {'env_config_path': env_config_path, 
                'use_kinematic': True, 
                'sim_cfg': rollout_sim_cfg,  
                'train': False, 
                'return_info': True}

    rollout_env = L5EnvWrapper(env = L5Env(**env_kwargs), \
                            raster_size= cfg['raster_params']['raster_size'][0], \
                            n_channels = 7,)
    tune.register_env("L5-CLE-V2", 
                    lambda config: L5EnvWrapper(env = L5Env(**env_kwargs), \
                                                raster_size= cfg['raster_params']['raster_size'][0], \
                                                n_channels = 7))
    rollout_env2 = L5EnvWrapperWithoutReshape(env = L5Env(**env_kwargs), \
                            raster_size= cfg['raster_params']['raster_size'][0], \
                            n_channels = 7,)
    tune.register_env("L5-CLE-V2", 
                    lambda config: L5EnvWrapperWithoutReshape(env = L5Env(**env_kwargs), \
                                                raster_size= cfg['raster_params']['raster_size'][0], \
                                                n_channels = 7))
    print(type(rollout_env) == L5EnvWrapper)
    from ray.rllib.algorithms.sac import SAC

    raster_size = cfg['raster_params']['raster_size'][0]
    traj1 = []
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
            i+=1
            if i % jump == 0:
                assert obs.shape == (raster_size, raster_size, 7),f'{(raster_size,raster_size,7)} != + {obs.shape})'
                if type(env)== L5EnvWrapper:
                    im = obs.reshape(7, raster_size, raster_size) # reshape again
                    plt.imshow(im[2])
                    plt.show()
                elif type(env) == L5EnvWrapperWithoutReshape:
                    im = obs.transpose(2,0,1) # 
                    plt.imshow(im[2])
                    plt.show()
                traj1.append([im, action])
            obs, _, done, info = env.step(action)
            if done:
                break

        # The episode outputs are present in the key "sim_outs"
        sim_out = info["sim_outs"][0]
        return sim_out
    from ray.rllib.algorithms.sac import SAC
    # checkpoint_path = 'l5kit/ray_results/01-01-2023_15-53-49/SAC/SAC_L5-CLE-V1_cf7bb_00000_0_2023-01-01_08-53-50/checkpoint_000170'
    checkpoint_path = dataset_path + 'ray_results/31-12-2022_07-53-04/SAC/SAC_L5-CLE-V1_7bae1_00000_0_2022-12-31_00-53-04/checkpoint_000360'
    model = SAC(config=config_param_space, env='L5-CLE-V2')
    model.restore(checkpoint_path)
    rollout_episode_rllib (model, rollout_env2)
