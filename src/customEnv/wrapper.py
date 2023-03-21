from typing import Dict
import gym
from l5kit.environment.envs.l5_env import GymStepOutput
import numpy as np
import matplotlib.pyplot as plt


class L5EnvWrapper(gym.Wrapper):
    def __init__(self, env, raster_size = 112, n_channels = 7):
        super().__init__(env)
        self.env = env
        self.n_channels = n_channels
        self.raster_size = raster_size
        obs_shape = (self.raster_size, self.raster_size, self.n_channels)
        self.observation_space =gym.spaces.Box(low=0, high=1, shape=obs_shape, dtype=np.float32)

    def step(self, action:  np.ndarray) -> GymStepOutput:
        # return GymStepOutput(obs, reward["total"], done, info)
        output =  self.env.step(action)
        # print(np.array(output.obs['image']).shape)
        onlyImageState = output.obs['image'].reshape(self.raster_size, self.raster_size, self.n_channels)
        # onlyImageState = output.obs['image']
        # plt.imshow(onlyImageState[:,:,0])
        # plt.imshow(onlyImageState[2])
        # plt.show()
        return GymStepOutput(onlyImageState, output.reward, output.done, output.info)

    def reset(self) -> Dict[str, np.ndarray]:
        return self.env.reset()['image'].reshape(self.raster_size, self.raster_size, self.n_channels)
        return self.env.reset()['image']