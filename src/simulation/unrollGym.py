from l5kit.data import LocalDataManager
from l5kit.data.map_api import MapAPI
from l5kit.visualization.visualizer.zarr_utils import episode_out_to_visualizer_scene_gym_cle
from l5kit.configs import load_config_data

dmg = LocalDataManager(None)
source_path = '/workspace/source/'
env_config_path = source_path + 'src/configs/gym_config.yaml'
cfg = load_config_data(env_config_path)
mapAPI = MapAPI.from_cfg(dmg, cfg)

def rollout_episode(model, env, idx = 0):
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
        while True:
            # action, _ = model.predict(obs, deterministic=True)
            action = model.compute_single_action(obs, deterministic=True)
            obs, _, done, info = env.step(action)
            if done:
                break

        # The episode outputs are present in the key "sim_outs"
        sim_out = info["sim_outs"][0]
        return sim_out

def unroll(model, rollout_env,num):
    # Rollout one episode
    # sim_out = rollout_episode(model, rollout_env)
    # Rollout 5 episodes
    sim_outs =[]
    for i in range(num):
        sim_outs.append(rollout_episode(model, rollout_env, i))
    return sim_outs

def visualize_outputs(sim_outs):
    for sim_out in sim_outs: # for each scene
        vis_in = episode_out_to_visualizer_scene_gym_cle(sim_out, mapAPI)
        show(visualize(sim_out.scene_id, vis_in))