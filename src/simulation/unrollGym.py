from l5kit.data import LocalDataManager
from l5kit.data.map_api import MapAPI
from l5kit.visualization.visualizer.zarr_utils import episode_out_to_visualizer_scene_gym_cle, simulation_out_to_visualizer_scene
from l5kit.configs import load_config_data
from bokeh.io import show
from l5kit.visualization.visualizer.visualizer import visualize
import torch
dmg = LocalDataManager(None)
from src.constant import SRC_PATH
env_config_path = f'{SRC_PATH}src/configs/gym_config.yaml'
cfg = load_config_data(env_config_path)
mapAPI = MapAPI.from_cfg(dmg, cfg)
from src.validate.validator import quantitative

from l5kit.environment.envs.l5_env2 import MAX_ACC, MAX_STEER
step_time = 0.1
steer_scale = MAX_STEER * step_time
acc_scale =   MAX_ACC * step_time 
# def rollout_episode(model, env, idx = 0, num_simulation_steps = None):
#         """Rollout a particular scene index and return the simulation output.

#         :param model: the RL policy
#         :param env: the gym environment
#         :param idx: the scene index to be rolled out
#         :return: the episode output of the rolled out scene
#         """

#         # Set the reset_scene_id to 'idx'
#         env.set_reset_id(idx)

#         # Rollout step-by-step
#         obs = env.reset()
#         done = False
#         actions = []
#         while True:
#             # action, _ = model.predict(obs, deterministic=True)
#             action = model.compute_single_action(obs, explore=False)
#             # rescale actions
#             action[0] = steer_scale * action[0]
#             action[1] = acc_scale * action[1]

#             actions.append(action)
#             obs, _, done, info = env.step(action)
#             if num_simulation_steps and idx >= num_simulation_steps:
#                 done = True
#             if done:
#                 break

#         # The episode outputs are present in the key "sim_outs"
#         sim_out = info["sim_outs"][0]
#         return sim_out, actions
        
from src.customEnv.action_utils import inverseUnicycle, standard_normalizer_nonKin, standard_normalizer_kin
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'


def rollout_episode(model, env, idx = 0,num_simulation_steps = None,  model_type = 'OPENED_LOOP', use_kin = True):
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
    actions = []
    while True:
        # print(obs)
        # action = np.array(env.action_space.sample())
        if model_type == 'CLOSED_LOOP':
            action = model.compute_single_action(obs, explore = False)
            newAction = action.copy()
            newAction[0] = steer_scale * action[0]
            newAction[1] = acc_scale * action[1]
            actions.append(newAction)

        elif model_type == 'OPENED_LOOP':
            if type(obs) == dict:
                obs = {k: torch.as_tensor(v).view(1, *torch.as_tensor(v).shape).to(device) for k, v in obs.items()}
            elif type(obs) == np.ndarray:
                obs = torch.as_tensor(obs).view(1, *obs.shape)

            logits = model(obs)
            if type(logits) == dict: # TODO: Change Vectorized output from dict -> numpy.ndarray
                pred_x = logits['positions'][:,0, 0].view(-1,1)# take the first action 
                pred_y = logits['positions'][:,0, 1].view(-1,1)# take the first action
                pred_yaw = logits['yaws'][:,0,:].view(-1,1)# take the first action
            else:
                batch_size = len(obs)
                predicted = logits.view(batch_size, -1, 3) # B, N, 3 (X,Y,yaw)
                pred_x = predicted[:, 0, 0].view(-1,1) # take the first action 
                pred_y = predicted[:, 0, 1].view(-1,1) # take the first action
                pred_yaw = predicted[:, 0, 2].view(-1,1)# take the first action
            # Normalize kin actions
            # print(pred_x.size())
            if use_kin:
                # action = torch.cat((pred_x, pred_y, pred_yaw), dim = -1).detach().cpu().numpy()
                # pred_x, pred_y, pred_yaw = torch.as_tensor(standard_normalizer_nonKin(action).reshape(-1)).view(-1, 1, 1)
                action = inverseUnicycle(pred_x, pred_y, pred_yaw, obs['old_speed']) # B, 1
                # print(f'inverse Unicycle: {output_logits}')
                actions.append(action.detach().numpy().reshape(-1))
                # action = action.detach().numpy().reshape(-1)
                action = standard_normalizer_kin(action).detach().numpy().reshape(-1) # scale actions
                # print(f'normalize actions: {action}')
            else:
                action = torch.cat((pred_x, pred_y, pred_yaw), dim = -1).detach().cpu().numpy()
                action = standard_normalizer_nonKin(action).reshape(-1)

        obs, _, done, info = env.step(action)
        # rescale actions
        # action[0] = steer_scale * action[0]
        # action[1] = acc_scale * action[1]
        # actions.append(action)
        if num_simulation_steps and idx >= num_simulation_steps:
            done = True
        if done:
            break
    # The episode outputs are present in the key "sim_outs"
    sim_out = info["sim_outs"][0]
    return sim_out, actions


import numpy as np
def rollout_episode_attentionModel(model, env, idx = 0, num_simulation_steps = None):

    config = model.get_config()

    state = np.zeros(
        config.to_dict()["model"]["attention_dim"]
    )
    prev_states_list = [
        state for _ in range(
            config.to_dict()["model"]["attention_memory_inference"]
        )
    ]
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
        input_dict = {
            "obs": obs,
            "state_in": np.stack(
                prev_states_list[-config.to_dict()["model"]["attention_memory_inference"]:]
            )
        }
        action, state, _ = model.compute_single_action(
            input_dict=input_dict, full_fetch=True)
        # action = model.compute_single_action(obs, explore=False)
        obs, _, done, info = env.step(action)
        prev_states_list.append(state[0])
        if num_simulation_steps and idx >= num_simulation_steps:
            done = True
        if done:
            break

    # The episode outputs are present in the key "sim_outs"
    sim_out = info["sim_outs"][0]
    return sim_out

def unroll_to_quantitative(model, rollout_env, num_scenes_to_unroll, num_simulation_steps = None, firstId = 0, model_type = 'CLOSED_LOOP', use_kin = True):
    sim_outs, actions, indices = unroll(model, rollout_env, num_scenes_to_unroll, num_simulation_steps, firstId, model_type, use_kin)
    results = quantitative(sim_outs, actions)
    return sim_outs, actions, indices, results

def unroll(model, rollout_env, num_scenes_to_unroll, num_simulation_steps = None, firstId = 0, model_type = 'CLOSED_LOOP', use_kin = True):
    scenes_to_unroll = list(range(firstId, firstId + num_scenes_to_unroll))
    sim_outs =[]
    actions = []
    idx = 0
    indices = []
    for i in scenes_to_unroll:
        indices.append(idx)
        print('scene id:', i)
        ret = rollout_episode(model, rollout_env, i, num_simulation_steps, model_type=model_type, use_kin= use_kin)
        sim_outs.append(ret[0])
        actions.extend(ret[1])
        idx = idx + len(ret[1])
    return sim_outs, actions, indices

def visualize_outputs(sim_outs):
    for sim_out in sim_outs: # for each scene
        vis_in = episode_out_to_visualizer_scene_gym_cle(sim_out, mapAPI)
        show(visualize(sim_out.scene_id, vis_in))

def visualize_outputs_simulation(sim_outs):
    for sim_out in sim_outs: # for each scene
        vis_in = simulation_out_to_visualizer_scene(sim_out, mapAPI)
        show(visualize(sim_out.scene_id, vis_in))
