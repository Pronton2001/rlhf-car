import matplotlib.pyplot as plt
import numpy as np
from src.constant import SRC_PATH
from src.customEnv.wrapper import L5EnvWrapper
import torch
from prettytable import PrettyTable

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset

from l5kit.dataset import EgoDatasetVectorized
from l5kit.vectorization.vectorizer_builder import build_vectorizer

from l5kit.simulation.dataset import SimulationConfig
from l5kit.simulation.unroll import ClosedLoopSimulator

from l5kit.visualization.visualizer.visualizer import visualize_plot
from bokeh.io import output_notebook, show
from l5kit.data import MapAPI
from bokeh.models import Button
from l5kit.environment.envs.l5_env import GymStepOutput, SimulationConfigGym, L5Env

from l5kit.configs import load_config_data
from l5kit.environment.feature_extractor import CustomFeatureExtractor
from l5kit.environment.callbacks import L5KitEvalCallback
from l5kit.environment.envs.l5_env import SimulationConfigGym

from l5kit.visualization.visualizer.zarr_utils import episode_out_to_visualizer_scene_gym_cle, simulation_out_to_visualizer_scene
from l5kit.visualization.visualizer.visualizer import visualize
from bokeh.io import output_notebook, show

import os
import gym

# set env variable for data
dataset_path = '/workspace/datasets/'
os.environ["L5KIT_DATA_FOLDER"] = dataset_path
dm = LocalDataManager(None)
env_config_path = SRC_PATH + 'src/configs/gym_config84.yaml'
cfg = load_config_data(env_config_path)
# Train on episodes of length 32 time steps
train_eps_length = 32
train_envs = 4


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
    rollout_sim_cfg.num_simulation_steps = 20

    env_kwargs = {'env_config_path': env_config_path, 
                'use_kinematic': True, 
                'sim_cfg': rollout_sim_cfg,  
                'train': False, 
                'return_info': True}

    rollout_env = L5EnvWrapper(env = L5Env(**env_kwargs), \
                            raster_size= cfg['raster_params']['raster_size'][0], \
                            n_channels = 7,)
    tune.register_env("L5-CLE-EVAL-V1", 
                    lambda config: L5EnvWrapper(env = L5Env(**env_kwargs), \
                                                raster_size= cfg['raster_params']['raster_size'][0], \
                                                n_channels = 7))
    from ray.rllib.algorithms.sac import SAC
    # checkpoint_path = 'l5kit/ray_results/01-01-2023_15-53-49/SAC/SAC_L5-CLE-V1_cf7bb_00000_0_2023-01-01_08-53-50/checkpoint_000170'
    checkpoint_path = dataset_path + 'ray_results/31-12-2022_07-53-04(SAC ~-30)/SAC/SAC_L5-CLE-V1_7bae1_00000_0_2022-12-31_00-53-04/checkpoint_000360'
    algo = SAC(config=config_param_space, env='L5-CLE-EVAL-V1')
    algo.restore(checkpoint_path)
    return rollout_env, algo

def vectorNet():
    os.environ["L5KIT_DATA_FOLDER"] = "/workspace/datasets/"
    dm = LocalDataManager(None)
    # get config
    cfg = load_config_data(f"{SRC_PATH}src/configs/urban_driver.yaml")
    model_path = f"{SRC_PATH}src/model/OL_HS.pt"
    # model_path = "./BPTT.pt"
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    model = torch.load(model_path)
    # model = SAC.load("/home/pronton/rl/l5kit/examples/RL/gg colabs/logs/SAC_640000_steps.zip")
    model = model.eval()
    torch.set_grad_enabled(False)
    # ===== INIT DATASET
    eval_cfg = cfg["val_data_loader"]
    eval_zarr = ChunkedDataset(dm.require(eval_cfg["key"])).open()
    vectorizer = build_vectorizer(cfg, dm)
    eval_dataset = EgoDatasetVectorized(cfg, eval_zarr, vectorizer)
    print(eval_dataset)
    num_scenes_to_unroll = 2
    num_simulation_steps = 20
    # ==== DEFINE CLOSED-LOOP SIMULATION
    sim_cfg = SimulationConfig(use_ego_gt=False, use_agents_gt=True, disable_new_agents=True,
                            distance_th_far=500, distance_th_close=50, num_simulation_steps=num_simulation_steps,
                            start_frame_index=0, show_info=True)

    sim_loop = ClosedLoopSimulator(sim_cfg, eval_dataset, device, model_ego=model, model_agents=None)
    # ==== UNROLL
    scenes_to_unroll = list(range(0, len(eval_zarr.scenes), len(eval_zarr.scenes)//num_scenes_to_unroll))
    sim_outs = sim_loop.unroll(scenes_to_unroll)
    from src.validate.validator import quantify_outputs, save_data
    quantify_outputs(sim_outs)
    try:
        save_data(sim_outs, f'{SRC_PATH}src/validate/BC-T.obj')
    except Exception as e:
        print(e)
    return sim_outs

sim_outs = vectorNet()


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
        action = model.compute_single_action(obs, explore=False)
        # im = obs.reshape(7, 84, 84) # reshape again
        # for i in im:
        #     plt.imshow(i)
        #     plt.show()
        obs, _, done, info = env.step(action)
        if done:
            break

    # The episode outputs are present in the key "sim_outs"
    sim_out = info["sim_outs"][0]
    return sim_out
####################################################
import pickle
# file = open(f'{PATH}sac_cnn_checkpoint150(-73).obj', 'rb')
# file = open(f'{PATH}ppo_cnn_checkpoint570(-71).obj', 'rb')
#file = open(f'{PATH}PPO-T_RLfinetune_freeze_checkpoint_003260(-81).obj', 'rb')
#file = open(f'{PATH}sac_vector_RLfinetune_freeze_checkpoint570(-47).obj', 'rb')
# file = open(f'{PATH}bc_cnn.obj', '+wb')
# sim_outs = pickle.load(file)
mapAPI = MapAPI.from_cfg(dm, cfg)
from bokeh.layouts import column, LayoutDOM, row, gridplot
from bokeh.io import curdoc

cols = []
doc = curdoc()


from bokeh.io import export_png
for i,sim_out in enumerate(sim_outs): # for each scene
    # vis_in = episode_out_to_visualizer_scene_gym_cle(sim_out, mapAPI)
    vis_in = simulation_out_to_visualizer_scene(sim_out, mapAPI) # for vectornet
    f= visualize_plot(0, vis_in[0], doc)
    cols.append(column(f))
    f= visualize_plot(1, vis_in[5], doc)
    cols.append(column(f))
    f= visualize_plot(2, vis_in[10], doc)
    cols.append(column(f))
    f= visualize_plot(3, vis_in[15], doc)
    cols.append(column(f))
    f= visualize_plot(4, vis_in[17], doc)
    cols.append(column(f))

    # f= visualize_plot(0.5, vis_in[25], doc)
    # cols.append(column(f))
    # f= visualize_plot(1.5, vis_in[75], doc)
    # cols.append(column(f))
    # f= visualize_plot(2.5, vis_in[125], doc)
    # cols.append(column(f))
    # f= visualize_plot(3.5, vis_in[175], doc)
    # cols.append(column(f))
    # f= visualize_plot(4.5, vis_in[225], doc)
    # cols.append(column(f))
    export_png(row(cols), filename= f'{SRC_PATH}/src/validate/BC-T/{i}.png')
    cols = []
# grid = gridplot(cols, ncols=5, plot_width=50, plot_height=50)


# doc.add_root(grid)