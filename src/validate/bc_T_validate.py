import matplotlib.pyplot as plt
import numpy as np
import torch
from prettytable import PrettyTable

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset

from l5kit.dataset import EgoDatasetVectorized
from l5kit.vectorization.vectorizer_builder import build_vectorizer

from l5kit.simulation.dataset import SimulationConfig
from l5kit.simulation.unroll import ClosedLoopSimulator
from l5kit.cle.closed_loop_evaluator import ClosedLoopEvaluator, EvaluationPlan
from l5kit.cle.metrics import (CollisionFrontMetric, CollisionRearMetric, CollisionSideMetric,
                               DisplacementErrorL2Metric, DistanceToRefTrajectoryMetric)
from l5kit.cle.validators import RangeValidator, ValidationCountingAggregator

from l5kit.visualization.visualizer.zarr_utils import simulation_out_to_visualizer_scene
from l5kit.visualization.visualizer.visualizer import visualize
from bokeh.io import output_notebook, show
from l5kit.data import MapAPI

from collections import defaultdict
import os
# set env variable for data
from src.constant import SRC_PATH


os.environ["L5KIT_DATA_FOLDER"] = "/workspace/datasets/"
dm = LocalDataManager(None)
# get config
cfg = load_config_data(f"{SRC_PATH}src/configs/gym_vectorizer_config.yaml")

LOADED = False

if not LOADED:
    model_path = f"{SRC_PATH}src/model/OL_HS.pt"
    # model_path = "./BPTT.pt"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    model = torch.load(model_path).to(device)
    # model = SAC.load("/home/pronton/rl/l5kit/examples/RL/gg colabs/logs/SAC_640000_steps.zip")
    model = model.eval()
    torch.set_grad_enabled(False)
    # ===== INIT DATASET
    eval_cfg = cfg["val_data_loader"]
    eval_zarr = ChunkedDataset(dm.require(eval_cfg["key"])).open()
    vectorizer = build_vectorizer(cfg, dm)
    eval_dataset = EgoDatasetVectorized(cfg, eval_zarr, vectorizer)
    print(eval_dataset)
    num_scenes_to_unroll = 1
    # ==== DEFINE CLOSED-LOOP SIMULATION
    # num_simulation_steps = None
    # # sim_cfg = SimulationConfig(use_ego_gt=False, use_agents_gt=True, disable_new_agents=True,
    # #                         distance_th_far=500, distance_th_close=50, num_simulation_steps=num_simulation_steps,
    # #                         start_frame_index=0, show_info=True)

    # from l5kit.environment.envs.l5_env2 import GymStepOutput, SimulationConfigGym
    # sim_cfg = SimulationConfigGym()
    # sim_cfg.num_simulation_steps = None
    # sim_loop = ClosedLoopSimulator(sim_cfg, eval_dataset, device, model_ego=model, model_agents=None)
    # # ==== UNROLL
    # firstId = 0
    # scenes_to_unroll = list(range(firstId, firstId+ num_scenes_to_unroll))
    # sim_outs = sim_loop.unroll(scenes_to_unroll)
    # from src.validate.validator import quantify_outputs, save_data, CLEValidator, quantitative
    # quantitative(sim_outs, [[0,0],[0,0]])
    # # save_data(sim_outs, f'{SRC_PATH}src/validate/testset/BC-T.obj')
    
    from src.simulation.unrollGym import unroll_to_quantitative
    from l5kit.environment.envs.l5_env2 import SimulationConfigGym, GymStepOutput, L5Env2
    # env_config_path = SRC_PATH + 'src/configs/gym_vectorizer_config_hist3.yaml'
    env_config_path = f'{SRC_PATH}src/configs/gym_vectorizer_config.yaml'
    rollout_sim_cfg = SimulationConfigGym()
    rollout_sim_cfg.num_simulation_steps = None
    env_kwargs = {'env_config_path': env_config_path, 
            'use_kinematic': True, 
            'sim_cfg': rollout_sim_cfg,  
            'train': False,
            'return_info': True}
    rollout_env = L5Env2(**env_kwargs)
    sim_outs, actions, indices = unroll_to_quantitative(model, rollout_env, 5, model_type='OPENED_LOOP', use_kin=True)

    from src.validate.validator import save_data, quantitative
    name = 'BC-T-kin'
    print(name)
    save_data((actions, indices), f'{SRC_PATH}src/validate/testset/{name}_actions.obj')
    save_data(sim_outs, f'{SRC_PATH}src/validate/testset/{name}.obj')
    
    
else:
    import pickle
    file = open(f'{SRC_PATH}src/validate/testset/BC-T-100000.obj', 'rb')
    from src.validate.validator import quantify_outputs, save_data, CLEValidator, quantitative
    sim_outs = pickle.load(file)
    quantitative(sim_outs)
