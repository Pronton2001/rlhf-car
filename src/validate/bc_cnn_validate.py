import matplotlib.pyplot as plt
import numpy as np
import torch
from prettytable import PrettyTable

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset

from l5kit.dataset import EgoDatasetVectorized, EgoDataset
from l5kit.vectorization.vectorizer_builder import build_vectorizer
from l5kit.rasterization.rasterizer_builder import build_rasterizer

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
from l5kit.visualization.visualizer.zarr_utils import simulation_out_to_visualizer_scene

from collections import defaultdict
import os

from src.validate.validator import quantify_outputs, CLEValidator
# set env variable for data
from src.constant import SRC_PATH


os.environ["L5KIT_DATA_FOLDER"] = "/workspace/datasets/"

LOADED = False

if not LOADED:
    dm = LocalDataManager(None)
    # get config
    cfg = load_config_data(f"{SRC_PATH}src/configs/rasternet.yaml")
    model_path = f"{SRC_PATH}src/model/planning_model_20201208.pt"
    # model_path = "./BPTT.pt"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    model = torch.load(model_path)
    # model = SAC.load("/home/pronton/rl/l5kit/examples/RL/gg colabs/logs/SAC_640000_steps.zip")
    model = model.eval()
    torch.set_grad_enabled(False)

    eval_cfg = cfg["val_data_loader"]
    eval_zarr = ChunkedDataset(dm.require(eval_cfg["key"])).open()
    # vectorizer = build_vectorizer(cfg, dm)
    # eval_dataset = EgoDatasetVectorized(cfg, eval_zarr, vectorizer)
    vectorizer = build_rasterizer(cfg, dm)
    eval_dataset = EgoDataset(cfg, eval_zarr, vectorizer)
    print(eval_dataset)
    num_scenes_to_unroll = 100

    # ==== DEFINE CLOSED-LOOP SIMULATION
    # num_simulation_steps = None
    # sim_cfg = SimulationConfig(use_ego_gt=False, use_agents_gt=True, disable_new_agents=True,
    #                         distance_th_far=500, distance_th_close=50, num_simulation_steps=num_simulation_steps,
    #                         start_frame_index=0, show_info=True)
    from l5kit.environment.envs.l5_env import GymStepOutput, SimulationConfigGym
    sim_cfg = SimulationConfigGym()
    sim_cfg.num_simulation_steps = None

    sim_loop = ClosedLoopSimulator(sim_cfg, eval_dataset, device, model_ego=model, model_agents=None)

    # ==== UNROLL
    # sample 100 scenes at equally part of the whole 16220 scenes
    scenes_to_unroll = list(range(0, len(eval_zarr.scenes), len(eval_zarr.scenes)//num_scenes_to_unroll))
    sim_outs = sim_loop.unroll(scenes_to_unroll)

    from src.validate.validator import quantify_outputs, save_data
    quantify_outputs(sim_outs)
    CLEValidator(sim_outs)
    save_data(sim_outs, f'{SRC_PATH}src/validate/testset/bc_cnn.obj')
else:
    import pickle
    file = open(f'{SRC_PATH}src/validate/testset/bc_cnn.obj', 'rb')
    from src.validate.validator import quantify_outputs, CLEValidator
    sim_outs = pickle.load(file)
    quantify_outputs(sim_outs)
    CLEValidator(sim_outs)
