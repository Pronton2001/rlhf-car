#@title Download L5 Sample Dataset and install L5Kit
import os
os.environ["L5KIT_DATA_FOLDER"] = "/workspace/datasets/"
import matplotlib.pyplot as plt

import numpy as np

from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoDataset, AgentDataset

from l5kit.rasterization import build_rasterizer
from l5kit.configs import load_config_data
from l5kit.visualization import draw_trajectory, TARGET_POINTS_COLOR
from l5kit.geometry import transform_points
from tqdm import tqdm
from collections import Counter
from l5kit.data import PERCEPTION_LABELS
from prettytable import PrettyTable

import os

from l5kit.visualization.visualizer.zarr_utils import zarr_to_visualizer_scene
from l5kit.visualization.visualizer.visualizer import visualize
from bokeh.io import output_notebook, show
from l5kit.data import MapAPI
# Dataset is assumed to be on the folder specified
# in the L5KIT_DATA_FOLDER environment variable

# get config
cfg = load_config_data("src/configs/gym_config.yaml")

dm = LocalDataManager()
dataset_path = dm.require(cfg["train_data_loader"]["key"])
train_zarr_dataset = ChunkedDataset(dataset_path)
train_zarr_dataset.open()
print(train_zarr_dataset)
###################################
dm = LocalDataManager()
dataset_path = dm.require(cfg["val_data_loader"]["key"])
val_zarr_dataset = ChunkedDataset(dataset_path)
val_zarr_dataset.open()
print(val_zarr_dataset)