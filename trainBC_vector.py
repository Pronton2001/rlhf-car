import os

from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from tempfile import gettempdir

from l5kit.configs import load_config_data
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoDatasetVectorized
from l5kit.planning.vectorized.open_loop_model import VectorizedModel
from l5kit.vectorization.vectorizer_builder import build_vectorizer

#@title Download L5 Sample Dataset and install L5Kit
import os

from src.constant import SRC_PATH
os.environ["L5KIT_DATA_FOLDER"] = "/workspace/datasets/"

dm = LocalDataManager(None)
# get config

cfg = load_config_data(f"{SRC_PATH}/src/configs/gym_vectorizer_config.yaml")

train_zarr = ChunkedDataset(dm.require(cfg["train_data_loader"]["key"])).open()

vectorizer = build_vectorizer(cfg, dm)
print('start')
train_dataset = EgoDatasetVectorized(cfg, train_zarr, vectorizer)
print('end')
print(train_dataset)

weights_scaling = [1.0, 1.0, 1.0]

_num_predicted_frames = cfg["model_params"]["future_num_frames"]
_num_predicted_params = len(weights_scaling)

model = VectorizedModel(
    history_num_frames_ego=cfg["model_params"]["history_num_frames_ego"],
    history_num_frames_agents=cfg["model_params"]["history_num_frames_agents"],
    num_targets=_num_predicted_params * _num_predicted_frames,
    weights_scaling=weights_scaling,
    criterion=nn.L1Loss(reduction="none"),
    global_head_dropout=cfg["model_params"]["global_head_dropout"],
    disable_other_agents=cfg["model_params"]["disable_other_agents"],
    disable_map=cfg["model_params"]["disable_map"],
    disable_lane_boundaries=cfg["model_params"]["disable_lane_boundaries"],
)
# model = torch.load(f"{SRC_PATH}src/model/OL_HS.pt")
# print(model.eval())
train_cfg = cfg["train_data_loader"]
train_dataloader = DataLoader(train_dataset, shuffle=train_cfg["shuffle"], batch_size=train_cfg["batch_size"],
                              num_workers=train_cfg["num_workers"])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
print('device', device)

tr_it = iter(train_dataloader)
progress_bar = tqdm(range(cfg["train_params"]["max_num_steps"]))
losses_train = []
model.train()
torch.set_grad_enabled(True)

changed = False

for i,_ in enumerate(progress_bar):
    try:
        data = next(tr_it)
    except StopIteration:
        tr_it = iter(train_dataloader)
        data = next(tr_it)
    if i >=54 and changed:
        for g in optimizer.param_groups:
            g['lr'] = 1e-5
        changed = True

    # Forward pass
    data = {k: v.to(device) for k, v in data.items()}
    result = model(data)
    loss = result["loss"]
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses_train.append(loss.item())
    progress_bar.set_description(f"loss: {loss.item()} loss(avg): {np.mean(losses_train)}")

to_save = torch.jit.script(model.cpu())
path_to_save = f"{SRC_PATH}/src/model/OL_HS_61.pt"
to_save.save(path_to_save)
print(f"MODEL STORED at {path_to_save}")
