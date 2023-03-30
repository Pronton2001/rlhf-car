import torch
from torch.utils.data import Dataset, DataLoader
import logging
from l5kit.configs.config import load_config_data
from l5kit.rasterization.rasterizer_builder import build_rasterizer
from l5kit.dataset.ego import EgoDataset
from l5kit.data.local_data_manager import LocalDataManager
from l5kit.data.zarr_dataset import ChunkedDataset
from l5kit.geometry.transform import transform_points
from l5kit.visualization.utils import TARGET_POINTS_COLOR, draw_trajectory
from l5kit.visualization.video import write_video
from src.pref.model import RewardModelPredictor
from l5kit.configs import load_config_data
from src.pref.pref_db import PrefDB
from numpy.testing import assert_equal

import os
os.environ["L5KIT_DATA_FOLDER"] = "/mnt/datasets/"
source_path = '/workspace/source/'
dataset_path = '/workspace/datasets/'
dataset_path  = '/mnt/datasets'
source_path = "/home/pronton/rl/rlhf-car/"
logging.basicConfig(filename='src/log/info.log', level=logging.DEBUG, filemode='w')
BATCH_SIZE = 5
N_STEPS= 25

class PairwiseTrajDataset(Dataset):
    def __init__(self, pairwise_traj):
        self.pairwise_traj = pairwise_traj
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = 'cpu'
        
    def __len__(self):
        return len(self.pairwise_traj)
    
    def __getitem__(self, idx):
        traj1, traj2, pref = self.pairwise_traj[idx]
        
        # Convert each trajectory into a tensor of shape (length, 84, 84, 7) and (length, 3)
        traj1_obs = torch.tensor(np.array([step[0] for step in traj1]), dtype=torch.float32).to(self.device) # NOTE: convert list of np.ndarray -> np.array make torch faster
        traj1_act = torch.tensor(np.array([step[1] for step in traj1]), dtype=torch.float32).to(self.device)
        traj2_obs = torch.tensor(np.array([step[0] for step in traj2]), dtype=torch.float32).to(self.device)
        traj2_act = torch.tensor(np.array([step[1] for step in traj2]), dtype=torch.float32).to(self.device)
        pref = torch.tensor(pref, dtype=torch.float32)
        return traj1_obs, traj1_act, traj2_obs, traj2_act, pref


import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn as nn

class BTMultiLabelModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(BTMultiLabelModel, self).__init__()
        # (num_state + num_action) x 32, 32x16, 16x(2 probs)

        # Extract state information using a CNN layer
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(7, 16, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
        )

        # Concatenate state and action and feed to MLP layer
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(32*9*9 + 3, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, traj1_obs, traj1_act, traj2_obs, traj2_act):
        traj1_state = self.cnn(traj1_obs.reshape(-1, 84,84, 7).permute(0,3,1,2))  # permute already: (length, 84, 84, 7)
        traj2_state = self.cnn(traj2_obs.reshape(-1, 84, 84, 7).permute(0,3,1,2))
        print(traj1_state.shape)
        traj1_features = torch.cat([traj1_state, traj1_act.reshape(-1,3)], dim=1)
        traj2_features = torch.cat([traj2_state, traj2_act.reshape(-1,3)], dim=1)
        # pref_tensor = torch.tensor(pref, dtype=torch.float32)
        print(traj1_features.shape)
        print(traj2_features.shape)
        # cat_features=torch.cat([traj1_features, traj2_features], dim=0)
        # print(cat_features.shape)
        # pred_pref_tensor = self.mlp(torch.cat([traj1_features, traj2_features], dim=0))
        pred_pref1 = self.mlp(traj1_features) # 150,1
        pred_pref2 = self.mlp(traj2_features) # 150,1
        print(pred_pref1)
        print(pred_pref2)
        rs = pred_pref_tensor.sum(axis = 0)
        print('rs', rs)
        print('softmax', self.softmax(rs))
        
        return traj1_obs, traj1_act, traj2_obs, traj2_act, pref_tensor, pred_pref_tensor
        # Concatenate all state-action pairs from both trajectories
        # print('traj1', traj1.shape) # 3,3,2
        x = torch.cat((traj1, traj2)) # 2* num_steps = 2* 3= 6, num_batch = 3, num state + action = 2
        # print('concate', x.shape) # 6,3,2

        # Pass through fully connected layers with ReLU activation
        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        x = self.fc3(x)

        # Apply softmax to get probability distribution over labels
        # reshape to (2 * numsteps * num_batch)
        
        # print('before', x.shape)
        # x = x.view(-1,2)
        rs = x.sum(axis = 0)
        # print('sum', rs)
        # rs = rs.view(1,-1)
        # print('rehap', rs.shape)
        pred = self.softmax(rs) # softmax
        # print('softmax', pred)
        # print(pred)
        # Return predicted probability of trajectory 1 winning
        return pred

# pairwise_traj_dataset = PairwiseTrajDataset(pairwise_traj)
# dataloader = DataLoader(pairwise_traj_dataset, batch_size=32, shuffle=True)
import matplotlib.pyplot as plt
import numpy as np
cur = 107
prefs_train= PrefDB(maxlen=5).load(f'src/pref/preferences/{cur}.pkl.gz')
k1, k2, _ = prefs_train.prefs[0]

assert np.array(prefs_train.segments[k1][0][0]).shape == (7, 84, 84),f'error shape: {np.array(prefs_train.segments[k1][0][0]).shape} != (7, 84, 84)'
assert np.array(prefs_train.segments[k1][0][1]).shape == (3, ),f'error shape: {np.array(prefs_train.segments[k1][0][1]).shape} != (3,)'
assert np.array(prefs_train.segments[k1][0][0][0]).shape == (84, 84), f'error shape: {np.array(prefs_train.segments[k1][0][0][:,:,0]).shape} != (84, 84)'

def get_video():
    dm = LocalDataManager(None)
    # get config
    cfg = load_config_data(source_path + 'src/configs/gym_config84.yaml')

    cfg["raster_params"]["map_type"] = "py_semantic"
    rast = build_rasterizer(cfg, dm)
    dm = LocalDataManager()
    dataset_path = dm.require(cfg["train_data_loader"]["key"])
    train_zarr_dataset = ChunkedDataset(dataset_path)
    train_zarr_dataset.open()
    print(train_zarr_dataset)

    dataset = EgoDataset(cfg, train_zarr_dataset, rast)
    im1, im2 = [], []

    for i in range(cur - 5, cur):
        print()
        k1, k2, pref = prefs_train.prefs[i - cur + 5]
        for j, data in enumerate(prefs_train.segments[k1]):
            obs, actions = data
            im = obs.transpose(1, 2, 0) # C, W, H -> W,H,C
            im = dataset.rasterizer.to_rgb(im) 
            im1.append(im)
        for j, data in enumerate(prefs_train.segments[k2]):
            obs, actions = data
            im = obs.transpose(1, 2, 0) # C, W, H -> W,H,C
            im = dataset.rasterizer.to_rgb(im) 
            im2.append(im)
    write_video(f'src/pref/preferences/rl/scene{cur-5}_{cur}.mp4', im1, resolution=(224,224))
    write_video(f'src/pref/preferences/human/scene{cur-5}_{cur}.mp4', im2, resolution=(224,224))
    exit()
# get_video()

import os
directory = 'src/pref/preferences/'
pairwise_traj = []
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if not os.path.isfile(f): continue
    prefs_train= PrefDB(maxlen=5).load(f)
    assert_equal(len(prefs_train.prefs), 5)
    for i in range(len(prefs_train.prefs)):
        k1, k2, pref = prefs_train.prefs[i]
        assert_equal(len(prefs_train.segments[k1]), 25)
        assert_equal(len(prefs_train.segments[k2]), 25)


        pairwise_traj.append((prefs_train.segments[k1], prefs_train.segments[k2], pref))
pairwise_traj_dataset = PairwiseTrajDataset(pairwise_traj)

def inference(pairwise_traj_dataset, model_path = 'src/pref/model/model.pt', kwargs = dict(state_shape=(84, 84, 7), action_shape=(3,))):
    RWmodel= RewardModelPredictor(model_path=model_path, kwargs = kwargs)
    acc = 0
    i = 0
    dataloader = torch.utils.data.DataLoader(pairwise_traj_dataset, batch_size=1, shuffle=False)
    for traj1_obs, traj1_act, traj2_obs, traj2_act, pref in dataloader:
        normalized_reward1 = RWmodel.predict_traj_reward((traj1_obs, traj1_act))
        normalized_reward2 = RWmodel.predict_traj_reward((traj2_obs, traj2_act))
        acc += (normalized_reward1.sum(axis=0) >= normalized_reward2.sum(axis= 0)) == (pref[:,0] >= pref[:,1]).item()
    print(len(dataloader))
    print('accuracy', acc / len(dataloader))

if __name__ == '__main__':
    print(len(pairwise_traj_dataset))
    model_path = 'src/pref/model/model.pt'
    kwargs = dict(state_shape=(84, 84, 7), action_shape=(3,))
    # model = train_and_save_RW_model(pairwise_traj_dataset, model_path, kwargs = kwargs)
    inference(pairwise_traj_dataset, model_path, kwargs = kwargs)