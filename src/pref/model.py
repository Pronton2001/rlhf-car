
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.testing import assert_equal
import logging
from src.pref.pref_db import PrefDB
from src.pref.utils import RunningStat

source_path = '/workspace/source/'
dataset_path = '/workspace/datasets/'

N_STEPS = 25

class RewardModel(nn.Module):
    def __init__(self, state_shape, action_shape, hidden_size=128): #[batch_size, num_channels, height, width]
        super(RewardModel, self).__init__()
        assert state_shape[-1] < 15, f'wrong shape {state_shape}'
        self.conv_layers = nn.Sequential(
            nn.Conv2d(7, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.action_layer = nn.Linear(action_shape[0], hidden_size)
        self.reward_layer = nn.Sequential(
            nn.Linear(7*7*64+hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state, action):
        # print('state', state.shape)
        assert state.shape[-3:] == (7, 84, 84), f'{state.shape[-3:]} != (7, 84, 84)'
        x =  state.reshape(-1, 7, 84, 84) # B, N, # channels, W, H
        # print('state reshape', x.shape)
        x = self.conv_layers(x)
        # print('cnn', x.shape)
        x = x.view(x.size(0), -1)  # flatten output
        # print('flatten', x.shape)
        # print('action', action.shape)
        action = action.reshape(-1,3)
        # print('action reshape', action.shape)
        action = self.action_layer(action)
        # print('action layer', action.shape)
        x = torch.cat((x, action), dim=1)
        # print('cat layer', x.shape)
        x = self.reward_layer(x)
        # print('reward layer', x.shape)
        # x = x.reshape(-1,N_STEPS) # B, N
        # print('reshape reward layer', x.shape)
        # rs = rs.view(1,-1)
        # print('rehap', rs.shape)
        # pred = self.softmax(rs) # softmax
        return x
    
def train_and_save_RW_model(dataset, model_path = 'src/pref/model/model.pt', num_epochs=100, batch_size=10, lr=0.0001, kwargs = dict(state_shape=(84, 84, 7), action_shape=(3,))):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device =  'cpu' #NOTE: Remove for deploy
    # if state_shape == (7,84,84):
    model = RewardModel(**kwargs).to(device)
    # criterion = nn.BCEWithLogitsLoss()  # use binary cross-entropy loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(num_epochs):
        total_loss = 0.0
        # for traj1, traj2, pref in dataloader:
        for traj1_obs, traj1_act, traj2_obs, traj2_act, pref in dataloader:
            r1s = model(traj1_obs, traj1_act)
            r1s = r1s.reshape(-1, N_STEPS)
            sumR1 = r1s.sum(axis = 1)
            r2s = model(traj2_obs, traj2_act)
            r2s = r2s.reshape(-1, N_STEPS)
            sumR2 = r2s.sum(axis = 1)
            diff = sumR1 - sumR2

            log_p = torch.nn.functional.logsigmoid(diff)
            log_not_p = torch.nn.functional.logsigmoid(-diff)
            labels =torch.tensor(pref[:, 0] >= pref[:, 1], dtype=float) 
            print('labels', labels)
            print(f'r1: {sumR1}, r2: {sumR2}, r1 - r2', diff)

            losses = -1. * (labels * log_p + (1 - labels) * log_not_p)
            loss = losses.mean()
            print('loss', loss)
            
            # print(torch.tensor(pref[:, 0] >= pref[:, 1], dtype=float))
            # print('diff', diff.squeeze())
            # loss = criterion(diff.squeeze(), torch.tensor(pref[:, 0] >= pref[:, 1], dtype=float))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}: loss={total_loss / len(dataset)}")
    torch.save(model.state_dict(), model_path)
    return model

class RewardModelPredictor:
    def __init__(self, model_path = 'src/pref/model/model.pt', kwargs = dict(state_shape=(84, 84, 7), action_shape=(3,))):
        #NOTE: kwargs['state_shape'] (84,84,7) is different from obs.shape (7,84,84)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = 'cpu'#NOTE: Remove for deploy
        self.r_norm = RunningStat(shape=()) # float
        self.model_path = model_path
        self.kwargs  = kwargs 
        self.RWmodel = RewardModel(**self.kwargs)
        self.RWmodel.load_state_dict(torch.load(self.model_path))        
        self.n_channels = kwargs['state_shape'][-1]
        self.raster_size = kwargs['state_shape'][0]
        assert self.n_channels < self.raster_size, f'wrong shape'
        
    def predict_single_reward(self, obs_actions):
        obs, actions = obs_actions
        assert obs.shape[0] < obs.shape[1], f'wrong shape: {obs_.shape}' # C,W,H
        # Check valid obs, actions
        assert_equal(len(actions), 3) # x, y, yaw
        assert_equal(obs.shape, (self.n_channels,  self.raster_size, self.raster_size))    

        # Convert to Tensor
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device) 
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device) 

        # Load model
        pred = self.RWmodel(obs, actions).item()
        self.r_norm.push(pred)
        pred -= self.r_norm.mean
        pred /= (self.r_norm.std + 1e-12)
        pred *= 0.05
        
        # normalize reward
        logging.debug("Reward mean/stddev:\n%s %s",
                            self.r_norm.mean,
                            self.r_norm.std)
        logging.debug("Normalized rewards:\n%s", pred)
        return pred

    
    def predict_traj_reward(self, traj_dataset, n_steps = 25):
        # normalize reward
        traj_obs, traj_act = traj_dataset
        rs = self.RWmodel(traj_obs, traj_act).reshape(-1).detach().numpy()
        assert_equal(rs.shape, (n_steps,))# 50,
        for rs_step in rs:
            self.r_norm.push(rs_step)

        rs -= self.r_norm.mean
        rs /= (self.r_norm.std + 1e-12)
        rs *= 0.05
        assert_equal(rs.shape, (n_steps,))
        logging.debug("Reward mean/stddev:\n%s %s",
                        self.r_norm.mean,
                        self.r_norm.std)
        logging.debug("Normalized rewards:\n%s", rs)
        return rs
