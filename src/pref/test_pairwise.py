import torch
from torch.utils.data import Dataset, DataLoader
import logging
from src.pref.utils import RunningStat

from pref_db import PrefDB
<<<<<<< HEAD
source_path = '/workspace/source/'
dataset_path = '/workspace/datasets/'
=======
from numpy.testing import assert_equal

logging.basicConfig(filename='src/log/info.log', level=logging.DEBUG, filemode='w')

>>>>>>> 82fd9a0ee83cd280c7d1bcc9c254b002f5a103b1
class PairwiseTrajDataset(Dataset):
    def __init__(self, pairwise_traj):
        self.pairwise_traj = pairwise_traj
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
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
N_STEPS= 25
# pairwise_traj = [ # [([[s1,a1], [s2,a2], [s3,a3]], [[s1,a1], [s2,a2], [s3,a3]], (p1, p2))]
#     ([[1, 2], [2, 3], [3, 4]], [[2, 3], [3, 4], [4, 5]], (0.7, 0.3)),
#     ([[2, 3], [3, 4], [4, 5]], [[1, 2], [2, 3], [3, 4]], (0.2, 0.8)),
#     ([[3, 4], [4, 5], [5, 6]], [[4, 5], [5, 6], [6, 7]], (0.5, 0.5)),
# ]
# pairwise_traj_dataset = PairwiseTrajDataset(pairwise_traj)
# dataloader = DataLoader(pairwise_traj_dataset, batch_size=32, shuffle=True)
import torch
import torch.nn as nn
import torch.optim as optim
# class RewardModel(nn.Module):
#     def __init__(self, num_inputs, hidden_size):
#         super(RewardModel, self).__init__()
#         self.fc1 = nn.Linear(num_inputs, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, hidden_size)
#         self.fc3 = nn.Linear(hidden_size, 2)

#     def forward(self, input):
#         x = torch.relu(self.fc1(input))
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
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
        # self.fc1 = nn.Linear(state_dim + action_dim, 64) #84x84x7 + 3
        # self.fc2 = nn.Linear(64, 64)
        # self.fc3 = nn.Linear(64, 2)
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
import torch
import torch.nn as nn
import torch.optim as optim

<<<<<<< HEAD

prefs_train= PrefDB(maxlen=5).load(source_path + 'src/pref/preferences/10.pkl.gz')
pairwise_traj_dataset = PairwiseTrajDataset(pairwise_traj)
dataloader = DataLoader(pairwise_traj_dataset, batch_size=32, shuffle=True)

=======
# pairwise_traj_dataset = PairwiseTrajDataset(pairwise_traj)
# dataloader = DataLoader(pairwise_traj_dataset, batch_size=32, shuffle=True)
>>>>>>> 82fd9a0ee83cd280c7d1bcc9c254b002f5a103b1
import matplotlib.pyplot as plt
import numpy as np
prefs_train= PrefDB(maxlen=5).load(f'src/pref/preferences/5.pkl.gz')
k1, _, _ = prefs_train.prefs[0]

assert np.array(prefs_train.segments[k1][0][0]).shape == (7, 84, 84),f'error shape: {np.array(prefs_train.segments[k1][0][0]).shape} != (7, 84, 84)'
assert np.array(prefs_train.segments[k1][0][1]).shape == (3, ),f'error shape: {np.array(prefs_train.segments[k1][0][1]).shape} != (3,)'
<<<<<<< HEAD
assert np.array(prefs_train.segments[k1][0][0][:,:,0]).shape == (84, 84), f'error shape: {np.array(prefs_train.segments[k1][0][0][:,:,0]).shape} != (84, 84)'

#NOTE - Reshape for visualize
plt.imshow(np.array(prefs_train.segments[k1][0][0]).reshape(7,84,84)[3])
plt.show()

for i in range(len(prefs_train)):
    k1, k2, pref = prefs_train.prefs[i] # traj[y]: series of states, actions, traj[y]
    for k in [k1, k2]:
        traj = np.array(prefs_train.segments[k]) # num_steps, ...
        for y in range(traj.shape[0]): 
            # plt.imshow(traj[y][0].reshape(7, 84, 84)[5]) # k, 0, 0
            # plt.show()
            state = traj[y][0].reshape(7, 84, 84) #
            print('action:', traj[y][1])
    
=======
assert np.array(prefs_train.segments[k1][0][0][0]).shape == (84, 84), f'error shape: {np.array(prefs_train.segments[k1][0][0][:,:,0]).shape} != (84, 84)'
plt.imshow(prefs_train.segments[k1][0][0][3])
# # plt.imshow(np.array(prefs_train.segments[k1][0][0])[6,:,:]) # k, 0, 0
plt.show()

import os
directory = 'src/pref/preferences/'

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if not os.path.isfile(f): continue
    prefs_train= PrefDB(maxlen=5).load(f)
    pairwise_traj = []
    assert_equal(len(prefs_train.prefs), 5)
    for i in range(len(prefs_train.prefs)):
        k1, k2, pref = prefs_train.prefs[i]
        assert_equal(len(prefs_train.segments[k1]), 25)
        assert_equal(len(prefs_train.segments[k2]), 25)
        pairwise_traj.append((prefs_train.segments[k1], prefs_train.segments[k2], pref))
pairwise_traj_dataset = PairwiseTrajDataset(pairwise_traj)

# pairwise_traj = 
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
        x = x.reshape(-1,N_STEPS) # B, N
        # print('reshape reward layer', x.shape)
        # rs = rs.view(1,-1)
        # print('rehap', rs.shape)
        # pred = self.softmax(rs) # softmax
        return x

kwargs = dict(state_shape=(84, 84, 7), action_shape=(3,))
def train_bt_model(dataset, num_epochs=100, batch_size=10, lr=0.0001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
            sumR1 = r1s.sum(axis = 1)
            r2s = model(traj2_obs, traj2_act)
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
    return model

MODEL_PATH = 'src/pref/model/model.pt'
def trainModel():
    model = train_bt_model(pairwise_traj_dataset)
    torch.save(model.state_dict(), MODEL_PATH)

# Load model
def inferenceModel():
    newModel = RewardModel(**kwargs)
    newModel.load_state_dict(torch.load(MODEL_PATH))
    # normalize reward
    r_norm = RunningStat(shape=1)
    assert len(pairwise_traj_dataset[0]) == 5
    n_preds = 1
    n_steps = N_STEPS
    for traj1_obs, traj1_act, traj2_obs, traj2_act, pref in pairwise_traj_dataset:
        rewards1 = newModel(traj1_obs, traj1_act).detach() # 1, 25
        rewards2 = newModel(traj2_obs, traj2_act).detach()
        # rewards = newModel(traj1_obs, traj1_act)
        rs = np.concatenate((rewards1, rewards2), axis=1).reshape(1, -1) # 1, 50
        assert_equal(rs.shape, (1, 2* n_steps))

        rs = rs.transpose(1,0) # N_STEPS, N_PREDs
        assert_equal(rs.shape, (2* n_steps, 1))
        for rs_step in rs:
            r_norm.push(rs_step)
        
        rs -= r_norm.mean
        rs /= (r_norm.std + 1e-12)
        rs *= 0.05
        rs = rs.transpose()
        assert_equal(rs.shape, (1, 2 * n_steps))
        logging.debug("Reward mean/stddev:\n%s %s",
                        r_norm.mean,
                        r_norm.std)
        logging.debug("Normalized rewards:\n%s", rs)
>>>>>>> 82fd9a0ee83cd280c7d1bcc9c254b002f5a103b1

        # "...and then averaging the results."
        rs = np.mean(rs, axis=0)
        assert_equal(rs.shape, (2*n_steps, ))
        logging.debug("After ensemble averaging:\n%s", rs)
if __name__ == '__main__':

    # trainModel()
    inferenceModel()
exit()
for traj1_obs, traj1_act, traj2_obs, traj2_act, pref in pairwise_traj_dataset:
    rewards1 = newModel(traj1_obs, traj1_act) # 1, 25
    rewards2 = newModel(traj2_obs, traj2_act)
    # rewards = newModel(traj1_obs, traj1_act)
    rs = np.concatenate((rewards1, rewards2), axis=1).reshape(1, -1) # 1, 50
    assert_equal(rs.shape, (1, 2* n_steps))

    rs = rewards1.transpose() # N_STEPS, N_PREDs
    for rs_step in rs:
        r_norm.push(rs)
    
    rs -= r_norm.mean
    rs /= (r_norm.std + 1e-12)
    rs *= 0.05
    rs = rs.transpose()
    assert_equal(rs.shape, (n_preds, n_steps))
    logging.debug("Reward mean/stddev:\n%s %s",
                    r_norm.mean,
                    r_norm.std)
    logging.debug("Normalized rewards:\n%s", rs)

    # "...and then averaging the results."
    rs = np.mean(rs, axis=0)
    assert_equal(rs.shape, (n_steps, ))
    logging.debug("After ensemble averaging:\n%s", rs)
    #    


# Initialize reward model
reward_model = BTMultiLabelModel(state_dim=(84,84,7), action_dim=(3,))
# criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(reward_model.parameters(), lr=0.001)
criterion = nn.BCELoss()

for epoch in range(10000):
    for traj1_obs, traj1_action, traj2_obs, traj2_action, pref in dataloader:
        # Zero out optimizer gradients
        optimizer.zero_grad()

        # Get predicted probability of trajectory 1 winning
        p1_pred = reward_model(traj1_obs, traj1_action, traj2_obs, traj2_action) # 2, num_step, state,action

        # Convert preference label to binary tensor
        # p1_pred = 6,3 (2 * num)
        # print(pref) # 3,
         
        # mu1 = pref # 3,1
        # mu2 = 1 - pref# 3,1

        pref_tensor = torch.tensor(pref)

        # Compute binary cross-entropy loss with preference label
        loss = criterion(p1_pred, pref_tensor)

        # Backpropagate and update parameters
        loss.backward()
        optimizer.step()
    if epoch % 100 == 0:
        print('Iteration %d, loss = %.4f' % (epoch, loss.item()))