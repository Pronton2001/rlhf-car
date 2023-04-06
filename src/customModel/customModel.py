from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch.nn as nn
import torch
from torch.nn import functional as F
from l5kit.planning.vectorized.open_loop_model import VectorizedModel, CustomVectorizedModel

import logging
logging.basicConfig(filename='/home/pronton/rl/rlhf-car/src/log/info.log', level=logging.DEBUG, filemode='w')

class TorchGNCNN(TorchModelV2, nn.Module):
    """
    Simple Convolution agent that calculates the required linear output layer
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # raise ValueError(obs_space.shape)
        self._num_objects = obs_space.shape[2] # num_of_channels of input, size x size x channels
        assert self._num_objects < 15, f'wrong shape: {obs_space.shape}'
        self._num_actions = num_outputs
        self._feature_dim = model_config["custom_model_config"]['feature_dim']
        assert obs_space.shape[0] > self._num_objects, str(obs_space.shape) + '!=  (size, size, # channels)'

        self.network = nn.Sequential(
            nn.Conv2d(7, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.GroupNorm(4, 64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 32, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.GroupNorm(2, 32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=1568, out_features=self._feature_dim),
        )

        self._actor_head = nn.Sequential(
            nn.Linear(self._feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self._num_actions),
        )

        self._critic_head = nn.Sequential(
            nn.Linear(self._feature_dim, 1),
        )

# <<<<<<< HEAD
#     def forward(self, input_dict, state, seq_lens):
#         obs_transformed = input_dict['obs'].permute(0, 3, 1, 2) # input_dict['obs'].shape = [B, size, size, # channels] => obs_transformed.shape = [B, # channels, size, size]
#         assert input_dict['obs'].shape[3] < input_dict['obs'].shape[2] , \
#             str(input_dict['obs'].shape) + ' != (_ ,size,size,n_channels),  obs_transformed: ' + str(obs_transformed.shape)
#         network_output = self.network(obs_transformed) #  [B, # channels, size, size]
# =======
    def forward(self, input_dict, state, seq_lens): # from dataloader? get 32, 112, 112, 7
        # obs_transformed = input_dict['obs'].permute(0, 3, 1, 2) # 32 x 112 x 112 x 7 [B, size, size, channels]
        obs_transformed = input_dict['obs'].permute(0, 3, 1, 2) # [B, C, W, H] -> [B, W, H, C]
        print('forward', obs_transformed.shape)
        network_output = self.network(obs_transformed)
# >>>>>>> 82fd9a0ee83cd280c7d1bcc9c254b002f5a103b1
        value = self._critic_head(network_output)
        self._value = value.reshape(-1)
        logits = self._actor_head(network_output)
        return logits, state

    def value_function(self):
        return self._value
class TorchGNCNN_separated(TorchModelV2, nn.Module):
    """
    Simple Convolution agent that calculates the required linear output layer
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # raise ValueError(obs_space.shape)
        self._num_objects = obs_space.shape[2] # num_of_channels of input, size x size x channels
        self._num_actions = num_outputs
        self._feature_dim = model_config["custom_model_config"]['feature_dim']

        self._actor_head = nn.Sequential(
            nn.Conv2d(self._num_objects, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.GroupNorm(4, 64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 32, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.GroupNorm(2, 32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=1568, out_features=self._feature_dim),
            nn.Linear(self._feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self._num_actions),
        )

        self._critic_head = nn.Sequential(
            nn.Conv2d(self._num_objects, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.GroupNorm(4, 64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 32, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.GroupNorm(2, 32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=1568, out_features=self._feature_dim),
            nn.Linear(self._feature_dim, 1),
        )

    def forward(self, input_dict, state, seq_lens):
        obs_transformed = input_dict['obs'].permute(0, 3, 1, 2) # input_dict['obs'].shape = [B, size, size, # channels] => obs_transformed.shape = [B, # channels, size, size]
        assert input_dict['obs'].shape[3] < input_dict['obs'].shape[2] , \
            str(input_dict['obs'].shape) + ' != (_ ,size,size,n_channels),  obs_transformed: ' + str(obs_transformed.shape)
        # network_output = self.network(obs_transformed)
        value = self._critic_head(obs_transformed)
        self._value = value.reshape(-1)
        logits = self._actor_head(obs_transformed)
        return logits, state

    def value_function(self):
        return self._value

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.layers.children():
            nn.init.zeros_(layer.bias)
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class TorchAttentionModel2(TorchModelV2, nn.Module):
    """
    Attention Model agent
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        # raise ValueError(num_outputs)

        cfg = model_config["custom_model_config"]['cfg']
        d_model = 256
        self._num_actions = num_outputs
        self._actor_head = MLP(d_model, d_model * 4, output_dim= num_outputs, num_layers=3)
        self._critic_head = MLP(d_model, d_model * 4, output_dim= 1, num_layers=1)

        self.feature_extractor= CustomVectorizedModel(
            history_num_frames_ego=cfg["model_params"]["history_num_frames_ego"],
            history_num_frames_agents=cfg["model_params"]["history_num_frames_agents"],
            global_head_dropout=cfg["model_params"]["global_head_dropout"],
            disable_other_agents=cfg["model_params"]["disable_other_agents"],
            disable_map=cfg["model_params"]["disable_map"],
            disable_lane_boundaries=cfg["model_params"]["disable_lane_boundaries"])
        

    def forward(self, input_dict, state, seq_lens):
        obs_transformed = input_dict['obs']
        out, attns = self.feature_extractor(obs_transformed)
        logits = self._actor_head(out).view(-1, self._num_actions)
        self._value = self._critic_head(out).view(-1)
        return logits, state

    def value_function(self):
        return self._value

class TorchAttentionModel(TorchModelV2, nn.Module):
    """
    Attention Model agent
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        # raise ValueError(num_outputs)

        cfg = model_config["custom_model_config"]['cfg']
        weights_scaling = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        self._num_predicted_frames = cfg["model_params"]["future_num_frames"]
        self._num_predicted_params = len(weights_scaling)
        weights_scaling_critic = [1.0]
        _num_predicted_params_critic = len(weights_scaling_critic)

        self._actor_head = VectorizedModel(
            history_num_frames_ego=cfg["model_params"]["history_num_frames_ego"],
            history_num_frames_agents=cfg["model_params"]["history_num_frames_agents"],
            num_targets=self._num_predicted_params * self._num_predicted_frames, # N (X,Y,Yaw) 72
            weights_scaling=weights_scaling, # 6
            criterion=nn.L1Loss(reduction="none"),
            global_head_dropout=cfg["model_params"]["global_head_dropout"],
            disable_other_agents=cfg["model_params"]["disable_other_agents"],
            disable_map=cfg["model_params"]["disable_map"],
            disable_lane_boundaries=cfg["model_params"]["disable_lane_boundaries"])

        self._critic_head = VectorizedModel(
            history_num_frames_ego=cfg["model_params"]["history_num_frames_ego"],
            history_num_frames_agents=cfg["model_params"]["history_num_frames_agents"],
            num_targets=_num_predicted_params_critic, # just 1 (X,Y,Yaw)
            weights_scaling=weights_scaling_critic,
            criterion=nn.L1Loss(reduction="none"),
            global_head_dropout=cfg["model_params"]["global_head_dropout"],
            disable_other_agents=cfg["model_params"]["disable_other_agents"],
            disable_map=cfg["model_params"]["disable_map"],
            disable_lane_boundaries=cfg["model_params"]["disable_lane_boundaries"])

    def forward(self, input_dict, state, seq_lens):
        # obs_transformed = input_dict['obs'].permute(0, 3, 1, 2) # 32 x 112 x 112 x 7 [b, size, size, channels]
        # raise ValueError(input_dict)
        obs_transformed = input_dict['obs']
        # raise ValueError(input_dict['obs'])
        # network_output = self.network(obs_transformed)
        logits = self._actor_head(obs_transformed)
        # raise ValueError(str(logits['positions'].shape))
        # logits = torch.cat((logits['positions'], logits['yaws']),axis=-1)
        logits = logits.view(-1, int(self._num_predicted_frames * self._num_predicted_params))
        # raise ValueError(logits.shape)
        value = self._critic_head(obs_transformed)
        self._value = value.view(-1)
        # raise ValueError(logits.shape)
        # raise ValueError('positions: ' + str(logits['positions'].shape) + 'yaw:' + str(logits['yaws'].shape))
        return logits, state

    def value_function(self):
        return self._value

class TorchAttentionModel3(TorchModelV2, nn.Module):
    """
    Attention Model agent
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        cfg = model_config["custom_model_config"]['cfg']
        # print('action space:', action_space)
        # print('num output:', num_outputs)
        weights_scaling = [1.0, 1.0, 1.0]
        # self._std_x = 0.005
        # self._std_y = 0.005
        # self._std_yaw = 0.005

        self._num_predicted_frames = cfg["model_params"]["future_num_frames"]
        # self._num_predicted_frames = 1
        self._num_predicted_params = len(weights_scaling) #6
        weights_scaling_critic = [1.0]
        _num_predicted_params_critic = len(weights_scaling_critic)

        self._actor_head = VectorizedModel(
            history_num_frames_ego=cfg["model_params"]["history_num_frames_ego"],
            history_num_frames_agents=cfg["model_params"]["history_num_frames_agents"],
            num_targets=self._num_predicted_params * self._num_predicted_frames, # N (X,Y,Yaw) 36
            weights_scaling=weights_scaling, # 3
            criterion=nn.L1Loss(reduction="none"),
            global_head_dropout=cfg["model_params"]["global_head_dropout"],
            disable_other_agents=cfg["model_params"]["disable_other_agents"],
            disable_map=cfg["model_params"]["disable_map"],
            disable_lane_boundaries=cfg["model_params"]["disable_lane_boundaries"])

        self._critic_head = CustomVectorizedModel(
            history_num_frames_ego=cfg["model_params"]["history_num_frames_ego"],
            history_num_frames_agents=cfg["model_params"]["history_num_frames_agents"],
            global_head_dropout=cfg["model_params"]["global_head_dropout"],
            disable_other_agents=cfg["model_params"]["disable_other_agents"],
            disable_map=cfg["model_params"]["disable_map"],
            disable_lane_boundaries=cfg["model_params"]["disable_lane_boundaries"])

        d_model = 256
        self._critic_FF = MLP(d_model, d_model * 4, output_dim= 1, num_layers=1)
        model_path = "/home/pronton/rl/l5kit/examples/urban_driver/OL_HS.pt"
        # self._critic_head.load_state_dict(torch.load(model_path).state_dict(), strict = False)
        self._actor_head.load_state_dict(torch.load(model_path).state_dict())
        # self._critic_head.load_state_dict()
        # self.outputs = nn.ModuleList()
        # for i in range(action_space.shape[0]):
        #     self.outputs.append(nn.Linear(num_outputs, 1)) # 6x
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        

    def forward(self, input_dict, state, seq_lens):
        obs_transformed = input_dict['obs']
        # logging.debug('obs forward:'+ str(obs_transformed))
        logits = self._actor_head(obs_transformed)
        logging.debug('predict traj' + str(logits))
        STEP_TIME = 0.1
        pred_x = logits['positions'][:,0, 0].view(-1,1) * STEP_TIME# take the first action 
        pred_y = logits['positions'][:,0, 1].view(-1,1) * STEP_TIME# take the first action
        pred_yaw = logits['yaws'][:,0,:].view(-1,1) * STEP_TIME# take the first action
        std = torch.ones_like(pred_x).to(self.device) *-10 # 32,
        # assert ones.shape[1] == 1, f'{ones.shape[1]}'
        # output_logits_mean = torch.cat((pred_x, pred_y, pred_yaw), dim = -1)
        output_logits = torch.cat((pred_x,pred_y, pred_yaw, std, std, std), dim = -1)
        # print('pretrained action', output_logits[:,:3])
        assert output_logits.shape[1] == 6, f'{output_logits.shape[1]}'

        # dist = torch.distributions.Normal(output_logits_mean, torch.ones_like(output_logits_mean)*0.0005)
        # print('-----------------------------sample', dist.rsample())

        feature_value, attns = self._critic_head(obs_transformed)
        value = self._critic_FF(feature_value)
        self._value = value.view(-1)

        return output_logits, state
    def value_function(self):
        return self._value
if __name__ == '__main__':

    import ray.rllib.algorithms.ppo as ppo
    from pprint import PrettyPrinter
    import numpy as np
    from l5kit.configs.config import load_config_data
    from l5kit.data.local_data_manager import LocalDataManager
    from l5kit.environment.envs.l5_env2 import GymStepOutput, SimulationConfigGym, L5Env2
    import os
    from ray.rllib.models import ModelCatalog
    # model = TorchGNCNN(np.zeros((112,112,7)), np.array((3,)),3, model_config= {'custom_model_config': {'feature_dim': 128}}, name='')

    # # In L5env
    # batch_data = {'obs': torch.ones((32,7, 112, 112))}
    # print('batch', batch_data['obs'].shape)

    # # After process in L5envWrapper
    # batch_data = {'obs': torch.ones((32, 112, 112, 7))}


    # # obs_transformed = obs_batch.permute(0, 3, 1, 2) # 32 x 112 x 112 x 7 [B, size, size, channels]
    # # print('transformed', obs_transformed.shape)
    # # print(obs_transformed.shape)
    # model(input_dict=batch_data)
    os.environ["L5KIT_DATA_FOLDER"] = "/media/pronton/linux_files/a100code/l5kit/l5kit_dataset"
    env_config_path = '/home/pronton/rl/rlhf-car/src/configs/gym_vectorizer_config.yaml'
    dmg = LocalDataManager(None)
    cfg = load_config_data(env_config_path)
    # rollout_sim_cfg = SimulationConfigGym()
    # rollout_sim_cfg.num_simulation_steps = None
    # env_kwargs = {'env_config_path': env_config_path, 'use_kinematic': False, 'sim_cfg': rollout_sim_cfg,  'train': False, 'return_info': True, 'rescale_action': False}
    # print(rollout_env.action_space)
    # model = TorchAttentionModel3(np.zeros((112,112,7)), np.array((3,)),3, model_config= {"custom_model_config": {'cfg':cfg}}, name='')
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    ###################### TRAINING ######################
    ModelCatalog.register_custom_model( "TorchSeparatedAttentionModel", TorchAttentionModel3)
    from ray import tune
    import ray
    train_eps_length = 32
    train_sim_cfg = SimulationConfigGym()
    train_sim_cfg.num_simulation_steps = train_eps_length + 1
    env_kwargs = {'env_config_path': env_config_path, 'use_kinematic': False, 'sim_cfg': train_sim_cfg, 'rescale_action': False}
    tune.register_env("L5-CLE-V2", lambda config: L5Env2(**env_kwargs))
    l5_env2 = L5Env2(**env_kwargs)
    ray.init(num_cpus=4, ignore_reinit_error=True, log_to_driver=False, local_mode=True)
    # algo = ppo.PPO(
    #         env="L5-CLE-V2",
    #         config={
                # 'disable_env_checking':True,
    #             "framework": "torch",
    #             'log_level': 'INFO',
    #             'num_gpu': 0,
    #             'train_batch_size': 1,
    #             'sgd_minibatch_size': 1,
    #             'num_sgd_iter': 1,
    #             'seed': 42,
    #             'batch_mode': 'truncate_episodes',
    #             # "rollout_fragment_length": 32,
    #             "model": {
    #                 "custom_model": "TorchSeparatedAttentionModel",
    #                 # Extra kwargs to be passed to your model's c'tor.
    #                 "custom_model_config": {'cfg':cfg},
    #             },
    #             # "output": "/home/pronton/rl/l5kit/examples/RL/notebooks/logs/l5env2-out", 
    #             # "output_max_file_size": 5000000,
    #             '_disable_preprocessor_api': True,
    #         },
    #     )

    # for i in range(1):
    #     result = algo.train()
    #     print(PrettyPrinter(result))
    import ray
    from ray import air, tune
    import pytz
    import datetime
    hcmTz = pytz.timezone("Asia/Ho_Chi_Minh") 
    date = datetime.datetime.now(hcmTz).strftime("%d-%m-%Y_%H-%M-%S")
    ray_result_logdir = '~/ray_results/debug' + date

    train_envs = 4
    lr = 3e-3
    from src.customModel.utils import kl_divergence, PretrainedDistribution
    from ray.rllib.evaluation.postprocessing import compute_advantages

    from ray.rllib.policy.sample_batch import SampleBatch
    from ray.rllib.models.torch.torch_distributions import TorchDiagGaussian
    from ray.rllib.models.torch.torch_action_dist import TorchDiagGaussian as TorchActionDiagGaussian
    from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy

    class KLPPO(PPOTorchPolicy):
        def __init__(self, obs_space, action_space, config):
            # config["model"].get("max_seq_len", 20)
            config["model"]["max_seq_len"] = 20
            super().__init__(obs_space, action_space, config)
            # Load your pretrained model here and convert it to a distribution
            # self.pretrained_dist = ...
        
        # @override(PPOTorchPolicy)
        #TODO: Fix many bugssssss
        def postprocess_trajectory(
                self, sample_batch, other_agent_batches=None, episode=None):
            # Compute the usual GAE values using the base postprocess_trajectory
            # function
            
            # Get the actions from the sample batch
            actions = sample_batch[SampleBatch.ACTIONS]
            pretrained_policy = self.config['pretrained_policy']

            # Convert observations to tensor
            logging.debug('obs', sample_batch['obs'])
            print(sample_batch['obs'])
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # obs = torch.as_tensor(sample_batch['obs']).to(device)

            # Compute the KL divergence reward
            # ppo_action_dist = self.dist_class(self.model.outputs, self.model)
            # ppo_action_dist = self.action_dist(sample_batch['obs'])
            # ppo_logits, state = self.model(sample_batch)
            # ppo_action_dist = self.dist_class(ppo_logits, self.model)
            ppo_action_dist = TorchActionDiagGaussian(
            sample_batch[SampleBatch.ACTION_DIST_INPUTS], self.model)

            # pretrained_action_dist = pretrained_policy.dist_class(pretrained_policy.model.outputs, pretrained_policy.model)

            # TODO: cal distribution of pretrain model 's action
            # logging.debug('ppo traj' + str(self.model.outputs))
            obs = {}
            for k,v in sample_batch[SampleBatch.CUR_OBS].items():
                obs[k] = torch.as_tensor(v).to(self.device)
            logits = pretrained_policy(obs)
            logging.debug('pretrained traj' + str(logits))
            STEP_TIME = 0.1
            pred_x = logits['positions'][:,0, 0].view(-1,1) * STEP_TIME# take the first action 
            pred_y = logits['positions'][:,0, 1].view(-1,1) * STEP_TIME# take the first action
            pred_yaw = logits['yaws'][:,0,:].view(-1,1) * STEP_TIME# take the first action
            log_std = torch.ones_like(pred_x).to(device) *0.0005 # 32,
            output_logits = torch.cat((pred_x,pred_y, pred_yaw), dim = -1)
            
            # scale = log_std.exp()
            print('logits',output_logits, ', log_std',log_std)
            pretrained_action_dist = TorchDiagGaussian(loc=output_logits, scale=log_std)
            
            # Create a distribution from the pretrained model
            pretrained_dist = pretrained_action_dist.sample()
            
            # Calculate the KL divergence between the PPO and pretrained distributions
            # kl_div = torch.distributions.kl_divergence(
            #     self.action_dist, pretrained_dist).mean()
            kl_div = ppo_action_dist.kl(pretrained_action_dist)
            # kl_div = pretrained_action_dist.kl(ppo_action_dist)
            # kl_div = kl_divergence(pretrained_dist)
            
            # Add the KL penalty to the rewards
            # gae_data[Postprocessing.] -= kl_div.item()
            logging.debug(f'reward shape{sample_batch[SampleBatch.REWARDS].shape}')
            logging.debug(f'kl shape{kl_div.shape}')
            sample_batch[SampleBatch.REWARDS] -= kl_div.detach().numpy()
            
            gae_data = super().postprocess_trajectory(
                sample_batch, other_agent_batches, episode)
            
            return gae_data
    pretrained_policy = VectorizedModel(
        history_num_frames_ego=cfg["model_params"]["history_num_frames_ego"],
        history_num_frames_agents=cfg["model_params"]["history_num_frames_agents"],
        num_targets=3 * 12, # N (X,Y,Yaw) 72
        weights_scaling=[1.0, 1.0, 1.0], # 6
        criterion=nn.L1Loss(reduction="none"),
        global_head_dropout=cfg["model_params"]["global_head_dropout"],
        disable_other_agents=cfg["model_params"]["disable_other_agents"],
        disable_map=cfg["model_params"]["disable_map"],
        disable_lane_boundaries=cfg["model_params"]["disable_lane_boundaries"])
    
    model_path = "/home/pronton/rl/l5kit/examples/urban_driver/OL_HS.pt"
    pretrained_policy.load_state_dict(torch.load(model_path).state_dict())
    # pretrain_dist = PretrainedDistribution(pretrained_policy)

    config_param_space = {
        "env": "L5-CLE-V2",
        "framework": "torch",
        "num_gpus": 0,
        "num_workers": 1,
        "num_envs_per_worker": 1,
        'disable_env_checking':False,
        # "postprocess_fn": my_postprocess_fn,
        "pretrained_policy": pretrained_policy,
        "model": {
                "custom_model": "TorchSeparatedAttentionModel",
                # Extra kwargs to be passed to your model's c'tor.
                "custom_model_config": {'cfg':cfg},
                "custom_options": {
                    "squash_output": False, # Turn off output squashing
                },
                # "custom_action_distribution_fn": gaussian_action_distribution_fn,
                # "custom_action_dist": DiagGaussianDistribution,
                # "custom_action_dist_cls": "CustomTorchActionDist",
                # "squash_to_range": True,
                # "logit_dim": 6,
                # "free_log_std": True,
                # "std_share_network": False,
                },
        
        # 'custom_policy' : KLPPO,
        '_disable_preprocessor_api': True,
        "eager_tracing": True,
        "restart_failed_sub_environments": True,
        "lr": 0.00001,
        'seed': 42,
        # "lr_schedule": [
        #     [1e6, lr_start],
        #     [2e6, lr_end],
        # ],
        'train_batch_size': 2, # 8000 
        'sgd_minibatch_size': 1, #2048
        'num_sgd_iter': 1,#16,
        'seed': 42,
        # 'batch_mode': 'truncate_episodes',
        # "rollout_fragment_length": 32,
        'gamma': 0.8,    
    }

    # result_grid = tune.Tuner(
    #     "PPO",
    #     run_config=air.RunConfig(
    #         stop={"episode_reward_mean": 0, 'timesteps_total': int(6e6)},
    #         local_dir=ray_result_logdir,
    #         checkpoint_config=air.CheckpointConfig(num_to_keep=2, 
    #                                             checkpoint_frequency = 10, 
    #                                             checkpoint_score_attribute = 'episode_reward_mean'),
    #         # callbacks=[WandbLoggerCallback(project="l5kit2", save_code = True, save_checkpoints = False),],
    #         ),
    #     param_space=config_param_space).fit()
    from ray.rllib.agents.ppo import PPOTrainer
    
    # trainer = KLPPO(obs_space= l5_env2.observation_space,
    #                 action_space =l5_env2.action_space,
    #                 config=config_param_space)
    trainer = PPOTrainer(config=config_param_space)
    from ray.tune.logger import pretty_print
    for i in range(10000):
        result = trainer.train()
        print(pretty_print(result))

        
# from ray.rllib.models.tf.misc import normc_initializer
# from ray.rllib.models.tf.tf_modelv2 import TFModelV2
# from ray.rllib.utils.framework import try_import_tf
# from tensorflow import keras


# tf1, tf, tfv = try_import_tf()
# layers = tf.keras.layers

# class TFGNNCNN(TFModelV2):
#     """Custom model for policy gradient algorithms."""

#     def __init__(self, obs_space, action_space, num_outputs, model_config, name):
#         super(TFGNNCNN, self).__init__(
#             obs_space, action_space, num_outputs, model_config, name
#         )
#         self._num_objects = obs_space.shape[2] # num_of_channels of input, size x size x channels
#         self._num_actions = num_outputs
#         self._feature_dim = model_config["custom_model_config"]['feature_dim']
#         self.inputs = tf.keras.layers.Input(shape=obs_space.shape, name="observations")
#         layer_1 = layers.Conv2D(64,kernel_size= (7,7), strides=(2,2), padding='same', use_bias=False, kernel_initializer=normc_initializer(1.0))(self.inputs)
#         layer_2 = layers.GroupNormalization(64)(layer_1)
#         layer_3 = layers.ReLU()(layer_2)
#         layer_4 = layers.MaxPool2D(pool_size=(2,2), strides=2)(layer_3) # x
#         layer_5 = layers.Conv2D(32,kernel_size= (7,7), strides=(2,2), padding='same', use_bias=False, kernel_initializer=normc_initializer(1.0))(layer_4)
#         layer_6 = layers.GroupNormalization(32)(layer_5),
#         # raise ValueError(layer_6.__repr__())
#         layer_7 = layers.ReLU()(layer_6)
#         layer_8 = layers.MaxPool2D(pool_size=(2,2), strides=2)(layer_7)
#         layer_9 = layers.Flatten()(layer_8)
#         layer_10 = layers.Dense(self._feature_dim)(layer_9)
#             # nn.Conv2d(self._num_objects, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
#             # nn.GroupNorm(4, 64),
#             # nn.ReLU(),
#             # nn.MaxPool2d(kernel_size=2, stride=2),
#             # nn.Conv2d(64, 32, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
#             # nn.GroupNorm(2, 32),
#             # nn.ReLU(),
#             # nn.MaxPool2d(kernel_size=2, stride=2),
#             # nn.Flatten(),
#             # nn.Linear(in_features=1568, out_features=self._feature_dim),

#         actor_out_1 = layers.Dense( 256, kernel_initializer=normc_initializer(0.01),)(layer_10)
#         actor_out_2= layers.ReLU()(actor_out_1)
#         actor_out_3 = layers.Dense( num_outputs, kernel_initializer=normc_initializer(0.01),name='critic_out')(actor_out_2)
#         critic_out = tf.keras.layers.Dense( 1, name="value_out", activation=None, kernel_initializer=normc_initializer(0.01),)(layer_1)

#         self.base_model = tf.keras.Model(self.inputs, [actor_out_3, critic_out])

#         # self.network = tf.keras.Sequential(
#         #     [
#         #         tf.keras.Input(shape=obs_space.shape),
#         #         layers.Conv2D(64,kernel_size= (7,7), strides=(2,2), padding='same', use_bias=False),
#         #         layers.GroupNormalization(64),
#         #         layers.ReLU(),
#         #         layers.MaxPool2D(pool_size=(2,2), strides=2),
#         #         layers.Conv2D(32,kernel_size= (7,7), strides=(2,2), padding='same', use_bias=False),
#         #         layers.GroupNormalization(32),
#         #         layers.ReLU(),
#         #         layers.MaxPool2D(pool_size=(2,2), strides=2),
#         #         layers.Flatten(),
#         #         layers.Dense(self._feature_dim)
#         #     ]
#         # )
#         # self._actor_head = tf.keras.Sequential(
#         #     [
#         #         layers.Dense(256),
#         #         layers.ReLU(),
#         #         layers.Dense(self._num_actions)
#         #     ]
#         # )

#         # self._critic_head = tf.keras.Sequential([
#         #         layers.Dense(1),
#         # ]
#         # )
#             # nn.Conv2d(self._num_objects, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
#             # nn.GroupNorm(4, 64),
#             # nn.ReLU(),
#             # nn.MaxPool2d(kernel_size=2, stride=2),
#             # nn.Conv2d(64, 32, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
#             # nn.GroupNorm(2, 32),
#             # nn.ReLU(),
#             # nn.MaxPool2d(kernel_size=2, stride=2),
#             # nn.Flatten(),
#             # nn.Linear(in_features=1568, out_features=self._feature_dim),
#         # self.inputs = tf.keras.layers.Input(shape=obs_space.shape, name="observations")
#         # layer_1 = tf.keras.layers.Dense( 128,name="my_layer1",activation=tf.nn.relu,kernel_initializer=normc_initializer(1.0),)(self.inputs)
#         # layer_out = tf.keras.layers.Dense(
#         #     num_outputs,
#         #     name="my_out",
#         #     activation=None,
#         #     kernel_initializer=normc_initializer(0.01),
#         # )(layer_1)
#         # value_out = tf.keras.layers.Dense(
#         #     1,
#         #     name="value_out",
#         #     activation=None,
#         #     kernel_initializer=normc_initializer(0.01),
#         # )(layer_1)
#         # self.base_model = tf.keras.Model(self.inputs, [layer_out, value_out])

#     def forward(self, input_dict, state, seq_lens):
#         # obs_transformed = input_dict['obs'].permute(0, 3, 1, 2) # 32 x 112 x 112 x 7 [B, size, size, channels]
#         # raise ValueError(input_dict["obs"].shape)
#         actor_out, self._value_out = self.base_model(input_dict["obs"])
#         return actor_out, state

#     def value_function(self):
#         return tf.reshape(self._value_out, [-1])
# if __name__ == '__main__':
#     def testGCNN():
