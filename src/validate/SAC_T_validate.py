import os
from src.customModel.customKLSACTrainer import KLSAC
from src.customModel.customModel import TorchAttentionModel3, TorchVectorSharedSAC, TorchVectorQNet, TorchVectorPolicyNet

from src.constant import SRC_PATH
# from src.customModel.customSACTrainer import KLSAC
os.environ["L5KIT_DATA_FOLDER"] = '/workspace/datasets'
# os.environ['CUDA_VISIBLE_DEVICES']= '0'
# os.environ["TUNE_RESULT_DIR"] =  '/DATA/l5kit/rllib_tb_logs'
import gym
from l5kit.configs import load_config_data
from l5kit.environment.envs.l5_env import SimulationConfigGym, GymStepOutput, L5Env
from l5kit.environment.envs.l5_env2 import SimulationConfigGym, GymStepOutput, L5Env2
from l5kit.visualization.visualizer.zarr_utils import episode_out_to_visualizer_scene_gym_cle
from l5kit.visualization.visualizer.visualizer import visualize
from bokeh.io import output_notebook, show
from l5kit.environment.gym_metric_set import L2DisplacementYawMetricSet, CLEMetricSet
from prettytable import PrettyTable
import datetime
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch.nn as nn
import numpy as np
import gym
from typing import Dict
import numpy as np
import ray
import pytz
from ray import tune


from l5kit.configs import load_config_data

# get environment config
# env_config_path = '/workspace/source/configs/gym_config_history3.yaml'
# env_config_path = '/workspace/source/configs/gym_config84.yaml'
# env_config_path = SRC_PATH + 'src/configs/gym_vectorizer_config_hist3.yaml'
env_config_path = f'{SRC_PATH}src/configs/gym_vectorizer_config.yaml'
cfg = load_config_data(env_config_path)
rollout_env = 0


#################### Define Training and Evaluation Environments ####################
ModelCatalog.register_custom_model( "TorchVectorQNet", TorchVectorQNet)
ModelCatalog.register_custom_model( "TorchVectorPolicyNet", TorchVectorPolicyNet)
ModelCatalog.register_custom_model( "TorchVectorSAC", TorchVectorSharedSAC)

LOADED = False

if not LOADED:
    reward_kwargs = {
        'yaw_weight': 1.0,
        'dist_weight': 1.0,
        # 'd2r_weight': 0.0,
        'cf_weight': 20.0,
        'cr_weight': 20.0,
        'cs_weight': 20.0,
    }
    lr = 3e-4
    lr_start = 3e-5
    lr_end = 3e-6
    eval_config_param_space = {
        # "env": "L5-CLE-V2",
        "framework": "torch",
        "num_gpus": 1,
        "num_workers": 2,
        "num_envs_per_worker": 4,
        'q_model_config':{
            'custom_model': 'TorchVectorQNet',
            "custom_model_config": {
            'cfg':cfg,
            'freeze_for_RLtuning':  False,
            'load_pretrained': False,
            'share_feature_extractor': False, # policy, q and twin-q use 1 shared feature extractor -> more efficiency
            },
            # "post_fcnet_hiddens": [256],
            # "post_fcnet_activation": "relu",
        },
            'policy_model_config':{
                'custom_model': 'TorchVectorPolicyNet',
                "custom_model_config": {
                    'cfg':cfg,
                    'freeze_for_RLtuning': False,
                    'load_pretrained': True,
                    'share_feature_extractor': False,
                    'kl_div_weight': None,
                    'm_tau': 1,
                    'm_alpha': 0,
                    'm_l0': -1,
                    'm_entropy': 0.1,
                    'm_kl': 0,
                    'use_entropy_kl_params': True,
                    'sac_entropy_equal_m_entropy': True,
                    'log_std_acc': -1.5,
                    'log_std_steer': -1,
                    'reward_kwargs': reward_kwargs,
                },
            },
        #  'model':{
        # 'custom_model': 'TorchVectorSAC',
        # 'custom_model_config':{
        #     'cfg': cfg,
        #     'freezing': True,
        #     'kl_div_weight': 1,
        #     'log_std_acc': -1,
        #     'log_std_steer': -1,
        #     'reward_kwargs': reward_kwargs,
        #     },
        # },
        'tau': 0.005,
        'target_network_update_freq': 1,
        'replay_buffer_config':{
            'type': 'MultiAgentPrioritizedReplayBuffer',
            'capacity': int(1e5),
            "worker_side_prioritization": True,
        },
        'num_steps_sampled_before_learning_starts': 1024,
        
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
        'train_batch_size': 256, #512, #1024,#2048,
        'training_intensity' : 32, # (4x 'natural' value = 8) train_batch_size / (rollout_fragment_length x num_workers x num_envs_per_worker).
        'gamma': 0.8,
        'twin_q' : True,
        "lr": 3e-4,
        "min_sample_timesteps_per_iteration": 1024,
    }

    from src.customEnv.wrapper import L5Env2WrapperTorchCLEReward


    rollout_sim_cfg = SimulationConfigGym()
    rollout_sim_cfg.num_simulation_steps = None

    # Register , how your env should be constructed (always with 5, or you can take values from the `config` EnvContext object):
    env_kwargs = {'env_config_path': env_config_path, 
                'use_kinematic': True, 
                'sim_cfg': rollout_sim_cfg,  
                'train': False,
                'return_info': True}

    reward_kwargs = {
        'yaw_weight': 1.0,
        'dist_weight': 1.0,
        'cf_weight': 20.0,
        'cr_weight': 20.0,
        'cs_weight': 20.0,
    }
    tune.register_env("L5-CLE-EVAL-V2", lambda config: L5Env2WrapperTorchCLEReward(L5Env2(**env_kwargs), reward_kwargs=reward_kwargs))
    rollout_env = L5Env2(**env_kwargs)

    from ray.rllib.algorithms.sac import SAC
    # checkpoint_path = '/home/pronton/ray_results/luanvan/SAC-T_RLfinetune_CLEreward_1e6buffer_rcollision=-20_shared_but_not_efficiency10-05-2023_02-37-05/SAC/SAC_L5-CLE-V2_e114c_00000_0_2023-05-09_19-37-05/checkpoint_000520'
    # checkpoint_path = '/home/pronton/ray_results/luanvan/SAC-T_nonload_nonfreeze_CLEreward_1e6buffer_rcollision=-20_11-05-2023_09-43-38/SAC/SAC_L5-CLE-V2_a24ab_00000_0_2023-05-11_02-43-39/checkpoint_000230'
    # checkpoint_path = '/home/pronton/ray_results/luanvan/changedReward/debug/SAC-T_RLFT_CLEreward_1e6buffer_shared12-05-2023_11-52-54/SAC/SAC_L5-CLE-V2_dbb63_00000_0_2023-05-12_04-52-55/checkpoint_000620'
    # checkpoint_path = '/home/pronton/ray_results/luanvan/changedReward/debug/SAC-T_nonload_nonfreeze_CLEreward_1e6buffer_non_shared13-05-2023_11-52-19/SAC/SAC_L5-CLE-V2_f14de_00000_0_2023-05-13_04-52-20/checkpoint_000490'
    checkpoint_path = '/home/pronton/ray_results/luanvan/KL_debug/SAC-T_RLFT_KL_nonKin_02-06-2023_17-09-32/KLSAC_2023-06-02_10-09-32/KLSAC_L5-CLE-V2_91b9a_00000_0_2023-06-02_10-09-32/checkpoint_000180'
    checkpoint_path = '/home/pronton/ray_results/luanvan/KL_debug/SAC-T_RLFT_KL_Kin_03-06-2023_19-48-46/KLSAC_2023-06-03_12-48-46/KLSAC_L5-CLE-V2_fa953_00000_0_2023-06-03_12-48-46/checkpoint_000180'
    checkpoint_path = '/home/pronton/ray_results/luanvan/fixedConfig/KLSAC-T_RLFT_fixedKLKin_fixedConfig10-06-2023_16-55-47/KLSAC_2023-06-10_09-55-50/KLSAC_L5-CLE-V2_faf40_00000_0_2023-06-10_09-55-50/checkpoint_000340'
    checkpoint_path = '/home/pronton/ray_results/luanvan/fixedConfig/KLSAC-T_nonRLFT_fixedKLKin_fixedConfig12-06-2023_09-57-59/KLSAC_2023-06-12_02-58-02/KLSAC_L5-CLE-V2_f20dc_00000_0_2023-06-12_02-58-02/checkpoint_000400'
    # checkpoint_path = '/home/pronton/ray_results/luanvan/KLweight/KLSAC-T_nonRLFT_fixedKLKin_fixedConfig_KLweight=2_cs_weight=40_15-06-2023_21-35-34/KLSAC_2023-06-15_14-35-36/KLSAC_L5-CLE-V2_e4660_00000_0_2023-06-15_14-35-36/checkpoint_000260'
    checkpoint_path = '/home/pronton/ray_results/luanvan/KLweight/KLSAC-T_nonRLFT_fixedKLKin_fixedConfig_KLweight=1_collision_weight=5_17-06-2023_08-43-48/KLSAC_2023-06-17_01-43-50/KLSAC_L5-CLE-V2_68eb3_00000_0_2023-06-17_01-43-51/checkpoint_000180'
    checkpoint_path = '/home/pronton/ray_results/luanvan/KLweight/KLRewardSAC-T_nonRLFT_fixedKLKin_KLweight=1_collision_weight=20_1e519-06-2023_23-21-01/KLRewardSAC_2023-06-19_16-21-03/KLRewardSAC_L5-CLE-V2_498a9_00000_0_2023-06-19_16-21-04/checkpoint_000320'
    checkpoint_path = '/home/pronton/ray_results/luanvan/KLweight/KLRewardSAC-T_nonRLFT_fixedKLKin_KLweight=1_collision_weight=20_1e519-06-2023_23-21-01/KLRewardSAC_2023-06-19_16-21-03/KLRewardSAC_L5-CLE-V2_498a9_00000_0_2023-06-19_16-21-04/checkpoint_000460'
    
    checkpoint_path = '/home/pronton/ray_results/luanvan/KLweight/KLRewardSAC-T_nonRLFT_fixedKLKin_KLweight=1_collision_weight=20_1e519-06-2023_23-21-01/KLRewardSAC_2023-06-19_16-21-03/KLRewardSAC_L5-CLE-V2_498a9_00000_0_2023-06-19_16-21-04/checkpoint_000510'

    # checkpoint_path = '/home/pronton/ray_results/luanvan/KLweight/SAC-T_nonRLFT_3histConfig_collision_weight=20_1e620-06-2023_15-52-25/SAC/SAC_L5-CLE-V2_c8c0f_00000_0_2023-06-20_08-52-28/checkpoint_000240'

    checkpoint_path = '/home/pronton/ray_results/luanvan/KLweight/KLSAC-T_nonRLFT_fixedKLKin_fixedConfig_KLweight=1_13-06-2023_23-52-03/SAC_2023-06-13_16-52-05/KLSAC_L5-CLE-V2_a0926_00000_0_2023-06-13_16-52-05/checkpoint_000200'

    # checkpoint_path = '/home/pronton/ray_results/luanvan/KLweight/KLRewardSAC-T_nonRLFT_fixedKLKin_KLweight=1_only_collision_weight=2021-06-2023_08-22-56/KLRewardSAC_2023-06-21_01-22-59/KLRewardSAC_L5-CLE-V2_28759_00000_0_2023-06-21_01-22-59/checkpoint_000066'
    checkpoint_path = '/home/pronton/ray_results/luanvan/KLweight/KLRewardSAC-T_shared_freeze_KLWeight=1_22-06-2023_01-52-33/KLRewardSAC_2023-06-21_18-52-35/KLRewardSAC_L5-CLE-V2_c93ae_00000_0_2023-06-21_18-52-35/checkpoint_000290'
    checkpoint_path = '/home/pronton/ray_results/luanvan/KLweight/KLRewardSAC-T_noFT_KLweight=.5_23-06-2023_01-15-05/KLRewardSAC_2023-06-22_18-15-08/KLRewardSAC_L5-CLE-V2_b83fc_00000_0_2023-06-22_18-15-08/checkpoint_000400'
    checkpoint_path = '/home/pronton/ray_results/luanvan/KLweight/SAC-T_shared__KLWeight=.5_24-06-2023_00-58-04/KLRewardSAC_2023-06-23_17-58-06/KLRewardSAC_L5-CLE-V2_817c8_00000_0_2023-06-23_17-58-06/checkpoint_000460'
    checkpoint_path = '/home/pronton/ray_results/luanvan/KLweight/SAC-T_shared__KLWeight=.5_24-06-2023_00-58-04/KLRewardSAC_2023-06-23_17-58-06/KLRewardSAC_L5-CLE-V2_817c8_00000_0_2023-06-23_17-58-06/checkpoint_000290'

    checkpoint_path = '/home/pronton/ray_results/luanvan/KLweight/KLRewardSAC-T_noFT_KLweight=.9_24-06-2023_16-31-23/KLRewardSAC_2023-06-24_09-31-25/KLRewardSAC_L5-CLE-V2_e3d58_00000_0_2023-06-24_09-31-25/checkpoint_000490'

    checkpoint_path = '/home/pronton/ray_results/luanvan/KLweight/KLRewardSAC-T_noFT_KLweight=1_valset_cr3025-06-2023_19-05-32/KLRewardSAC_2023-06-25_12-05-34/KLRewardSAC_L5-CLE-V2_96f3c_00000_0_2023-06-25_12-05-34/checkpoint_000230'
    
    checkpoint_path = '/home/pronton/ray_results/luanvan/KLweight/KLRewardSAC-T_FT_KLWeight=1_trainset_cr30_27-06-2023_23-13-26/KLRewardSAC_2023-06-27_16-13-28/KLRewardSAC_L5-CLE-V2_8d1c1_00000_0_2023-06-27_16-13-28/checkpoint_000510'    
    checkpoint_path = '/home/pronton/ray_results/luanvan/KLweight/KLRewardSAC-T_FT_KLweight=1_trainset_28-06-2023_10-41-11/KLRewardSAC_2023-06-28_03-41-13/KLRewardSAC_L5-CLE-V2_a1109_00000_0_2023-06-28_03-41-13/checkpoint_000180'

    
    # checkpoint_path = '/home/pronton/ray_results/luanvan/SAC-T-FT_04-05-2023_22-00-18/SAC/SAC_L5-CLE-V2_62800_00000_0_2023-05-04_15-00-18/checkpoint_000630'
    checkpoint_path = '/home/pronton/ray_results/debug/luanvan/KLweight/KLRewardSAC-T_load_KLweight=1_trainset_28-06-2023_21-19-57/KLRewardSAC_2023-06-28_14-19-59/KLRewardSAC_L5-CLE-V2_dd401_00000_0_2023-06-28_14-19-59/checkpoint_000600'

    checkpoint_path = '/home/pronton/ray_results/luanvan/KLweight/KLRewardSAC-T_noFT_KLweight=1_trainset_28-06-2023_10-41-11/KLRewardSAC_2023-06-28_03-41-13/KLRewardSAC_L5-CLE-V2_a1109_00000_0_2023-06-28_03-41-13/checkpoint_000170'
    checkpoint_path = '/home/pronton/ray_results/debug/luanvan/KLweight/KLRewardSAC-T_load_KLweight=1_decay_trainset_1e6_adaptiveWeight04-07-2023_10-09-48/KLRewardSAC_2023-07-04_03-09-51/KLRewardSAC_L5-CLE-V2_3da6d_00000_0_2023-07-04_03-09-51/checkpoint_000180'
    # checkpoint_path = '/home/pronton/ray_results/luanvan/KLweight/KLRewardSAC-T_FT_KLWeight=1_valset_cr30_26-06-2023_16-49-56/KLRewardSAC_2023-06-26_09-49-58/KLRewardSAC_L5-CLE-V2_cfea2_00000_0_2023-06-26_09-49-58/checkpoint_000510'
    # checkpoint_path = '/home/pronton/ray_results/debug/luanvan/KLweight/KLRewardSAC-T_load_KLweight=1_decay_trainset_1e6_01-07-2023_09-40-32/KLRewardSAC_2023-07-01_02-40-35/KLRewardSAC_L5-CLE-V2_a7cdc_00000_0_2023-07-01_02-40-35/checkpoint_000293'
    # checkpoint_path = '/home/pronton/ray_results/luanvan/KLweight/COPY_KLSAC-T_nonRLFT_fixedKLKin_fixedConfig_KLweight=1_13-06-2023_23-52-03/SAC_2023-06-13_16-52-05/KLSAC_L5-CLE-V2_a0926_00000_0_2023-06-13_16-52-05/checkpoint_000500'
    checkpoint_path = '/home/pronton/ray_results/debug/luanvan/KLweight/COPY-KLRewardSAC-T_load_KLweight=1_trainset_5e5_05-07-2023_14-38-09/KLRewardSAC_2023-07-05_07-38-11/KLRewardSAC_L5-CLE-V2_e4bed_00000_0_2023-07-05_07-38-11/checkpoint_000290'
    checkpoint_path=  '/home/pronton/ray_results/debug/luanvan/KLweight/KLRewardSAC-T_load_KLweight=1_trainset_1e6_steer=-2_06-07-2023_16-46-38/KLRewardSAC_2023-07-06_09-46-40/KLRewardSAC_L5-CLE-V2_021bd_00000_0_2023-07-06_09-46-40/checkpoint_000200'
    checkpoint_path = '/home/pronton/ray_results/debug/luanvan/KLweight/KLRewardSAC-T_load_KLweight=.9_trainset_2e5_steer=-1.5_08-07-2023_12-33-52/KLRewardSAC_2023-07-08_05-33-54/KLRewardSAC_L5-CLE-V2_0773b_00000_0_2023-07-08_05-33-55/checkpoint_000180'
    # checkpoint_path = '/home/pronton/ray_results/debug/luanvan/KLweight/KLRewardSAC-T_load_KLweight=1_trainset_2e5_steer=-1.5_07-07-2023_14-36-36/KLRewardSAC_2023-07-07_07-36-38/KLRewardSAC_L5-CLE-V2_01e60_00000_0_2023-07-07_07-36-38/checkpoint_000280'
    checkpoint_path = '/home/pronton/ray_results/debug/luanvan/KLweight/KLRewardSAC-T_load_KLweight=1_trainset_2e5_steer=-1.5_08-07-2023_23-20-56/KLRewardSAC_2023-07-08_16-20-58/KLRewardSAC_L5-CLE-V2_6bfe8_00000_0_2023-07-08_16-20-58/checkpoint_000140'
    checkpoint_path = '/home/pronton/ray_results/luanvan/KLweight/KLSAC-T_nonRLFT_fixedKLKin_fixedConfig_KLweight=1_13-06-2023_23-52-03/SAC_2023-06-13_16-52-05/KLSAC_L5-CLE-V2_a0926_00000_0_2023-06-13_16-52-05/checkpoint_000130'
    # checkpoint_path = '/home/pronton/ray_results/luanvan/SAC-T-noFT-hist3-1e6_13-05-2023_11-52-19/SAC/SAC_L5-CLE-V2_f14de_00000_0_2023-05-13_04-52-20/checkpoint_000520'
    # checkpoint_path = '/home/pronton/ray_results/debug/luanvan/KLweight/KLRewardSAC-T_load_KLweight=1_trainset_2e5_steer=-109-07-2023_16-23-32/KLRewardSAC_2023-07-09_09-23-34/KLRewardSAC_L5-CLE-V2_4732c_00000_0_2023-07-09_09-23-34/checkpoint_000150'
    # checkpoint_path = ''
    # checkpoint_path = '/home/pronton/ray_results/debug/luanvan/KLweight/KLRewardSAC-T_load_KLweight=.8_trainset_2e5_steer=-1.510-07-2023_13-10-31/KLRewardSAC_2023-07-10_06-10-33/KLRewardSAC_L5-CLE-V2_7ae6e_00000_0_2023-07-10_06-10-33/checkpoint_000150'
    # checkpoint_path = '/home/pronton/ray_results/debug/luanvan/KLweight/KLRewardSAC-T_load_KLweight=1_trainset_2e5_steer=-109-07-2023_16-23-32/KLRewardSAC_2023-07-09_09-23-34/KLRewardSAC_L5-CLE-V2_4732c_00000_0_2023-07-09_09-23-34/checkpoint_000190'
    # checkpoint_path = '/home/pronton/ray_results/debug/luanvan/KLweight/KLRewardSAC-T_load_KLweight=1_trainset_2e5_steer=-1.513-07-2023_00-38-59/KLRewardSAC_2023-07-12_17-39-01/KLRewardSAC_L5-CLE-V2_fd3bd_00000_0_2023-07-12_17-39-02/checkpoint_000206'
    # checkpoint_path = 'ray_results/debug/luanvan/SAC-T_load_trainset_2e5-1hist16-07-2023_00-26-21/SAC/SAC_L5-CLE-V2_b735e_00000_0_2023-07-15_17-26-21/checkpoint_000270'
    # checkpoint_path = '/home/pronton/ray_results/debug/luanvan/SAC-T_load_trainset_2e5-1hist13-07-2023_15-29-11/SAC/SAC_L5-CLE-V2_57c4d_00000_0_2023-07-13_08-29-11/checkpoint_000160'
    # 100->120, 1
    cp = 280
    r = -58
    # cp = 270
    # r = -62
    # cp = 260
    # r = -70
    # cp = 250
    # r = -59
    # cp = 240
    # r = -59
    # cp = 230
    # r = -66
    # cp = 220
    # r = -73
    cp = 206
    r = -77
    # cp = 200
    # r = -76
    # cp = 190
    # r = -77
    # cp = 180
    # r = -75
    # cp = 170
    # r = -76
    # cp = 160
    # r = -81
    # cp = 150
    # r = -75
    # cp = 140
    # r = -88
    # cp = 130
    # r = -103
    # cp = 120
    # r = -82
    # cp = 110
    # r = -81
    # cp = 100
    # r = -90
    # cp = '090'
    # r = -78
    # cp = '080'
    # r = -88
    # checkpoint_path = f'/home/pronton/ray_results/fixedInversed_l5env2/luanvan/KLRewardSAC-T_load_KLweight=1_trainset_2e5_steer=-1.5_24-07-2023_23-29-24/KLRewardSAC_2023-07-24_16-29-26/KLRewardSAC_L5-CLE-V2_41cad_00000_0_2023-07-24_16-29-27/checkpoint_000{cp}'
    # checkpoint_path = f'/home/pronton/ray_results/debug/KLSAC_25-07-2023_13-23-34/KLSAC_2023-07-25_06-23-36/KLSAC_L5-CLE-V2_c9a70_00000_0_2023-07-25_06-23-36/checkpoint_000{cp}'
    # checkpoint_path = f'/home/pronton/ray_results/debug/luanvan/KLweight/KLRewardSAC-T_load_KLweight=.3_trainset_2e5_steer=-1.511-07-2023_23-49-49/KLRewardSAC_2023-07-11_16-49-52/KLRewardSAC_L5-CLE-V2_f49ee_00000_0_2023-07-11_16-49-52/checkpoint_000{cp}'
    # checkpoint_path = f'/home/pronton/ray_results/fixedInversed_l5env2/luanvan/KLRewardSAC-T_load_KLweight=.1_trainset_2e5_steer=-1.5_26-07-2023_23-00-29/KLRewardSAC_2023-07-26_16-00-31/KLRewardSAC_L5-CLE-V2_8c0f4_00000_0_2023-07-26_16-00-31/checkpoint_000{cp}'
    # checkpoint_path = f'/home/pronton/ray_results/debug/luanvan/KLweight/KLRewardSAC-T_load_KLweight=.5_trainset_2e5_steer=-1.511-07-2023_00-28-28/KLRewardSAC_2023-07-10_17-28-31/KLRewardSAC_L5-CLE-V2_30726_00000_0_2023-07-10_17-28-31/checkpoint_000{cp}'
    # checkpoint_path = f'/home/pronton/ray_results/fixedInversed_l5env2/luanvan/KLRewardSAC-T_load_KLweight=.5_trainset_2e5_steer=-1.5_28-07-2023_02-04-36/KLRewardSAC_2023-07-27_19-04-39/KLRewardSAC_L5-CLE-V2_6f8d0_00000_0_2023-07-27_19-04-39/checkpoint_000{cp}'
    checkpoint_path = f'/home/pronton/ray_results/fixedInversed_l5env2/luanvan/KLRewardSAC-T_load_KLweight=.1_trainset_2e5_steer=-1.5_26-07-2023_23-00-29/KLRewardSAC_2023-07-26_16-00-31/KLRewardSAC_L5-CLE-V2_8c0f4_00000_0_2023-07-26_16-00-31/checkpoint_000{cp}'
    checkpoint_path = f'/home/pronton/ray_results/debug/luanvan/KLweight/KLRewardSAC-T_load_KLweight=.5_trainset_2e5_steer=-1.511-07-2023_00-28-28/KLRewardSAC_2023-07-10_17-28-31/KLRewardSAC_L5-CLE-V2_30726_00000_0_2023-07-10_17-28-31/checkpoint_000{cp}'
    checkpoint_path = f'/home/pronton/ray_results/debug/luanvan/SAC-T_load_trainset_2e5-1hist13-07-2023_15-29-11/SAC/SAC_L5-CLE-V2_57c4d_00000_0_2023-07-13_08-29-11/checkpoint_000{cp}'
    checkpoint_path = f'/home/pronton/ray_results/fixedInversed_l5env2/luanvan/KLRewardSAC-T_load_KLweight=.3_trainset_1e5_steer=-1.5_01-08-2023_10-53-25/KLRewardSAC_2023-08-01_03-53-27/KLRewardSAC_L5-CLE-V2_f8e53_00000_0_2023-08-01_03-53-27/checkpoint_000{cp}'
    checkpoint_path = f'/home/pronton/ray_results/fixedInversed_l5env2/luanvan/KLRewardSAC-T_load_KLweight=.7_trainset_1e5_steer=-1.5_01-08-2023_01-08-26/KLRewardSAC_2023-07-31_18-08-28/KLRewardSAC_L5-CLE-V2_405af_00000_0_2023-07-31_18-08-29/checkpoint_000{cp}'
    checkpoint_path = f'/home/pronton/ray_results/fixedInversed_l5env2/luanvan/KLRewardSAC-T_load_KLweight=.8_trainset_1e5_steer=-1.5_02-08-2023_23-29-12/KLRewardSAC_2023-08-02_16-29-14/KLRewardSAC_L5-CLE-V2_b8008_00000_0_2023-08-02_16-29-14/checkpoint_000{cp}'

    checkpoint_path = f'/home/pronton/ray_results/fixedInversed_l5env2/luanvan/KLRewardSAC-T_load_KLweight=.3_trainset_1e5_acc_steer=-1.5_03-08-2023_17-20-12/KLRewardSAC_2023-08-03_10-20-14/KLRewardSAC_L5-CLE-V2_55ee6_00000_0_2023-08-03_10-20-14/checkpoint_000{cp}'
    checkpoint_path = f'/home/pronton/ray_results/fixedInversed/luanvan/KLRewardSAC-T_load_KLweight=.3_trainset_1e5_acc_steer=-1.5_05-08-2023_01-07-53/KLRewardSAC_2023-08-04_18-07-55/KLRewardSAC_L5-CLE-V2_d64b4_00000_0_2023-08-04_18-07-56/checkpoint_000{cp}'
    checkpoint_path = f'/home/pronton/ray_results/luanvan/KLRewardSAC-T_load_KLweight=.3_trainset_1e5_acc_steer=-1.5_05-08-2023_15-17-05/KLRewardSAC_2023-08-05_08-17-07/KLRewardSAC_L5-CLE-V2_77d5f_00000_0_2023-08-05_08-17-07/checkpoint_000{cp}'
    checkpoint_path = f'/home/pronton/ray_results/luanvan/KLRewardSAC-T_load_KLweight=.5_trainset_1e5_acc=-1.5_08-08-2023_00-16-49/KLRewardSAC_2023-08-07_17-16-51/KLRewardSAC_L5-CLE-V2_33161_00000_0_2023-08-07_17-16-51/checkpoint_000{cp}'
    checkpoint_path = f'/home/pronton/ray_results/luanvan/KLRewardSAC-nonkin-T_load_KLweight=.5_trainset_1e5_acc=-1.5_09-08-2023_10-19-28/SAC_2023-08-09_03-19-30/SAC_L5-CLE-V2_8df1b_00000_0_2023-08-09_03-19-30/checkpoint_000{cp}'
    checkpoint_path = f'/home/pronton/ray_results/fixedInversed_l5env2/luanvan/KLSAC-T_load_acc=-1.5_MTAU=entropy_alpha_M_ALPHA=0.8_bs12810-08-2023_13-54-09/KLSAC_2023-08-10_06-54-11/KLSAC_L5-CLE-V2_b6035_00000_0_2023-08-10_06-54-11/checkpoint_000{cp}'
    checkpoint_path = f'/home/pronton/ray_results/fixedInversed_l5env2/luanvan/KLSAC-T_load_acc=-1.5_MTAU=entropy_alpha_M_ALPHA=0.3_batch12815-08-2023_00-48-44/KLSAC_2023-08-14_17-48-47/KLSAC_L5-CLE-V2_d1da6_00000_0_2023-08-14_17-48-47/checkpoint_000{cp}'
    checkpoint_path = f'/home/pronton/ray_results/fixedInversed_l5env2/luanvan/KLSAC-T_load_acc=-1.5_MTAU=entropy_alpha_M_ALPHA=0.3_batch12816-08-2023_00-24-58/KLSAC_2023-08-15_17-25-00/KLSAC_L5-CLE-V2_a9c72_00000_0_2023-08-15_17-25-00/checkpoint_000{cp}'
    checkpoint_path = f'/home/pronton/ray_results/fixedInversed_l5env2/luanvan/KLSAC-T_load_acc=-1.5_MTAU=entropy_alpha_M_ALPHA=1_batch12816-08-2023_18-54-00/KLSAC_2023-08-16_11-54-02/KLSAC_L5-CLE-V2_98183_00000_0_2023-08-16_11-54-02/checkpoint_000{cp}'
    checkpoint_path = f'/home/pronton/ray_results/debug/fixedInversed_l5env2/luanvan/KLSAC-T_load_acc=-1.5_MTAU=alphaEntropy=0.03_M_ALPHA=0.9_batch256_17-08-2023_17-31-14/KLSAC_2023-08-17_10-31-16/KLSAC_L5-CLE-V2_3292c_00000_0_2023-08-17_10-31-17/checkpoint_000{cp}'
    checkpoint_path = f'/home/pronton/ray_results/debug/fixedInversed_l5env2/luanvan/KLSAC-T_load_acc=-1.5_MTAU=alphaEntropy(constant)=0.03_M_ALPHA=0.9_batch256_18-08-2023_11-55-34/KLSAC_2023-08-18_04-55-36/KLSAC_L5-CLE-V2_787db_00000_0_2023-08-18_04-55-36/checkpoint_000{cp}'
    checkpoint_path = f'/home/pronton/ray_results/debug/fixedInversed_l5env2/luanvan/KLSAC-T_load_acc=-1.5_MTAU=alphaEntropy(constant)=0.3_M_ALPHA=0.9_batch256_19-08-2023_00-03-13/KLSAC_2023-08-18_17-03-15/KLSAC_L5-CLE-V2_1f348_00000_0_2023-08-18_17-03-15/checkpoint_000{cp}'
    checkpoint_path = f'/home/pronton/ray_results/debug/fixedInversed_l5env2/luanvan/KLSAC-T_load_acc=-1.5_MTAU=alphaEntropy(constant)=0.5_M_ALPHA=0.9_batch256_19-08-2023_14-52-50/KLSAC_2023-08-19_07-52-52/KLSAC_L5-CLE-V2_66a26_00000_0_2023-08-19_07-52-53/checkpoint_000{cp}'
    checkpoint_path = f'/home/pronton/ray_results/debug/fixedInversed_l5env2/luanvan/KLSAC-T_load_acc=-1.5_MTAU=alphaEntropy(constant)=0.7_M_ALPHA=0.9_batch256_20-08-2023_00-47-46/KLSAC_2023-08-19_17-47-49/KLSAC_L5-CLE-V2_83789_00000_0_2023-08-19_17-47-49/checkpoint_000{cp}'
    checkpoint_path = f'/home/pronton/ray_results/debug/fixedInversed_l5env2/luanvan/KLSAC-T_load_acc=-1.5_MTAU=alphaEntropy(constant)=0.5_M_ALPHA=0.5_batch256_20-08-2023_23-48-00/KLSAC_2023-08-20_16-48-03/KLSAC_L5-CLE-V2_545e2_00000_0_2023-08-20_16-48-03/checkpoint_000{cp}'
    checkpoint_path = f'/home/pronton/ray_results/debug/fixedInversed_l5env2/luanvan/KLSAC-T_load_acc=-1.5_MTAU=alphaEntropy(constant)=0.5_M_ALPHA=0.7_batch256_20-08-2023_13-47-27/KLSAC_2023-08-20_06-47-30/KLSAC_L5-CLE-V2_6ee18_00000_0_2023-08-20_06-47-30/checkpoint_000{cp}'
    checkpoint_path = f'/home/pronton/ray_results/debug/fixedInversed_l5env2/luanvan/KLSAC-T_load_acc=-1.5_MTAU=alphaEntropy(constant)=0.7_M_ALPHA=0.5_batch256_21-08-2023_12-26-01/KLSAC_2023-08-21_05-26-04/KLSAC_L5-CLE-V2_38f9f_00000_0_2023-08-21_05-26-04/checkpoint_000{cp}'
    checkpoint_path = f'/home/pronton/ray_results/debug/fixedInversed_l5env2/luanvan/KLSAC-T_load_acc=-1.5_MTAU=alphaEntropy(constant)=0.9_M_ALPHA=0.5_batch256_22-08-2023_00-03-57/KLSAC_2023-08-21_17-04-00/KLSAC_L5-CLE-V2_b928c_00000_0_2023-08-21_17-04-00/checkpoint_000{cp}'
    checkpoint_path = f'/home/pronton/ray_results/fixedInversed_l5env2/luanvan/KLSAC-T_load_acc=-1.5_alphaEntropy(constant)=M_TAU_M_ENTROPY=0.5_M_KL=0.3_batch256_22-08-2023_15-27-19/KLSAC_2023-08-22_08-27-21/KLSAC_L5-CLE-V2_b6ca3_00000_0_2023-08-22_08-27-21/checkpoint_000{cp}'
    checkpoint_path = f'/home/pronton/ray_results/fixedInversed_l5env2/luanvan/KLSAC-T_load_acc=-1.5_alphaEntropy(constant)=M_TAU_M_ENTROPY=0.5_M_KL=0.5_batch256_23-08-2023_01-06-29/KLSAC_2023-08-22_18-06-32/KLSAC_L5-CLE-V2_a0117_00000_0_2023-08-22_18-06-32/checkpoint_000{cp}'
    checkpoint_path = f'/home/pronton/ray_results/fixedInversed_l5env2/luanvan/KLSAC-T_load_acc=-1.5_alphaEntropy(constant)=M_ENTROPY=0.7_M_KL=0.5_batch256_23-08-2023_23-07-55/KLSAC_2023-08-23_16-07-57/KLSAC_L5-CLE-V2_39ba4_00000_0_2023-08-23_16-07-57/checkpoint_000{cp}'
    # checkpoint_path = f'/home/pronton/ray_results/fixedInversed_l5env2/luanvan/KLSAC-T_load_acc=-1.5_alphaEntropy(constant)=M_ENTROPY=1_M_KL=1_batch256_24-08-2023_10-01-17/KLSAC_2023-08-24_03-01-19/KLSAC_L5-CLE-V2_7ff58_00000_0_2023-08-24_03-01-19/checkpoint_000{cp}'
    checkpoint_path = f'/home/pronton/ray_results/fixedInversed_l5env2/luanvan/KLSAC-T_load_acc=-1.5_alphaEntropy(constant)=M_ENTROPY=0.9_M_KL=0.7_batch256_24-08-2023_22-05-20/KLSAC_2023-08-24_15-05-22/KLSAC_L5-CLE-V2_a5dc4_00000_0_2023-08-24_15-05-22/checkpoint_000{cp}'
    checkpoint_path = f'/home/pronton/ray_results/fixedInversed_l5env2/luanvan/KLSAC-T_load_acc=-1.5_alphaEntropy(constant)=M_ENTROPY=0.7_M_KL=0.7_batch256_25-08-2023_11-47-09/KLSAC_2023-08-25_04-47-12/KLSAC_L5-CLE-V2_74c03_00000_0_2023-08-25_04-47-12/checkpoint_000{cp}'
    checkpoint_path = f'/home/pronton/ray_results/debug/fixedInversed_l5env2/luanvan/KLSAC-T_load_acc=-1.5_alphaEntropy(constant)=M_ENTROPY=0.1_M_KL=0.7_batch256_26-08-2023_11-36-26/KLSAC_2023-08-26_04-36-29/KLSAC_L5-CLE-V2_1fec0_00000_0_2023-08-26_04-36-29/checkpoint_000{cp}'
    checkpoint_path = f'/home/pronton/ray_results/luanvan/SAC-T_load_trainset_2e5-1hist13-07-2023_15-29-11/SAC/SAC_L5-CLE-V2_57c4d_00000_0_2023-07-13_08-29-11/checkpoint_000{cp}'
    checkpoint_path = f'/home/pronton/ray_results/fixedInversed_l5env2/luanvan/KLRewardSAC-T_load_KLweight=.3_trainset_2e5_acc=-1.511-07-2023_23-49-49/KLRewardSAC_2023-07-11_16-49-52/KLRewardSAC_L5-CLE-V2_f49ee_00000_0_2023-07-11_16-49-52/checkpoint_000{cp}'
    checkpoint_path = f'/home/pronton/ray_results/debug/fixedInversed_l5env2/luanvan/KLSAC-T_load_acc=-1.5_alphaEntropy(constant)=M_ENTROPY=0.7_M_KL=0.25_batch256_28-08-2023_01-07-37/KLSAC_2023-08-27_18-07-40/KLSAC_L5-CLE-V2_9c777_00000_0_2023-08-27_18-07-40/checkpoint_000{cp}'
    checkpoint_path = f'/home/pronton/ray_results/debug/fixedInversed_l5env2/luanvan/KLSAC-T_load_acc=-1.5_alphaEntropy(constant)=M_ENTROPY=0.7_M_KL=0_batch256_27-08-2023_01-06-26/KLSAC_2023-08-26_18-06-28/KLSAC_L5-CLE-V2_47750_00000_0_2023-08-26_18-06-28/checkpoint_000{cp}'
    checkpoint_path = f'/home/pronton/ray_results/fixedInversed_l5env2/luanvan/KLSAC-T_load_acc=-1.5_alphaEntropy(constant)=M_ENTROPY=0.7_M_KL=0.6_batch256_29-08-2023_01-02-17/KLSAC_2023-08-28_18-02-20/KLSAC_L5-CLE-V2_0842c_00000_0_2023-08-28_18-02-20/checkpoint_000{cp}'
    checkpoint_path = f'/home/pronton/ray_results/fixedInversed_l5env2/luanvan/KLSAC-T_load_acc=-1.5_alphaEntropy(constant)=M_ENTROPY=0.7_M_KL=0.9_batch256_28-08-2023_15-09-21/KLSAC_2023-08-28_08-09-23/KLSAC_L5-CLE-V2_331c3_00000_0_2023-08-28_08-09-24/checkpoint_000{cp}'
    checkpoint_path = f'/home/pronton/ray_results/fixedInversed_l5env2/luanvan/KLSAC-T_load_acc=-1.5_alphaEntropy(constant)=M_ENTROPY=0.7_M_KL=0.1_batch256_29-08-2023_23-58-56/KLSAC_2023-08-29_16-58-59/KLSAC_L5-CLE-V2_58e65_00000_0_2023-08-29_16-58-59/checkpoint_000{cp}'
    checkpoint_path = f'/home/pronton/ray_results/fixedInversed_l5env2/luanvan/KLSAC-T_load_acc=-1.5_alphaEntropy(constant)=M_ENTROPY=0.7_M_KL=0.4_batch256_29-08-2023_14-07-42/KLSAC_2023-08-29_07-07-44/KLSAC_L5-CLE-V2_c0709_00000_0_2023-08-29_07-07-44/checkpoint_000{cp}'
    checkpoint_path = f'/home/pronton/ray_results/fixedInversed_l5env2/luanvan/KLSAC-T_load_acc=-1.5_alphaEntropy(constant)=M_ENTROPY=0.3_M_KL=0.5_batch256_30-08-2023_12-00-38/KLSAC_2023-08-30_05-00-41/KLSAC_L5-CLE-V2_2b01a_00000_0_2023-08-30_05-00-41/checkpoint_000{cp}'
    checkpoint_path = f'/home/pronton/ray_results/fixedInversed_l5env2/luanvan/KLSAC-T_load_acc=-1.5_alphaEntropy(constant)=M_ENTROPY=0.9_M_KL=0.5_batch256_30-08-2023_21-32-00/KLSAC_2023-08-30_14-32-03/KLSAC_L5-CLE-V2_fc95e_00000_0_2023-08-30_14-32-03/checkpoint_000{cp}'
    name = f'KLSAC-T_load_acc=-1.5_alphaEntropy(constant)=M_ENTROPY=0.9_M_KL=0.5_batch256'

    print(f'-----------------------> {name}_cp{cp}({r})')

    # model = SAC(config=eval_config_param_space, env='L5-CLE-EVAL-V2')
    # model = SAC(config=eval_config_param_space, env='L5-CLE-EVAL-V2')
    model = KLSAC(config=eval_config_param_space, env='L5-CLE-EVAL-V2')
    model.restore(checkpoint_path)

    from src.simulation.unrollGym import unroll_to_quantitative
    sim_outs, actions, indices, results = unroll_to_quantitative(model, rollout_env, 100)
    
    print(f'{name}_cp{cp}({r})')

    from src.validate.validator import save_data
    file1 = open("/home/pronton/rlhf-car/results.csv", "a")  # append mode
    listResults = [f'{name}_cp{cp}({r})'] + results
    file1.write(str(listResults)[2:-1]+'\n')
    file1.close()
 

    if not os.path.exists(f'{SRC_PATH}src/validate/testset/{name}'):
        os.mkdir(f'{SRC_PATH}src/validate/testset/{name}')
    # save_data(sim_outs, f'{SRC_PATH}src/validate/testset/{name}/cp{cp}({r}).obj')
    save_data((actions, indices), f'{SRC_PATH}src/validate/testset/{name}/cp{cp}({r})_actions.obj')
    

else: 
    import pickle
    cp = 280
    r = -58
    # cp = 270
    # r = -62
    # cp = 260
    # r = -70
    # cp = 250
    # r = -59
    # cp = 240
    # r = -59
    # cp = 230
    # r = -66
    # cp = 220
    # r = -73
    # cp = 210
    # r = -67
    # cp = 200
    # r = -67
    # cp = 190
    # r = -66
    # cp = 180
    # r = -57
    # cp = 170
    # r = -67
    # cp = 160
    # r = -66
    # cp = 150
    # r = -67
    name = f'KLSAC-T_load_acc=-1.5_alphaEntropy(constant)=M_ENTROPY=0.7_M_KL=0.7_batch256'
    name = f'SAC-T_load_trainset_2e5-1hist'
    name = f'KLRewardSAC-T_load_KLweight=.3_trainset_2e5_acc=-1.5'
    file = open(f'{SRC_PATH}src/validate/testset/{name}/cp{cp}({r}).obj', 'rb')
    sim_outs = pickle.load(file)
    file = open(f'{SRC_PATH}src/validate/testset/{name}/cp{cp}({r})_actions.obj', 'rb')
    actions, indices = pickle.load(file)
    from src.validate.validator import quantitative
    results = quantitative(sim_outs, actions)
    file1 = open("/home/pronton/rlhf-car/results.csv", "a")  # append mode
    listResults = [f'{name}_cp{cp}({r})'] + results
    file1.write(str(listResults)[2:-1]+'\n')
    file1.close()
    print(f'{name}_cp{cp}({r})')
    # idx = 0
    # print(np.asarray(actions)[:indices[idx+1], 1])
