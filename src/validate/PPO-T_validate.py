import os
from src.customModel.customModel import TorchAttentionModel3, TorchVectorPPO

from src.constant import SRC_PATH
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

# ray.init(num_cpus=9, ignore_reinit_error=True, log_to_driver=False,  local_mode=False)


from l5kit.configs import load_config_data
LOADED = False
if not LOADED:
    reward_kwargs = {
        'yaw_weight': 1.0,
        'dist_weight': 1.0,
        # 'd2r_weight': 0.0,
        'cf_weight': 20.0,
        'cr_weight': 20.0,
        'cs_weight': 40.0,
    }

    env_config_path = SRC_PATH + 'src/configs/gym_vectorizer_config.yaml'
    # env_config_path = SRC_PATH + 'src/configs/gym_vectorizer_config_hist3.yaml'
    cfg = load_config_data(env_config_path)

    from ray import tune
    from src.customEnv.wrapper import L5EnvWrapper, L5EnvWrapperTorch, L5Env2WrapperTorchCLEReward
    ModelCatalog.register_custom_model( "TorchVectorPPO", TorchVectorPPO)


    train_envs = 4
    lr = 3e-3
    lr_start = 3e-4
    lr_end = 3e-5
    lr_time = int(4e6)

    config_param_space = {
        "framework": "torch",
        "num_gpus": 1,
        "num_workers": 8,
        "num_envs_per_worker": train_envs,
        "model": {
            "custom_model": "TorchVectorPPO",
            "custom_model_config": {
                'cfg':cfg,
                'freeze_for_RLtuning':  False,
                'load_pretrained': False,
                'shared_feature_extractor': False,
                'kl_div_weight': 1,
                'log_std_acc': -1,
                'log_std_steer': -1,
                'reward_kwargs': reward_kwargs,
            },
        },
        
        '_disable_preprocessor_api': True,
        "eager_tracing": True,
        "restart_failed_sub_environments": True,
        "lr": lr,
        'seed': 42,
        "lr_schedule": [
            [7e5, lr_start],
            [2e6, lr_end],
        ],
        'train_batch_size': 1024, # 8000 
        'sgd_minibatch_size': 512, #2048
        'num_sgd_iter': 10,#16,
        'batch_mode': 'truncate_episodes',
        # "rollout_fragment_length": 32,
        'gamma': 0.8,     
    }
    # checkpoint_path = '/home/pronton/ray_results/luanvan/PPO-T_nonfreeze_nonload08-05-2023_17-22-08/PPO/PPO_L5-CLE-V2_2fe02_00000_0_2023-05-08_10-22-08/checkpoint_000260'
    # checkpoint_path = '/home/pronton/ray_results/luanvan/PPO-T_RLfinetune_CLEreward_freeze_04-05-2023_08-52-42/PPO/PPO_L5-CLE-V2_5bde8_00000_0_2023-05-04_01-52-43/checkpoint_003260'
    # checkpoint_path = '/home/pronton/ray_results/luanvan/KL/PPO-T_freeze_load_CLEreward_shared_nonKin_KL31-05-2023_20-23-59/KLPPO_2023-05-31_13-23-59/KLPPO_L5-CLE-V2_671d9_00000_0_2023-05-31_13-23-59/checkpoint_004670'
    # checkpoint_path = '/home/pronton/ray_results/luanvan/KL_debug/PPO-T_freeze_load_CLEreward_shared_nonKin_nonKL31-05-2023_11-08-14/PPO/PPO_L5-CLE-V2_c42bd_00000_0_2023-05-31_04-08-15/checkpoint_002550'
    # checkpoint_path = '/home/pronton/ray_results/luanvan/KL_debug/PPO-T_freeze_load_CLEreward_shared_nonKin_KL01-06-2023_22-19-45/KLPPO_2023-06-01_15-19-45/KLPPO_L5-CLE-V2_bd5ab_00000_0_2023-06-01_15-19-45/checkpoint_003170'
    # checkpoint_path = '/home/pronton/ray_results/luanvan/KL_debug/PPO-T_RLFT_fixedKLKin_04-06-2023_16-17-56/KLPPO_2023-06-04_09-17-56/KLPPO_L5-CLE-V2_b179a_00000_0_2023-06-04_09-17-57/checkpoint_005350'
    checkpoint_path = '/home/pronton/ray_results/luanvan/KLweight/KLRewardPPO-T_FT_KLweight=1_valset_cr30_27-06-2023_07-04-11/KLRewardPPO_2023-06-27_00-04-11/KLRewardPPO_L5-CLE-V2_2534d_00000_0_2023-06-27_00-04-11/checkpoint_000720'
    checkpoint_path = '/home/pronton/ray_results/luanvan/KLweight/KLRewardPPO-T_noFT_KLweight=1_train9Gset_cr30_27-06-2023_08-50-16/KLRewardPPO_2023-06-27_01-50-16/KLRewardPPO_L5-CLE-V2_f7051_00000_0_2023-06-27_01-50-16/checkpoint_000610'
    checkpoint_path = '/home/pronton/ray_results/debug/KLRewardPPO-T_noFT_KLweight=1_train9Gset_cr30_adaptiveKL03-07-2023_23-46-02/KLRewardPPO_2023-07-03_16-46-02/KLRewardPPO_L5-CLE-V2_188bf_00000_0_2023-07-03_16-46-02/checkpoint_000590'

    #################### Define Training and Evaluation Environments ####################

    from ray import tune
    rollout_sim_cfg = SimulationConfigGym()
    rollout_sim_cfg.num_simulation_steps = None
    eval_env_kwargs = {'env_config_path': env_config_path, 'use_kinematic': True, 'sim_cfg': rollout_sim_cfg, 'train': False, 'return_info': True}
    # rollout_env = L5Env2(**eval_env_kwargs)
    # tune.register_env("L5-CLE-EVAL-V2", lambda config: L5Env2(**eval_env_kwargs))
    rollout_env = L5Env2WrapperTorchCLEReward(L5Env2(**eval_env_kwargs))
    tune.register_env("L5-CLE-EVAL-V2", lambda config: L5Env2WrapperTorchCLEReward(L5Env2(**eval_env_kwargs)))


    from ray.rllib.algorithms.ppo import PPO
    # from src.customModel.customPPOTrainer import KLPPO
    # model = PPO(config=config_param_space, env='L5-CLE-EVAL-V2')
    model = PPO(config=config_param_space, env='L5-CLE-EVAL-V2')
    model.restore(checkpoint_path)
    # from torchsummary import summary
    # summary(model.get_policy().model)


    from src.simulation.unrollGym import unroll_to_quantitative
    sim_outs = unroll_to_quantitative(model, rollout_env, 100)
    from src.validate.validator import save_data
    # quantitative(sim_outs)
    # name = 'KLRewardPPO-T_FT_KLweight=1_valset_cr30_cp720(-71)'
    name = 'KLRewardPPO-T_noFT_KLweight=1_train9Gset_cr30_adaptiveKL_cp590(-44)'
    print(name)
    save_data(sim_outs, f'{SRC_PATH}src/validate/testset/{name}.obj')

else:
    import pickle
    # name = 'KLRewardPPO-T_FT_KLweight=1_valset_cr30_cp720(-71)'
    name = 'KLRewardPPO-T_noFT_KLweight=1_train9Gset_cr30_cp610(-39)'
    file = open(f'{SRC_PATH}src/validate/testset/{name}.obj', 'rb')
    from src.validate.validator import quantitative
    sim_outs = pickle.load(file)
    quantitative(sim_outs)
    print(name)