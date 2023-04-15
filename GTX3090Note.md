## Installation: 
Private repo cloning: https://Pronton2001@github.com/Pronton2001/reponame
```bash
cd /path/to/l5kit/l5kit
pipenv sync --dev # pip install -e git+https://Pronton2001@github.com/Pronton2001/l5kit2@1bf5f8666850f7f8d915232c5bffaed9fb87643c#egg=l5kit&subdirectory=l5kit
pip install stable-baselines3==1.7
pip install wandb
pip install tensorflow-probability
pip install ray[rllib]==2.2.0
pip install tensorflow==2.11
pip install torch==1.13
p3 install setuptools==65.5.0
pip install gym==0.21.0 #(very important, l5kit can not run without it)
pip install numpy==1.22
```

## Use ssh tunneling:
```bash
ssh -N -L 8888:localhost:8888 root@172.17.0.2 # for jupyter lab
ssh -N -L 5006:localhost:5006 root@172.17.0.2 # for bokeh
```
# Transformer
1. Input: 
- ego: 1 history traj
- agents: 3 history traj
- polytypes:
    "AGENT_OF_INTEREST": 0,
    "AGENT_NO": 1, # Unknown agent
    "AGENT_CAR": 2,
    "AGENT_BIKE": 3,
    "AGENT_PEDESTRIAN": 4,
    "TL_UNKNOWN": 5,  # unknown TL state for lane
    "TL_RED": 6,
    "TL_YELLOW": 7,
    "TL_GREEN": 8,
    "TL_NONE": 9,  # no TL for lane
    "CROSSWALK": 10,
    "LANE_BDRY_LEFT": 11,
    "LANE_BDRY_RIGHT": 12,
PERCEPTION_LABELS = [
    "PERCEPTION_LABEL_NOT_SET",
    "PERCEPTION_LABEL_UNKNOWN",
    "PERCEPTION_LABEL_DONTCARE",
    "PERCEPTION_LABEL_CAR",
    "PERCEPTION_LABEL_VAN",
    "PERCEPTION_LABEL_TRAM",
    "PERCEPTION_LABEL_BUS",
    "PERCEPTION_LABEL_TRUCK",
    "PERCEPTION_LABEL_EMERGENCY_VEHICLE",
    "PERCEPTION_LABEL_OTHER_VEHICLE",
    "PERCEPTION_LABEL_BICYCLE",
    "PERCEPTION_LABEL_MOTORCYCLE",
    "PERCEPTION_LABEL_CYCLIST",
    "PERCEPTION_LABEL_MOTORCYCLIST",
    "PERCEPTION_LABEL_PEDESTRIAN",
    "PERCEPTION_LABEL_ANIMAL",
    "AVRESEARCH_LABEL_DONTCARE",
]
=> 17 types, 13 embedding type, but 13 
TL_FACE_LABELS = [
    "ACTIVE",
    "INACTIVE",
    "UNKNOWN",
]

2. data_batch:

dict_keys(['extent', 'type', 'agent_from_world', 'world_from_agent', 'target_positions', 'target_yaws', 'target_extents', 'target_availabilities', 'history_positions', 'history_yaws', 'history_extents', 'history_availabilities', 'centroid', 'yaw', 'speed', 'all_other_agents_history_positions', 'all_other_agents_history_yaws', 'all_other_agents_history_extents', 'all_other_agents_history_availability', 'all_other_agents_future_positions', 'all_other_agents_future_yaws', 'all_other_agents_future_extents', 'all_other_agents_future_availability', 'all_other_agents_types', 'agent_trajectory_polyline', 'agent_polyline_availability', 'other_agents_polyline', 'other_agents_polyline_availability', 'lanes', 'lanes_availabilities', 'lanes_mid', 'lanes_mid_availabilities', 'crosswalks', 'crosswalks_availabilities'])
k=extent| v=<class 'numpy.ndarray'>| shape=(3,)| min=1.8| max=4.87| mean=2.8400000000000003
k=type| v=<class 'int'>| shape=None| min=3| max=3| mean=3.0
k=agent_from_world| v=<class 'numpy.ndarray'>| shape=(3, 3)| min=-0.8367017118872029| max=1765.194471972579| mean=357.92625695995747
k=world_from_agent| v=<class 'numpy.ndarray'>| shape=(3, 3)| min=-2183.32763671875| max=680.6197509765625| mean=-166.73472977651232
k=target_positions| v=<class 'numpy.ndarray'>| shape=(12, 2)| min=-0.111104906| max=12.992863| mean=3.4934924
k=target_yaws| v=<class 'numpy.ndarray'>| shape=(12, 1)| min=-0.0191679| max=-0.00039577484| mean=-0.0091530485
k=target_extents| v=<class 'numpy.ndarray'>| shape=(12, 2)| min=1.85| max=4.87| mean=3.36
k=target_availabilities| v=<class 'numpy.ndarray'>| shape=(12,)| min=True| max=True| mean=1.0
k=history_positions| v=<class 'numpy.ndarray'>| shape=(4, 2)| min=0.0| max=2.2737368e-13| mean=2.842171e-14
k=history_yaws| v=<class 'numpy.ndarray'>| shape=(4, 1)| min=0.0| max=0.0| mean=0.0
k=history_extents| v=<class 'numpy.ndarray'>| shape=(4, 2)| min=0.0| max=4.87| mean=0.84
k=history_availabilities| v=<class 'numpy.ndarray'>| shape=(4,)| min=False| max=True| mean=0.25
k=centroid| v=<class 'numpy.ndarray'>| shape=(2,)| min=-2183.32763671875| max=680.6197509765625| mean=-751.3539428710938
k=yaw| v=<class 'float'>| shape=None| min=0.9912326975497529| max=0.9912326975497529| mean=0.9912326975497529
k=speed| v=<class 'numpy.float32'>| shape=None| min=10.675738| max=10.675738| mean=10.675738
k=all_other_agents_history_positions| v=<class 'numpy.ndarray'>| shape=(30, 4, 2)| min=-28.770815| max=0.0| mean=-0.12073036
k=all_other_agents_history_yaws| v=<class 'numpy.ndarray'>| shape=(30, 4, 1)| min=-0.0| max=0.025442362| mean=0.00021201969
k=all_other_agents_history_extents| v=<class 'numpy.ndarray'>| shape=(30, 4, 2)| min=0.0| max=4.3913283| mean=0.025854828
k=all_other_agents_history_availability| v=<class 'numpy.ndarray'>| shape=(30, 4)| min=False| max=True| mean=0.008333333333333333
k=all_other_agents_future_positions| v=<class 'numpy.ndarray'>| shape=(30, 12, 2)| min=-25.557451| max=0.035187725| mean=-0.3462321
k=all_other_agents_future_yaws| v=<class 'numpy.ndarray'>| shape=(30, 12, 1)| min=-0.0043439865| max=0.016661882| mean=0.0001295646
k=all_other_agents_future_extents| v=<class 'numpy.ndarray'>| shape=(30, 12, 2)| min=0.0| max=1.2727729| mean=0.03113103
k=all_other_agents_future_availability| v=<class 'numpy.ndarray'>| shape=(30, 12)| min=False| max=True| mean=0.03333333333333333
k=all_other_agents_types| v=<class 'numpy.ndarray'>| shape=(30,)| min=0| max=3| mean=0.1
k=agent_trajectory_polyline| v=<class 'numpy.ndarray'>| shape=(4, 3)| min=0.0| max=2.2737368e-13| mean=1.8947807e-14
k=agent_polyline_availability| v=<class 'numpy.ndarray'>| shape=(4,)| min=False| max=True| mean=0.25
k=other_agents_polyline| v=<class 'numpy.ndarray'>| shape=(30, 4, 3)| min=-28.770815| max=0.025442362| mean=-0.08041623
k=other_agents_polyline_availability| v=<class 'numpy.ndarray'>| shape=(30, 4)| min=False| max=True| mean=0.008333333333333333
k=lanes| v=<class 'numpy.ndarray'>| shape=(60, 20, 3)| min=-54.735256| max=88.40731| mean=3.3844721
k=lanes_availabilities| v=<class 'numpy.ndarray'>| shape=(60, 20)| min=False| max=True| mean=0.5758333333333333
k=lanes_mid| v=<class 'numpy.ndarray'>| shape=(30, 20, 3)| min=-54.734856| max=87.18307| mean=4.7187033
k=lanes_mid_availabilities| v=<class 'numpy.ndarray'>| shape=(30, 20)| min=False| max=True| mean=0.5733333333333334
k=crosswalks| v=<class 'numpy.ndarray'>| shape=(20, 20, 3)| min=-6.4767556| max=25.782284| mean=0.05331912
k=crosswalks_availabilities| v=<class 'numpy.ndarray'>| shape=(20, 20)| min=False| max=True| mean=0.01

3. Lane feature: MAX_LANES x MAX_VECTORS x 3 (XY + 1 TL-feature)
    x,
    y,
    tl_feature: tl_color_to_priority_idx = {"unknown": 0, "green": 1, "yellow": 2, "red": 3, "none": 4}
    lane_mid = interpolate average of lane_boundary_left and lane_boundary_right
 

## Typical bugs:
* Change _rescale_action() in l5env for ray rllib parallel training => all action -> [0,0] or [0,0,0] SAC and PPO train in wrong env, but why SAC still work??? -> not affect action in 
* Error:
ValueError: ('Unknown space type for serialization, ', <class 'gym.spaces.multi_binary.MultiBinary'>)
/root/.local/share/virtualenvs/l5kit-ZbMednhg/lib/python3.8/site-packages/ray/rllib/utils/serialization.py
-> go to gym file, update 

```python
@DeveloperAPI
def gym_space_to_dict(space: gym.spaces.Space) -> Dict:


    def _multi_binary(sp: gym.spaces.MultiBinary) -> Dict:
        return {
            "space": "multi-binary",
            "n": sp.n,
            "dtype": sp.dtype.str,
        }
    elif isinstance(space, gym.spaces.MultiBinary):
        return _multi_binary(space)

def gym_space_from_dict(d: Dict) -> gym.spaces.Space:


    def _multi_binary(d: Dict) -> gym.spaces.MultiBinary:
        return gym.spaces.MultiBinary(**__common(d))

    space_map = {
        "box": _box,
        "discrete": _discrete,
        "multi-discrete": _multi_discrete,
        "multi-binary": _multi_binary,
        "tuple": _tuple,
        "dict": _dict,
        "simplex": _simplex,
        "repeated": _repeated,
        "flex_dict": _flex_dict,
    }

```
* AttributeError: 'Box' object has no attribute 'low_repr'

* SAC is differ from PPO since it do not permute data obs
In L5envWrapper: (both SAC, PPO)
    obs has type: 
        + env.reset()['image'].reshape(self.raster_size, self.raster_size, self.n_channels)
        + onlyImageState = output.obs['image'].reshape(self.rasteimager_size, self.raster_size, self.n_channels)
    => 112,112,7 (and SAC use Model from Ray rllib, it just accept NxNxdim
    
In TorchGNCNN: (only PPO)
    input_dict['obs'] # 32, 112, 112, 7 (from l5envWrapper)
    obs_transformed = input_dict['obs'].permute(0, 3, 1, 2) # 32 x 7 x 112 x 112 [B, size, size, channels]
    => 32x 7 x 112 x 112 |=> Neural Net |=> action 
        => self._num_objects = obs_space.shape[2] # num_of_channels of input, channels x size x size => 112 (quite wrong) => PPO fail to work since it can only see (7x112) pixels
           

* AssertionError: torch.Size([32, 112, 112, 7]) != (_ ,112,112,7),  obs_transformed: torch.Size([32, 7, 112, 112])
but obs_shape = (112,112,7)

* LOW MEM:
-> reduce num_workers may work
* Reward drop intermediately    
-> Maybe due to PPO try to find action to maximize Reward in 
* 'Box' has no 'low_repr'
->pip3 install setuptools==65.5.0
-> downgrade gym==0.22 -> gym==0.21
* ValueError: loaded state dict contains a parameter group that doesn't match the size of optimizer's group
-> pip uninstall stable-baselines3
-> pip install stable-baselines3

* SAC, PPO crasg at beginning
=> reduce train batch size

* L5wrapper is wrong
-> Use reshape -> not change order of tensor but change shape
-> data['obs'].shape = (7,112,112)
-> data['obs'].reshape(112,112,7) => Very wrong
-> im = data['obs'].permute(2,0,1) => im.shape = (112, 112, 7)
-> Torch Conv need input as form: (Batch size,# channels, Width, Height)
* TypeError: conv2d() received an invalid combination of arguments - got (numpy.ndarray, Parameter, Parameter, tuple, tuple, tuple, int), but expected one of:
(Tensor input, Tensor weight, Tensor bias, tuple of ints stride, tuple of ints padding, tuple of ints dilation, int groups) didn't match because some of the arguments have invalid types: (numpy.ndarray, Parameter, Parameter, tuple, tuple, tuple, int)
-> convert input to Tensor before feed into Torch NN
* RuntimeError: imag is not implemented for tensors with non-complex dtypes.
-> convert tensor output of Torch NN to float
=> model(input).item() # tensor -> reshape to 1 -> detach to numpy -> 
Rayrllib:
`NOTE` SAC can even learn reshaped image (data['obs'].reshape(112,112,7)) with very high accuracy using model by ray rllib. Maybe rllib reshape it again, maybe SAC can extract good pattern from very wierd image :). PPO rllib also test with -30 reward (higher than SAC) but cannot drive successful since it. Even worse, PPO rllib now cannot achieve -100 (wrong action?, reshaped obs?) The reason: PPO directly act and improve learned policy and that maximize reward while SAC learn policy by maximize value and act by greedy policy (argmax Q).

SAC rllib beat PPO stable-baselines3, which mean that SAC although use reshaped obs, it can still learn.

* This cannot happen 
```python
    "post_fcnet_hiddens": [256],
    "vf_share_layers": True,  
```

* ambigous in image shape
At beginning, data['image'].shape = W,H,C
In code: l5kit/l5kit/l5kit/dataset/ego.py
```python
  def get_frame(self, scene_index: int, state_index: int, track_id: Optional[int] = None) -> dict:
        data = super().get_frame(scene_index, state_index, track_id=track_id)
        # TODO (@lberg): this should not be here but in the rasterizer
        data["image"] = data["image"].transpose(2, 0, 1)  # 0,1,C -> C,0,1
        return data
```
The dataset after that become: C, W, H
To visualize, It need to convert again to W, H, C
```python
    im = data["image"].transpose(1, 2, 0) # C, W, H -> W,H,C
    im = dataset.rasterizer.to_rgb(im)
```

Our L5EnvWrapperWithoutReshape: convert C, W, H from data_batch and convert to W, H, C
```python
    def step(self, action:  np.ndarray) -> GymStepOutput:
        ...
        onlyImageState = output.obs['image'].transpose(1,2,0)  # C,W,H -> W, H, C (for SAC/PPO torch model)
        ...

    def reset(self) -> Dict[str, np.ndarray]:
        return self.env.reset()['image'].transpose(1,2,0)
```

* TorchDistributionWrapper has no attribute 'dist'
-> replace with TorchDiagGaussian(TorchDistributionWrapper)
```python
class TorchDiagGaussian(TorchDistributionWrapper):
    """Wrapper class for PyTorch Normal distribution."""

    @override(ActionDistribution)
    def __init__(
        self,
        inputs: List[TensorType],
        model: TorchModelV2,
        *,
        action_space: Optional[gym.spaces.Space] = None
    ):
        super().__init__(inputs, model)
        mean, log_std = torch.chunk(self.inputs, 2, dim=1)
        self.log_std = log_std
        self.dist = torch.distributions.normal.Normal(mean, torch.exp(log_std))
        # Remember to squeeze action samples in case action space is Box(shape)
        self.zero_action_dim = action_space and action_space.shape == ()

```

* TorchDiagGaussian(TorchDistribution) has attribute 'dist'
-> add self.dist
```python
@DeveloperAPI
class TorchDistribution(Distribution, abc.ABC):
    """Wrapper class for torch.distributions."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._dist = self._get_torch_distribution(*args, **kwargs)
        self.dist = self._dist # TRI HUYNH

    @abc.abstractmethod
    def _get_torch_distribution(
        self, *args, **kwargs
    ) -> torch.distributions.Distribution:
        """Returns the torch.distributions.Distribution object to use."""

    @override(Distribution)
```
*   self.scene_index = self.np_random.randint(0, self.max_scene_id)
AttributeError: 'numpy.random._generator.Generator' object has no attribute 'randint'
-> change randint -> integers

* Do not use gpu before ray rllib (PPO.train). At customModel in def __init__() the model loaded is still on cpu, but after that it is all on gpu (def forward(), in CustomPPOTorchPolicy,...)


# Scenes 
SAC_RLlib: 
+ at scene 57 turn left although the GT turn right.
+ 82: go straight while GT turn left
+ 89: collide

pointnet , PointNet ++, apply permutation invariant operations (e.g. max pooling) on learned point em-beddings.
PointNet can solve:
- learn invariant of N! unorder points?
- local interaction
- global interation: rotation, translation,... invariance

max-pooling ->r as a symmetric function
