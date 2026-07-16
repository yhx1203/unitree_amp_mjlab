# unitree_amp_mjlab

## Overview

`unitree_amp_mjlab` is a reinforcement learning codebase built on MJLab for AMP-based Unitree G1 humanoid locomotion.

## Installation
**Conda environment**

```bash
conda create -n mjlab python=3.11
conda activate mjlab
```

**Install dependencies**
```bash
sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev
```

**Install unitree_amp_mjlab**
```bash
git clone https://github.com/yhx1203/unitree_amp_mjlab
```

```bash
cd unitree_amp_mjlab
pip install -e .
```


## Training
```bash
# You can visualize the motion before training
python scripts/view_csv_in_mujoco.py src/assets/motions/g1/walk1_subject1_0_1400.csv \
  --once \
  --speed 3
```
![replay](docs/replay.gif)

```bash
python scripts/train.py Unitree-G1-AMP-Flat \
  --env.scene.num-envs 4096 \
  --agent.run-name amp_walk_test \
  --agent.upload-model False
```

## Evaluate
**native**
```bash
python scripts/play.py Unitree-G1-AMP-Flat \
  --checkpoint-file logs/rsl_rl/g1_amp_walking/amp_walk_test/model_1000.pt \
  --num-envs 1 \
  --viewer native
```
![native](docs/native.gif)

**viser**
```bash
python scripts/play.py Unitree-G1-AMP-Flat \
  --checkpoint-file logs/rsl_rl/g1_amp_walking/amp_walk_test/model_1000.pt \
  --num-envs 1 \
  --viewer viser
```
![viser](docs/viser.gif)

## Sim2sim(Unitree SDK2 + unitree_mujoco)

Installing the [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco)

configure the following settings in

~/unitree_mujoco/simulate_python/config.py

```python
ROBOT = "g1"
ROBOT_SCENE = "../unitree_robots/g1/scene_29dof.xml"
DOMAIN_ID = 1
INTERFACE = "lo"
USE_JOYSTICK = 0
JOYSTICK_TYPE = "xbox"
JOYSTICK_DEVICE = 0

PRINT_SCENE_INFORMATION = True
ENABLE_ELASTIC_BAND = True

SIMULATE_DT = 0.005
VIEWER_DT = 0.02
```
terminal 1
```bash
cd unitree_mujoco/simulate_python
conda activate mjlab
python unitree_mujoco.py
```

terminal 2
```bash
conda activate mjlab
python deploy/amp/sim2sim/g1_amp_sim2sim.py \
  --checkpoint-file logs/rsl_rl/g1_amp_walking/amp_walk_test/model_1000.pt \
  --domain-id 1 \
  --interface lo \
  --cmd-x 0.5 \
  --cmd-y 0.0 \
  --cmd-yaw 0.0
```

![mujoco](docs/mujoco.gif)



## Acknowledgements

This project builds upon and benefits from the following open-source repositories:

- [mjlab](https://github.com/mujocolab/mjlab)
- [unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab)
- [TienKung-Lab](https://github.com/Open-X-Humanoid/TienKung-Lab)
- [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco)
