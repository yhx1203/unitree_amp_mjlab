# G1 AMP sim2sim（Unitree SDK2）

这套部署端不直接创建 MuJoCo 模型，而是通过 Unitree SDK2 DDS 与
`unitree_mujoco` 通信：

- 订阅 `rt/lowstate`，读取 G1 29DOF 关节、骨盆 IMU；
- 重建训练时的 96 维 actor observation；
- 加载 `model_*.pt` 中的确定性 actor；
- 向 `rt/lowcmd` 发布 PR 空间（`mode_pr=0`）的 29DOF PD 目标。

## 1. 启动 unitree_mujoco

推荐使用 `unitree_mujoco` 的 C++ 仿真器。确保 DDS domain 为 `1`、网卡为
`lo`，并加载 G1 29DOF 场景：

```bash
cd /home/s135/codespace/unitree_mujoco/simulate/build
./unitree_mujoco -r g1 -s scene_29dof.xml -i 1 -n lo
```

如果使用 Python 仿真器，请先在
`/home/s135/codespace/unitree_mujoco/simulate_python/config.py` 中设置：

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

以上是完整配置，不能只保留前五项；即使关闭 joystick，仿真入口仍会读取其余
字段。

然后启动：

```bash
cd unitree_mujoco/simulate_python
conda activate mjlab
python unitree_mujoco.py
```

## 2. 验证并运行策略

先在本项目目录验证 checkpoint、观测维度和同目录 `policy.onnx` 元数据：

```bash
python deploy/amp/sim2sim/g1_amp_sim2sim.py \
  --checkpoint-file logs/rsl_rl/g1_amp_walking/2026-07-16_12-18-46_amp_omni_walk_3500/model_1000.pt \
  --validate-only
```

启动固定速度指令的 sim2sim：

```bash
conda activate mjlab1
python deploy/amp/sim2sim/g1_amp_sim2sim.py \
  --checkpoint-file logs/rsl_rl/g1_amp_walking/2026-07-16_12-18-46_amp_omni_walk_3500/model_1200.pt \
  --domain-id 1 \
  --interface lo \
  --cmd-x 0.0 \
  --cmd-y 0.6 \
  --cmd-yaw 0.0


python deploy/amp/sim2sim/g1_amp_sim2sim.py \
  --checkpoint-file logs/rsl_rl/g1_amp_walking/amp_walk_test/model_1000.pt \
  --domain-id 1 \
  --interface lo \
  --cmd-x 0.5 \
  --cmd-y 0.0 \
  --cmd-yaw 0.0
```

控制器先用 2 秒从仿真当前姿态平滑过渡到训练默认姿态，再保持 0.5 秒，
随后以 50 Hz 运行策略、以 200 Hz 发布底层命令。按 `Ctrl+C` 后会释放位置
刚度，避免仿真继续执行最后一帧策略目标。

## 3. 可选手柄控制

将 `unitree_mujoco` 的 joystick 功能打开后，部署端添加 `--wireless`：

```bash
python deploy/amp/sim2sim/g1_amp_sim2sim.py \
  --checkpoint-file logs/rsl_rl/g1_amp_walking/2026-07-16_12-18-46_amp_omni_walk_3500/model_1000.pt \
  --wireless
```

映射为：左摇杆 Y 前后、左摇杆 X 横移、右摇杆 X 转向。默认指令范围与当前
AMP 训练配置一致：`vx=[-0.4, 0.8]`、`vy=[-0.3, 0.3]`、
`wz=[-0.5, 0.5]`。

> 当前入口明确用于 `unitree_mujoco` sim2sim。上实物前还需要加入运控服务释放、
> 急停、姿态/关节限位、通信失联处理和实物启动状态机，不要直接把此入口用于真机。
