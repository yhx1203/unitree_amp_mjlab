"""Lightweight MuJoCo sim2sim runner for the G1 AMP walking policy.

This script intentionally bypasses MJLab env/managers. It loads the raw MuJoCo
scene, reconstructs the AMP actor observation, runs the RSL-RL actor checkpoint,
and applies a PD torque controller to the MuJoCo motors.
"""

from __future__ import annotations

import argparse
import os
import struct
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch
from rsl_rl.models.mlp_model import MLPModel


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XML = REPO_ROOT / "src/assets/robots/unitree_g1/xmls/scene_g1.xml"

JOINT_NAMES = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)

DEFAULT_JOINT_POS = np.array(
  [
    -0.1,
    0.0,
    0.0,
    0.3,
    -0.2,
    0.0,
    -0.1,
    0.0,
    0.0,
    0.3,
    -0.2,
    0.0,
    0.0,
    0.0,
    0.0,
    0.35,
    0.18,
    0.0,
    0.87,
    0.0,
    0.0,
    0.0,
    0.35,
    -0.18,
    0.0,
    0.87,
    0.0,
    0.0,
    0.0,
  ],
  dtype=np.float32,
)

ACTION_SCALE = np.array(
  [
    0.55,
    0.35,
    0.55,
    0.35,
    0.44,
    0.44,
    0.55,
    0.35,
    0.55,
    0.35,
    0.44,
    0.44,
    0.55,
    0.44,
    0.44,
    0.44,
    0.44,
    0.44,
    0.44,
    0.44,
    0.07,
    0.07,
    0.44,
    0.44,
    0.44,
    0.44,
    0.44,
    0.07,
    0.07,
  ],
  dtype=np.float32,
)

KP = np.array(
  [
    40.2,
    99.1,
    40.2,
    99.1,
    28.5,
    28.5,
    40.2,
    99.1,
    40.2,
    99.1,
    28.5,
    28.5,
    40.2,
    28.5,
    28.5,
    14.3,
    14.3,
    14.3,
    14.3,
    14.3,
    16.8,
    16.8,
    14.3,
    14.3,
    14.3,
    14.3,
    14.3,
    16.8,
    16.8,
  ],
  dtype=np.float32,
)

KD = np.array(
  [
    2.6,
    6.3,
    2.6,
    6.3,
    1.8,
    1.8,
    2.6,
    6.3,
    2.6,
    6.3,
    1.8,
    1.8,
    2.6,
    1.8,
    1.8,
    0.9,
    0.9,
    0.9,
    0.9,
    0.9,
    1.1,
    1.1,
    0.9,
    0.9,
    0.9,
    0.9,
    0.9,
    1.1,
    1.1,
  ],
  dtype=np.float32,
)


class Sim2SimCommand:
  def __init__(
    self,
    lin_vel_x: float,
    lin_vel_y: float,
    ang_vel_z: float,
    lin_step: float,
    yaw_step: float,
    lin_vel_x_range: tuple[float, float],
    lin_vel_y_range: tuple[float, float],
    ang_vel_z_range: tuple[float, float],
  ) -> None:
    self.value = np.array([lin_vel_x, lin_vel_y, ang_vel_z], dtype=np.float32)
    self.lin_step = lin_step
    self.yaw_step = yaw_step
    self.ranges = np.array(
      (lin_vel_x_range, lin_vel_y_range, ang_vel_z_range), dtype=np.float32
    )
    self.clamp()

  def handle_key(self, key: int) -> None:
    # GLFW key codes used by MuJoCo's native viewer.
    key_left, key_right, key_down, key_up = 263, 262, 264, 265
    key_kp_2, key_kp_4, key_kp_6, key_kp_8 = 322, 324, 326, 328
    if key in (key_up, key_kp_8):
      self.value[0] += self.lin_step
    elif key in (key_down, key_kp_2):
      self.value[0] -= self.lin_step
    elif key in (key_left, key_kp_4):
      self.value[2] += self.yaw_step
    elif key in (key_right, key_kp_6):
      self.value[2] -= self.yaw_step
    elif key in (ord("0"), ord(" ")):
      self.value[:] = 0.0
    else:
      return
    self.clamp()
    self.print()

  def set_from_unit_inputs(
    self,
    lin_vel_x: float,
    lin_vel_y: float,
    ang_vel_z: float,
  ) -> None:
    self.value[0] = self._scale_unit_input(lin_vel_x, self.ranges[0])
    self.value[1] = self._scale_unit_input(lin_vel_y, self.ranges[1])
    self.value[2] = self._scale_unit_input(ang_vel_z, self.ranges[2])
    self.clamp()

  def clamp(self) -> None:
    self.value[:] = np.clip(self.value, self.ranges[:, 0], self.ranges[:, 1])

  def print(self) -> None:
    print(
      "[CMD] "
      f"lin_vel_x={self.value[0]:+.2f} m/s, "
      f"lin_vel_y={self.value[1]:+.2f} m/s, "
      f"ang_vel_z={self.value[2]:+.2f} rad/s"
    )

  @staticmethod
  def _scale_unit_input(value: float, command_range: np.ndarray) -> float:
    value = float(np.clip(value, -1.0, 1.0))
    if value >= 0.0:
      return value * float(command_range[1])
    return -abs(value) * abs(float(command_range[0]))


class LinuxJoystick:
  """Small non-blocking reader for Linux /dev/input/js* devices."""

  EVENT_FORMAT = "IhBB"
  EVENT_SIZE = struct.calcsize(EVENT_FORMAT)
  JS_EVENT_BUTTON = 0x01
  JS_EVENT_AXIS = 0x02
  JS_EVENT_INIT = 0x80

  def __init__(
    self,
    device: str,
    deadzone: float,
    axis_lx: int,
    axis_ly: int,
    axis_rx: int,
  ) -> None:
    self.device = device
    self.deadzone = deadzone
    self.axis_lx = axis_lx
    self.axis_ly = axis_ly
    self.axis_rx = axis_rx
    self.axes: dict[int, float] = {}
    self._file = open(device, "rb", buffering=0)
    os.set_blocking(self._file.fileno(), False)

  def close(self) -> None:
    self._file.close()

  def poll(self, command: Sim2SimCommand) -> None:
    saw_axis_event = False
    while True:
      try:
        data = self._file.read(self.EVENT_SIZE)
      except BlockingIOError:
        break
      if not data:
        break
      if len(data) != self.EVENT_SIZE:
        break

      _, value, event_type, number = struct.unpack(self.EVENT_FORMAT, data)
      event_type &= ~self.JS_EVENT_INIT
      if event_type != self.JS_EVENT_AXIS:
        continue

      self.axes[number] = self._normalize_axis(value)
      saw_axis_event = True

    if saw_axis_event:
      lx = self.axes.get(self.axis_lx, 0.0)
      ly = self.axes.get(self.axis_ly, 0.0)
      rx = self.axes.get(self.axis_rx, 0.0)
      command.set_from_unit_inputs(
        lin_vel_x=-ly,
        lin_vel_y=-lx,
        ang_vel_z=-rx,
      )

  def _normalize_axis(self, value: int) -> float:
    normalized = float(value) / 32767.0
    normalized = float(np.clip(normalized, -1.0, 1.0))
    if abs(normalized) < self.deadzone:
      return 0.0
    return normalized


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Run a G1 AMP checkpoint directly in raw MuJoCo."
  )
  parser.add_argument(
    "--checkpoint-file",
    required=True,
    help="Path to model_*.pt produced by the AMP training run.",
  )
  parser.add_argument(
    "--xml",
    default=str(DEFAULT_XML),
    help="MuJoCo XML scene path.",
  )
  parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
  parser.add_argument("--cmd-x", type=float, default=0.3)
  parser.add_argument("--cmd-y", type=float, default=0.0)
  parser.add_argument("--cmd-yaw", type=float, default=0.0)
  parser.add_argument("--cmd-step", type=float, default=0.05)
  parser.add_argument("--yaw-step", type=float, default=0.05)
  parser.add_argument("--cmd-x-range", type=float, nargs=2, default=(-0.1, 0.8))
  parser.add_argument("--cmd-y-range", type=float, nargs=2, default=(-0.1, 0.1))
  parser.add_argument("--cmd-yaw-range", type=float, nargs=2, default=(-0.1, 0.1))
  parser.add_argument("--gamepad", action="store_true", help="Read velocity commands from a Linux joystick.")
  parser.add_argument("--joystick-device", default="/dev/input/js0")
  parser.add_argument("--joystick-deadzone", type=float, default=0.08)
  parser.add_argument("--joystick-axis-lx", type=int, default=0)
  parser.add_argument("--joystick-axis-ly", type=int, default=1)
  parser.add_argument("--joystick-axis-rx", type=int, default=3)
  parser.add_argument("--sim-dt", type=float, default=0.005)
  parser.add_argument("--control-dt", type=float, default=0.02)
  parser.add_argument("--base-height", type=float, default=0.793)
  parser.add_argument("--duration", type=float, default=None, help="Seconds to run; omit for infinite.")
  parser.add_argument("--no-viewer", action="store_true", help="Run headless.")
  parser.add_argument(
    "--status-interval",
    type=float,
    default=1.0,
    help="Seconds between status prints; set <=0 to disable.",
  )
  return parser.parse_args()


def load_actor(checkpoint_file: str, device: str) -> MLPModel:
  obs_dim = 96
  action_dim = len(JOINT_NAMES)
  actor = MLPModel(
    obs={"actor": torch.zeros(1, obs_dim)},
    obs_groups={"actor": ["actor"]},
    obs_set="actor",
    output_dim=action_dim,
    hidden_dims=(512, 256, 128),
    activation="elu",
    obs_normalization=True,
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    },
  )
  checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
  actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
  actor.to(device)
  actor.eval()
  return actor


def get_sensor_slice(model: mujoco.MjModel, name: str) -> slice | None:
  sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
  if sensor_id < 0:
    return None
  start = int(model.sensor_adr[sensor_id])
  dim = int(model.sensor_dim[sensor_id])
  return slice(start, start + dim)


def get_joint_addresses(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
  qpos_ids = []
  qvel_ids = []
  for name in JOINT_NAMES:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
      raise ValueError(f"Joint not found in XML: {name}")
    qpos_ids.append(model.jnt_qposadr[joint_id])
    qvel_ids.append(model.jnt_dofadr[joint_id])
  return np.asarray(qpos_ids, dtype=np.int32), np.asarray(qvel_ids, dtype=np.int32)


def quat_to_matrix_wxyz(quat: np.ndarray) -> np.ndarray:
  w, x, y, z = quat
  return np.array(
    [
      [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ],
    dtype=np.float32,
  )


def build_observation(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  command: np.ndarray,
  qpos_ids: np.ndarray,
  qvel_ids: np.ndarray,
  last_action: np.ndarray,
  imu_gyro_slice: slice | None,
) -> np.ndarray:
  if imu_gyro_slice is not None:
    base_ang_vel = np.asarray(data.sensordata[imu_gyro_slice], dtype=np.float32)
  else:
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    vel = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, pelvis_id, vel, 1)
    base_ang_vel = vel[:3].astype(np.float32)

  root_quat = np.asarray(data.qpos[3:7], dtype=np.float32)
  root_quat = root_quat / np.linalg.norm(root_quat)
  rot_body_to_world = quat_to_matrix_wxyz(root_quat)
  gravity_world = np.asarray(model.opt.gravity, dtype=np.float32)
  gravity_world = gravity_world / np.linalg.norm(gravity_world)
  projected_gravity = rot_body_to_world.T @ gravity_world

  joint_pos_rel = np.asarray(data.qpos[qpos_ids], dtype=np.float32) - DEFAULT_JOINT_POS
  joint_vel_rel = np.asarray(data.qvel[qvel_ids], dtype=np.float32)
  return np.concatenate(
    (
      base_ang_vel,
      projected_gravity.astype(np.float32),
      command.astype(np.float32),
      joint_pos_rel,
      joint_vel_rel,
      last_action.astype(np.float32),
    )
  )


def apply_pd_control(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  qpos_ids: np.ndarray,
  qvel_ids: np.ndarray,
  action: np.ndarray,
) -> None:
  joint_pos = np.asarray(data.qpos[qpos_ids], dtype=np.float32)
  joint_vel = np.asarray(data.qvel[qvel_ids], dtype=np.float32)
  target_pos = DEFAULT_JOINT_POS + action * ACTION_SCALE
  torque = KP * (target_pos - joint_pos) - KD * joint_vel
  torque = np.clip(torque, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
  data.ctrl[:] = torque


def reset_robot(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  qpos_ids: np.ndarray,
  base_height: float,
) -> None:
  data.qpos[:] = 0.0
  data.qvel[:] = 0.0
  data.ctrl[:] = 0.0
  data.qpos[0:3] = np.array([0.0, 0.0, base_height])
  data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
  data.qpos[qpos_ids] = DEFAULT_JOINT_POS
  mujoco.mj_forward(model, data)


def get_local_root_velocity(
  model: mujoco.MjModel,
  data: mujoco.MjData,
) -> tuple[np.ndarray, np.ndarray]:
  pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
  vel = np.zeros(6, dtype=np.float64)
  mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, pelvis_id, vel, 1)
  return vel[3:].astype(np.float32), vel[:3].astype(np.float32)


def main() -> None:
  args = parse_args()
  xml_path = Path(args.xml)
  checkpoint_path = Path(args.checkpoint_file)
  if not xml_path.exists():
    raise FileNotFoundError(xml_path)
  if not checkpoint_path.exists():
    raise FileNotFoundError(checkpoint_path)

  model = mujoco.MjModel.from_xml_path(str(xml_path))
  model.opt.timestep = args.sim_dt
  data = mujoco.MjData(model)
  qpos_ids, qvel_ids = get_joint_addresses(model)
  imu_gyro_slice = get_sensor_slice(model, "imu_gyro")

  actor = load_actor(str(checkpoint_path), args.device)
  command = Sim2SimCommand(
    lin_vel_x=args.cmd_x,
    lin_vel_y=args.cmd_y,
    ang_vel_z=args.cmd_yaw,
    lin_step=args.cmd_step,
    yaw_step=args.yaw_step,
    lin_vel_x_range=tuple(args.cmd_x_range),
    lin_vel_y_range=tuple(args.cmd_y_range),
    ang_vel_z_range=tuple(args.cmd_yaw_range),
  )
  joystick = None
  if args.gamepad:
    try:
      joystick = LinuxJoystick(
        device=args.joystick_device,
        deadzone=args.joystick_deadzone,
        axis_lx=args.joystick_axis_lx,
        axis_ly=args.joystick_axis_ly,
        axis_rx=args.joystick_axis_rx,
      )
    except OSError as exc:
      raise OSError(
        f"Could not open joystick device '{args.joystick_device}'. "
        "Check that the gamepad is connected and that you have permission to "
        "read the device."
      ) from exc
    command.value[:] = 0.0
    command.clamp()
  decimation = max(1, round(args.control_dt / args.sim_dt))
  last_action = np.zeros(len(JOINT_NAMES), dtype=np.float32)
  reset_robot(model, data, qpos_ids, args.base_height)

  print(f"[INFO] XML: {xml_path}")
  print(f"[INFO] Checkpoint: {checkpoint_path}")
  print(
    f"[INFO] sim_dt={args.sim_dt:.4f}, control_dt={args.control_dt:.4f}, "
    f"decimation={decimation}"
  )
  print(
    "[INFO] Keys: Up/Down adjust lin_vel_x, Left/Right adjust ang_vel_z, "
    "Space or 0 zeros the command."
  )
  if joystick is not None:
    print(
      "[INFO] Gamepad enabled: "
      f"{args.joystick_device} "
      f"(lx axis {args.joystick_axis_lx}, "
      f"ly axis {args.joystick_axis_ly}, "
      f"rx axis {args.joystick_axis_rx}, "
      f"deadzone {args.joystick_deadzone:.2f})."
    )
    print(
      "[INFO] Gamepad mapping: left stick Y -> lin_vel_x, "
      "left stick X -> lin_vel_y, right stick X -> ang_vel_z."
    )
  command.print()

  viewer = None
  if not args.no_viewer:
    viewer = mujoco.viewer.launch_passive(
      model,
      data,
      key_callback=command.handle_key,
    )

  start_wall = time.time()
  next_status_time = 0.0
  step_count = 0
  try:
    while viewer is None or viewer.is_running():
      sim_time = data.time
      if args.duration is not None and sim_time >= args.duration:
        break

      if step_count % decimation == 0:
        if joystick is not None:
          joystick.poll(command)
        obs_np = build_observation(
          model,
          data,
          command.value,
          qpos_ids,
          qvel_ids,
          last_action,
          imu_gyro_slice,
        )
        obs = torch.from_numpy(obs_np).unsqueeze(0).to(args.device)
        with torch.inference_mode():
          action_tensor = actor({"actor": obs})
        last_action = action_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)

      apply_pd_control(model, data, qpos_ids, qvel_ids, last_action)
      mujoco.mj_step(model, data)
      step_count += 1

      if viewer is not None:
        viewer.sync()
        elapsed = time.time() - start_wall
        target_elapsed = data.time
        if target_elapsed > elapsed:
          time.sleep(target_elapsed - elapsed)

      if args.status_interval > 0.0 and data.time >= next_status_time:
        local_lin_vel, local_ang_vel = get_local_root_velocity(model, data)
        print(
          "[STATE] "
          f"t={data.time:6.2f}s, "
          f"cmd=({command.value[0]:+.2f}, {command.value[1]:+.2f}, {command.value[2]:+.2f}), "
          f"height={data.qpos[2]:.3f}, "
          f"vel_b=({local_lin_vel[0]:+.2f}, {local_lin_vel[1]:+.2f}), "
          f"yaw_rate_b={local_ang_vel[2]:+.2f}"
        )
        next_status_time = data.time + args.status_interval
  finally:
    if viewer is not None:
      viewer.close()
    if joystick is not None:
      joystick.close()


if __name__ == "__main__":
  main()
