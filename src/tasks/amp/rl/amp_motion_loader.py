from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


class AmpMotionLoader:
  """Load G1 expert AMP states from CSV or NPZ motion files.

  The AMP observation layout is:
    [joint_pos - default_joint_pos, joint_vel]
  """

  def __init__(
    self,
    motion_files: str | Sequence[str],
    default_joint_pos: Sequence[float],
    device: str,
    time_between_frames: float,
    input_fps: float = 30.0,
    preload_transitions: bool = True,
    num_preload_transitions: int = 200000,
  ) -> None:
    self.device = device
    self.time_between_frames = time_between_frames
    self.input_fps = input_fps
    self.default_joint_pos = torch.tensor(
      default_joint_pos,
      dtype=torch.float32,
      device=device,
    )
    self.num_joints = self.default_joint_pos.numel()

    self.trajectories: list[torch.Tensor] = []
    self.trajectory_names: list[str] = []
    self.trajectory_idxs: list[int] = []
    self.trajectory_weights: list[float] = []
    self.trajectory_frame_durations: list[float] = []
    self.trajectory_lens: list[float] = []
    self.trajectory_num_frames: list[int] = []

    resolved_files = self._expand_motion_files(motion_files)
    if not resolved_files:
      raise FileNotFoundError(f"No AMP motion files found from {motion_files}.")

    for motion_idx, motion_file in enumerate(resolved_files):
      trajectory, frame_duration, weight = self._load_motion_file(motion_file)
      if trajectory.shape[-1] != self.observation_dim_from_joints:
        raise ValueError(
          f"AMP motion '{motion_file}' has dim {trajectory.shape[-1]}, "
          f"expected {self.observation_dim_from_joints}."
        )

      self.trajectories.append(trajectory)
      self.trajectory_names.append(str(motion_file))
      self.trajectory_idxs.append(motion_idx)
      self.trajectory_weights.append(weight)
      self.trajectory_frame_durations.append(frame_duration)
      self.trajectory_num_frames.append(trajectory.shape[0])
      self.trajectory_lens.append(max((trajectory.shape[0] - 1) * frame_duration, 0.0))
      print(
        f"Loaded AMP motion {motion_file}: "
        f"{trajectory.shape[0]} frames, dt={frame_duration:.4f}s."
      )

    weights = np.asarray(self.trajectory_weights, dtype=np.float64)
    self.trajectory_weights_np = weights / np.sum(weights)
    self.trajectory_frame_durations_np = np.asarray(
      self.trajectory_frame_durations,
      dtype=np.float64,
    )
    self.trajectory_lens_np = np.asarray(self.trajectory_lens, dtype=np.float64)

    self.preload_transitions = preload_transitions
    if preload_transitions:
      self._preload_transitions(num_preload_transitions)

  @property
  def observation_dim_from_joints(self) -> int:
    return self.num_joints * 2

  @property
  def observation_dim(self) -> int:
    return self.trajectories[0].shape[1]

  @property
  def num_motions(self) -> int:
    return len(self.trajectories)

  def feed_forward_generator(
    self,
    num_mini_batches: int,
    mini_batch_size: int,
  ):
    for _ in range(num_mini_batches):
      if self.preload_transitions:
        sample_idxs = torch.randint(
          low=0,
          high=self.preloaded_state.shape[0],
          size=(mini_batch_size,),
          device=self.device,
        )
        yield self.preloaded_state[sample_idxs], self.preloaded_next_state[sample_idxs]
      else:
        traj_idxs = self.weighted_traj_idx_sample_batch(mini_batch_size)
        times = self.traj_time_sample_batch(traj_idxs)
        states = self.get_frame_at_time_batch(traj_idxs, times)
        next_states = self.get_frame_at_time_batch(
          traj_idxs,
          times + self.time_between_frames,
        )
        yield states, next_states

  def weighted_traj_idx_sample_batch(self, size: int) -> np.ndarray:
    return np.random.choice(
      self.trajectory_idxs,
      size=size,
      p=self.trajectory_weights_np,
      replace=True,
    )

  def traj_time_sample_batch(self, traj_idxs: np.ndarray) -> np.ndarray:
    max_times = np.maximum(
      self.trajectory_lens_np[traj_idxs] - self.time_between_frames,
      0.0,
    )
    return np.random.uniform(size=len(traj_idxs)) * max_times

  def get_frame_at_time_batch(
    self,
    traj_idxs: np.ndarray,
    times: np.ndarray,
  ) -> torch.Tensor:
    frames = torch.zeros(
      len(traj_idxs),
      self.observation_dim,
      device=self.device,
    )
    for traj_idx in set(traj_idxs.tolist()):
      traj_mask = traj_idxs == traj_idx
      mask_idxs = np.nonzero(traj_mask)[0]
      trajectory = self.trajectories[traj_idx]
      num_frames = trajectory.shape[0]
      frame_duration = self.trajectory_frame_durations_np[traj_idx]

      local_times = np.clip(
        times[traj_mask],
        0.0,
        self.trajectory_lens_np[traj_idx],
      )
      frame_pos = local_times / frame_duration if frame_duration > 0.0 else 0.0
      idx_low = np.floor(frame_pos).astype(np.int64)
      idx_low = np.clip(idx_low, 0, num_frames - 1)
      idx_high = np.clip(idx_low + 1, 0, num_frames - 1)
      idx_low_t = torch.tensor(idx_low, dtype=torch.long, device=self.device)
      idx_high_t = torch.tensor(idx_high, dtype=torch.long, device=self.device)
      mask_idxs_t = torch.tensor(mask_idxs, dtype=torch.long, device=self.device)
      blend = torch.tensor(
        frame_pos - idx_low,
        dtype=torch.float32,
        device=self.device,
      ).unsqueeze(-1)

      frame_start = trajectory[idx_low_t]
      frame_end = trajectory[idx_high_t]
      frames[mask_idxs_t] = frame_start * (1.0 - blend) + frame_end * blend
    return frames

  def _preload_transitions(self, num_preload_transitions: int) -> None:
    print(f"Preloading {num_preload_transitions} AMP transitions.")
    traj_idxs = self.weighted_traj_idx_sample_batch(num_preload_transitions)
    times = self.traj_time_sample_batch(traj_idxs)
    self.preloaded_state = self.get_frame_at_time_batch(traj_idxs, times)
    self.preloaded_next_state = self.get_frame_at_time_batch(
      traj_idxs,
      times + self.time_between_frames,
    )

  def _load_motion_file(self, motion_file: Path) -> tuple[torch.Tensor, float, float]:
    suffix = motion_file.suffix.lower()
    if suffix == ".npz":
      joint_pos, joint_vel, frame_duration, weight = self._load_npz_motion(motion_file)
    elif suffix == ".csv":
      joint_pos, joint_vel, frame_duration, weight = self._load_csv_motion(motion_file)
    else:
      raise ValueError(f"Unsupported AMP motion extension: {motion_file}")

    if joint_pos.shape != joint_vel.shape:
      raise ValueError(
        f"AMP motion '{motion_file}' joint_pos and joint_vel shapes differ: "
        f"{joint_pos.shape} vs {joint_vel.shape}."
      )
    if joint_pos.shape[-1] != self.num_joints:
      raise ValueError(
        f"AMP motion '{motion_file}' has {joint_pos.shape[-1]} joints, "
        f"expected {self.num_joints}."
      )

    joint_pos_t = torch.tensor(joint_pos, dtype=torch.float32, device=self.device)
    joint_vel_t = torch.tensor(joint_vel, dtype=torch.float32, device=self.device)
    joint_pos_rel = joint_pos_t - self.default_joint_pos
    return torch.cat((joint_pos_rel, joint_vel_t), dim=-1), frame_duration, weight

  def _load_npz_motion(
    self,
    motion_file: Path,
  ) -> tuple[np.ndarray, np.ndarray, float, float]:
    data = np.load(motion_file)
    if "amp_obs" in data:
      amp_obs = np.asarray(data["amp_obs"], dtype=np.float32)
      if amp_obs.shape[-1] != self.observation_dim_from_joints:
        raise ValueError(
          f"NPZ amp_obs dim is {amp_obs.shape[-1]}, "
          f"expected {self.observation_dim_from_joints}."
        )
      joint_pos = amp_obs[:, : self.num_joints] + self.default_joint_pos.cpu().numpy()
      joint_vel = amp_obs[:, self.num_joints :]
    elif "joint_pos" in data:
      joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
      if "joint_vel" in data:
        joint_vel = np.asarray(data["joint_vel"], dtype=np.float32)
      else:
        joint_vel = self._finite_difference(joint_pos, self.input_fps)
    else:
      raise ValueError(
        f"NPZ AMP motion '{motion_file}' must contain 'joint_pos' or 'amp_obs'."
      )

    fps = self.input_fps
    if "fps" in data:
      fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    weight = float(np.asarray(data["weight"]).reshape(-1)[0]) if "weight" in data else 1.0
    return joint_pos, joint_vel, 1.0 / fps, weight

  def _load_csv_motion(
    self,
    motion_file: Path,
  ) -> tuple[np.ndarray, np.ndarray, float, float]:
    motion = np.loadtxt(motion_file, delimiter=",", dtype=np.float32)
    if motion.ndim == 1:
      motion = motion[None, :]

    if motion.shape[1] >= 7 + self.num_joints:
      joint_pos = motion[:, 7 : 7 + self.num_joints]
    elif motion.shape[1] == self.num_joints:
      joint_pos = motion
    else:
      raise ValueError(
        f"CSV AMP motion '{motion_file}' has {motion.shape[1]} columns. "
        f"Expected {self.num_joints} or at least {7 + self.num_joints}."
      )

    joint_vel = self._finite_difference(joint_pos, self.input_fps)
    return joint_pos, joint_vel, 1.0 / self.input_fps, 1.0

  @staticmethod
  def _finite_difference(joint_pos: np.ndarray, fps: float) -> np.ndarray:
    if joint_pos.shape[0] <= 1:
      return np.zeros_like(joint_pos, dtype=np.float32)
    edge_order = 2 if joint_pos.shape[0] > 2 else 1
    return np.gradient(joint_pos, 1.0 / fps, axis=0, edge_order=edge_order).astype(
      np.float32
    )

  @staticmethod
  def _expand_motion_files(motion_files: str | Sequence[str]) -> list[Path]:
    if isinstance(motion_files, str):
      motion_files = (motion_files,)

    resolved: list[Path] = []
    for pattern in motion_files:
      expanded = os.path.expanduser(os.path.expandvars(str(pattern)))
      matches = sorted(glob.glob(expanded))
      resolved.extend(Path(match) for match in matches)
    return resolved
