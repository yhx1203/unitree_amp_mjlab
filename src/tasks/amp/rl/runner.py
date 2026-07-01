from __future__ import annotations

import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from rsl_rl.env import VecEnv
from rsl_rl.utils import check_nan

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.rl.runner import MjlabOnPolicyRunner

from .amp_motion_loader import AmpMotionLoader
from .discriminator import AmpDiscriminator, AmpRunningNormalizer
from .replay_buffer import AmpReplayBuffer


class AmpOnPolicyRunner(MjlabOnPolicyRunner):
  """MJLab/RSL-RL v5 runner with an AMP discriminator reward."""

  env: RslRlVecEnvWrapper

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    train_cfg.setdefault("algorithm", {}).setdefault("rnd_cfg", None)
    super().__init__(env, train_cfg, log_dir, device)

    obs = self.env.get_observations().to(self.device)
    if "amp" not in obs:
      raise ValueError(
        "AMP runner requires an 'amp' observation group in the environment config."
      )
    self.amp_obs_dim = obs["amp"].shape[-1]

    amp_default_joint_pos = self.cfg.get("amp_default_joint_pos")
    if amp_default_joint_pos is None:
      raise ValueError("AMP runner requires 'amp_default_joint_pos' in rl_cfg.")

    self.amp_motion_loader = AmpMotionLoader(
      motion_files=self.cfg["amp_motion_files"],
      default_joint_pos=amp_default_joint_pos,
      device=self.device,
      time_between_frames=self._env_step_dt(),
      input_fps=self.cfg.get("amp_motion_input_fps", 30.0),
      preload_transitions=True,
      num_preload_transitions=self.cfg.get("amp_num_preload_transitions", 200000),
    )
    if self.amp_motion_loader.observation_dim != self.amp_obs_dim:
      raise ValueError(
        f"AMP observation dim mismatch: env={self.amp_obs_dim}, "
        f"motion={self.amp_motion_loader.observation_dim}."
      )

    self.amp_discriminator = AmpDiscriminator(
      obs_dim=self.amp_obs_dim,
      reward_coef=self.cfg.get("amp_reward_coef", 2.0),
      hidden_dims=tuple(self.cfg.get("amp_discr_hidden_dims", (1024, 512))),
      task_reward_lerp=self.cfg.get("amp_task_reward_lerp", 0.3),
      reward_scale=(
        self._env_step_dt()
        if self.cfg.get("amp_reward_scale_by_dt", True)
        else 1.0
      ),
    ).to(self.device)
    self.amp_normalizer = (
      AmpRunningNormalizer(
        self.amp_obs_dim,
        device=self.device,
        clip=self.cfg.get("amp_normalizer_clip", 5.0),
      )
      if self.cfg.get("amp_normalize", True)
      else None
    )
    self.amp_replay_buffer = AmpReplayBuffer(
      obs_dim=self.amp_obs_dim,
      buffer_size=self.cfg.get("amp_replay_buffer_size", 100000),
      device=self.device,
    )

    weight_decay = self.cfg.get("amp_discr_weight_decay", 1.0e-4)
    head_weight_decay = self.cfg.get("amp_discr_head_weight_decay", weight_decay)
    self.amp_optimizer = optim.Adam(
      [
        {"params": self.amp_discriminator.trunk.parameters(), "weight_decay": weight_decay},
        {
          "params": self.amp_discriminator.amp_linear.parameters(),
          "weight_decay": head_weight_decay,
        },
      ],
      lr=self.cfg.get("amp_discr_learning_rate", 1.0e-4),
    )

    default_amp_batch_size = (
      self.env.num_envs
      * self.cfg["num_steps_per_env"]
      // self.cfg["algorithm"]["num_mini_batches"]
    )
    self.amp_batch_size = self.cfg.get("amp_batch_size") or default_amp_batch_size
    self.amp_grad_penalty_coef = self.cfg.get("amp_grad_penalty_coef", 10.0)
    self.amp_max_grad_norm = self.cfg.get("amp_max_grad_norm", 1.0)

  def learn(
    self,
    num_learning_iterations: int,
    init_at_random_ep_len: bool = False,
  ) -> None:
    """Run PPO rollouts while replacing task rewards with AMP-mixed rewards."""
    if init_at_random_ep_len:
      self.env.episode_length_buf = torch.randint_like(
        self.env.episode_length_buf,
        high=int(self.env.max_episode_length),
      )

    obs = self.env.get_observations().to(self.device)
    amp_obs = obs["amp"].clone()
    self.alg.train_mode()
    self.amp_discriminator.train()

    if self.is_distributed:
      print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
      self.alg.broadcast_parameters()

    self.logger.init_logging_writer()

    start_it = self.current_learning_iteration
    total_it = start_it + num_learning_iterations
    for it in range(start_it, total_it):
      start = time.time()
      with torch.inference_mode():
        for _ in range(self.cfg["num_steps_per_env"]):
          actions = self.alg.act(obs)
          next_obs, task_rewards, dones, extras = self.env.step(
            actions.to(self.env.device)
          )
          if self.cfg.get("check_for_nan", True):
            check_nan(next_obs, task_rewards, dones)
          next_obs, task_rewards, dones = (
            next_obs.to(self.device),
            task_rewards.to(self.device),
            dones.to(self.device),
          )

          next_amp_obs = next_obs["amp"]
          done_mask = dones.to(dtype=torch.bool).view(-1, 1)
          next_amp_obs_for_amp = torch.where(done_mask, amp_obs, next_amp_obs)
          amp_rewards, _ = self.amp_discriminator.predict_amp_reward(
            amp_obs,
            next_amp_obs_for_amp,
            task_rewards,
            normalizer=self.amp_normalizer,
          )
          self.amp_replay_buffer.insert(amp_obs, next_amp_obs_for_amp)

          obs = next_obs
          amp_obs = next_amp_obs.clone()
          self.alg.process_env_step(obs, amp_rewards, dones, extras)

          intrinsic_rewards = (
            self.alg.intrinsic_rewards
            if self.cfg["algorithm"].get("rnd_cfg")
            else None
          )
          self.logger.process_env_step(
            amp_rewards,
            dones,
            extras,
            intrinsic_rewards,
          )

        stop = time.time()
        collect_time = stop - start
        start = stop

        self.alg.compute_returns(obs)

      loss_dict = self.alg.update()
      loss_dict.update(self._update_amp())

      stop = time.time()
      learn_time = stop - start
      self.current_learning_iteration = it

      self.logger.log(
        it=it,
        start_it=start_it,
        total_it=total_it,
        collect_time=collect_time,
        learn_time=learn_time,
        loss_dict=loss_dict,
        learning_rate=self.alg.learning_rate,
        action_std=self.alg.get_policy().output_std,
        rnd_weight=(
          self.alg.rnd.weight if self.cfg["algorithm"].get("rnd_cfg") else None
        ),
      )

      if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
        self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore[arg-type]

    if self.logger.writer is not None:
      self.save(
        os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt")  # type: ignore[arg-type]
      )
      self.logger.stop_logging_writer()

  def save(self, path: str, infos=None) -> None:
    env_state = {
      "common_step_counter": self.env.unwrapped.common_step_counter,
    }
    infos = {**(infos or {}), "env_state": env_state}
    saved_dict = self.alg.save()
    saved_dict["amp_discriminator_state_dict"] = self.amp_discriminator.state_dict()
    saved_dict["amp_optimizer_state_dict"] = self.amp_optimizer.state_dict()
    if self.amp_normalizer is not None:
      saved_dict["amp_normalizer_state_dict"] = self.amp_normalizer.state_dict()
    saved_dict["iter"] = self.current_learning_iteration
    saved_dict["infos"] = infos
    torch.save(saved_dict, path)

    if self.cfg["upload_model"]:
      self.logger.save_model(path, self.current_learning_iteration)

    self._export_policy_with_metadata(path)

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    infos = super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location)
    loaded_dict = torch.load(path, map_location=map_location, weights_only=False)

    if "amp_discriminator_state_dict" in loaded_dict:
      self.amp_discriminator.load_state_dict(
        loaded_dict["amp_discriminator_state_dict"],
        strict=strict,
      )
    if load_cfg is None or load_cfg.get("amp_optimizer", True):
      if "amp_optimizer_state_dict" in loaded_dict:
        self.amp_optimizer.load_state_dict(loaded_dict["amp_optimizer_state_dict"])
    if self.amp_normalizer is not None and "amp_normalizer_state_dict" in loaded_dict:
      self.amp_normalizer.load_state_dict(loaded_dict["amp_normalizer_state_dict"])
    return infos

  def _update_amp(self) -> dict[str, float]:
    num_batches = (
      self.cfg["algorithm"]["num_learning_epochs"]
      * self.cfg["algorithm"]["num_mini_batches"]
    )
    policy_generator = self.amp_replay_buffer.feed_forward_generator(
      num_batches,
      self.amp_batch_size,
    )
    expert_generator = self.amp_motion_loader.feed_forward_generator(
      num_batches,
      self.amp_batch_size,
    )

    mean_amp_loss = 0.0
    mean_grad_pen_loss = 0.0
    mean_policy_pred = 0.0
    mean_expert_pred = 0.0
    mean_reward = 0.0

    for policy_sample, expert_sample in zip(policy_generator, expert_generator):
      policy_state, policy_next_state = policy_sample
      expert_state, expert_next_state = expert_sample

      if self.amp_normalizer is not None:
        self.amp_normalizer.update(policy_state)
        self.amp_normalizer.update(expert_state)
        policy_state = self.amp_normalizer.normalize(policy_state)
        policy_next_state = self.amp_normalizer.normalize(policy_next_state)
        expert_state = self.amp_normalizer.normalize(expert_state)
        expert_next_state = self.amp_normalizer.normalize(expert_next_state)

      policy_pred = self.amp_discriminator(
        torch.cat((policy_state, policy_next_state), dim=-1)
      )
      expert_pred = self.amp_discriminator(
        torch.cat((expert_state, expert_next_state), dim=-1)
      )
      expert_loss = nn.MSELoss()(
        expert_pred,
        torch.ones_like(expert_pred, device=self.device),
      )
      policy_loss = nn.MSELoss()(
        policy_pred,
        -torch.ones_like(policy_pred, device=self.device),
      )
      amp_loss = 0.5 * (expert_loss + policy_loss)
      grad_pen_loss = self.amp_discriminator.compute_grad_pen(
        expert_state,
        expert_next_state,
        lambda_=self.amp_grad_penalty_coef,
      )
      loss = amp_loss + grad_pen_loss

      self.amp_optimizer.zero_grad()
      loss.backward()
      nn.utils.clip_grad_norm_(
        self.amp_discriminator.parameters(),
        self.amp_max_grad_norm,
      )
      self.amp_optimizer.step()

      with torch.no_grad():
        reward, _ = self.amp_discriminator.predict_amp_reward(
          policy_state,
          policy_next_state,
          torch.zeros(policy_state.shape[0], device=self.device),
          normalizer=None,
        )

      mean_amp_loss += amp_loss.item()
      mean_grad_pen_loss += grad_pen_loss.item()
      mean_policy_pred += policy_pred.mean().item()
      mean_expert_pred += expert_pred.mean().item()
      mean_reward += reward.mean().item()

    return {
      "amp": mean_amp_loss / num_batches,
      "amp_grad_pen": mean_grad_pen_loss / num_batches,
      "amp_policy_pred": mean_policy_pred / num_batches,
      "amp_expert_pred": mean_expert_pred / num_batches,
      "amp_reward": mean_reward / num_batches,
    }

  def _env_step_dt(self) -> float:
    step_dt = getattr(self.env.unwrapped, "step_dt", None)
    if step_dt is not None:
      return float(step_dt)
    cfg = self.env.unwrapped.cfg
    return float(cfg.sim.mujoco.timestep * cfg.decimation)

  def _export_policy_with_metadata(self, checkpoint_path: str) -> None:
    policy_path = os.path.dirname(checkpoint_path)
    filename = "policy.onnx"
    self.export_policy_to_onnx(policy_path, filename)
    logger_type = getattr(self.logger, "logger_type", "local")
    run_name: str = (
      wandb.run.name if logger_type == "wandb" and wandb.run else "local"
    )  # type: ignore[assignment]
    onnx_path = os.path.join(policy_path, filename)
    metadata = get_base_metadata(self.env.unwrapped, run_name)
    attach_metadata_to_onnx(onnx_path, metadata)
    if logger_type == "wandb":
      wandb.save(os.path.join(policy_path, filename), base_path=policy_path)
