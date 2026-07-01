from __future__ import annotations

import torch


class AmpReplayBuffer:
  """Fixed-size replay buffer for policy AMP transitions."""

  def __init__(self, obs_dim: int, buffer_size: int, device: str) -> None:
    self.states = torch.zeros(buffer_size, obs_dim, device=device)
    self.next_states = torch.zeros(buffer_size, obs_dim, device=device)
    self.buffer_size = buffer_size
    self.device = device
    self.step = 0
    self.num_samples = 0

  def insert(self, states: torch.Tensor, next_states: torch.Tensor) -> None:
    states = states.detach().to(self.device)
    next_states = next_states.detach().to(self.device)
    num_states = states.shape[0]
    if num_states == 0:
      return

    if num_states >= self.buffer_size:
      self.states[:] = states[-self.buffer_size :]
      self.next_states[:] = next_states[-self.buffer_size :]
      self.step = 0
      self.num_samples = self.buffer_size
      return

    end_idx = self.step + num_states
    if end_idx > self.buffer_size:
      first_count = self.buffer_size - self.step
      self.states[self.step :] = states[:first_count]
      self.next_states[self.step :] = next_states[:first_count]
      wrap_count = end_idx - self.buffer_size
      self.states[:wrap_count] = states[first_count:]
      self.next_states[:wrap_count] = next_states[first_count:]
    else:
      self.states[self.step : end_idx] = states
      self.next_states[self.step : end_idx] = next_states

    self.step = end_idx % self.buffer_size
    self.num_samples = min(self.buffer_size, self.num_samples + num_states)

  def feed_forward_generator(
    self,
    num_mini_batches: int,
    mini_batch_size: int,
  ):
    if self.num_samples <= 0:
      raise RuntimeError("AMP replay buffer is empty.")

    for _ in range(num_mini_batches):
      sample_idxs = torch.randint(
        low=0,
        high=self.num_samples,
        size=(mini_batch_size,),
        device=self.device,
      )
      yield self.states[sample_idxs], self.next_states[sample_idxs]

