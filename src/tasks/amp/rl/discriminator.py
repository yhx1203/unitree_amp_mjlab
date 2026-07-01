from __future__ import annotations

import torch
import torch.nn as nn
from torch import autograd


class AmpRunningNormalizer:
  """Running mean/std normalizer for AMP observations."""

  def __init__(
    self,
    obs_dim: int,
    device: str,
    epsilon: float = 1.0e-4,
    clip: float | None = 5.0,
  ) -> None:
    self.device = device
    self.clip = clip
    self.mean = torch.zeros(obs_dim, device=device)
    self.var = torch.ones(obs_dim, device=device)
    self.count = torch.tensor(epsilon, device=device)

  def update(self, x: torch.Tensor) -> None:
    x = x.detach().to(self.device)
    if x.numel() == 0:
      return
    batch_mean = torch.mean(x, dim=0)
    batch_var = torch.var(x, dim=0, unbiased=False)
    batch_count = torch.tensor(float(x.shape[0]), device=self.device)
    self._update_from_moments(batch_mean, batch_var, batch_count)

  def normalize(self, x: torch.Tensor) -> torch.Tensor:
    x = (x - self.mean) / torch.sqrt(self.var + 1.0e-8)
    if self.clip is not None:
      x = torch.clamp(x, -self.clip, self.clip)
    return x

  def state_dict(self) -> dict[str, torch.Tensor | float | None]:
    return {
      "mean": self.mean,
      "var": self.var,
      "count": self.count,
      "clip": self.clip,
    }

  def load_state_dict(self, state_dict: dict) -> None:
    self.mean = state_dict["mean"].to(self.device)
    self.var = state_dict["var"].to(self.device)
    self.count = state_dict["count"].to(self.device)
    self.clip = state_dict.get("clip", self.clip)

  def _update_from_moments(
    self,
    batch_mean: torch.Tensor,
    batch_var: torch.Tensor,
    batch_count: torch.Tensor,
  ) -> None:
    delta = batch_mean - self.mean
    total_count = self.count + batch_count

    new_mean = self.mean + delta * batch_count / total_count
    m_a = self.var * self.count
    m_b = batch_var * batch_count
    m_2 = m_a + m_b + torch.square(delta) * self.count * batch_count / total_count

    self.mean = new_mean
    self.var = m_2 / total_count
    self.count = total_count


class AmpDiscriminator(nn.Module):
  """Discriminator used for adversarial motion prior rewards."""

  def __init__(
    self,
    obs_dim: int,
    reward_coef: float,
    hidden_dims: tuple[int, ...],
    task_reward_lerp: float = 0.0,
    reward_scale: float = 1.0,
  ) -> None:
    super().__init__()
    self.obs_dim = obs_dim
    self.input_dim = obs_dim * 2
    self.reward_coef = reward_coef
    self.task_reward_lerp = task_reward_lerp
    self.reward_scale = reward_scale

    layers: list[nn.Module] = []
    last_dim = self.input_dim
    for hidden_dim in hidden_dims:
      layers.append(nn.Linear(last_dim, hidden_dim))
      layers.append(nn.ReLU())
      last_dim = hidden_dim

    self.trunk = nn.Sequential(*layers)
    self.amp_linear = nn.Linear(last_dim, 1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.amp_linear(self.trunk(x))

  def compute_grad_pen(
    self,
    expert_state: torch.Tensor,
    expert_next_state: torch.Tensor,
    lambda_: float,
  ) -> torch.Tensor:
    expert_data = torch.cat((expert_state, expert_next_state), dim=-1)
    expert_data = expert_data.detach().requires_grad_(True)
    disc = self.forward(expert_data)
    ones = torch.ones_like(disc)
    grad = autograd.grad(
      outputs=disc,
      inputs=expert_data,
      grad_outputs=ones,
      create_graph=True,
      retain_graph=True,
      only_inputs=True,
    )[0]
    return lambda_ * torch.square(torch.linalg.norm(grad, ord=2, dim=1)).mean()

  def predict_amp_reward(
    self,
    state: torch.Tensor,
    next_state: torch.Tensor,
    task_reward: torch.Tensor,
    normalizer: AmpRunningNormalizer | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
      was_training = self.training
      self.eval()
      if normalizer is not None:
        state = normalizer.normalize(state)
        next_state = normalizer.normalize(next_state)

      disc = self.forward(torch.cat((state, next_state), dim=-1))
      amp_reward = self.reward_scale * self.reward_coef * torch.clamp(
        1.0 - 0.25 * torch.square(disc - 1.0),
        min=0.0,
      )
      amp_reward = amp_reward.squeeze(-1)

      if self.task_reward_lerp > 0.0:
        task_reward = task_reward.reshape_as(amp_reward)
        amp_reward = (
          (1.0 - self.task_reward_lerp) * amp_reward
          + self.task_reward_lerp * task_reward
        )

      if was_training:
        self.train()
    return amp_reward, disc.squeeze(-1)
