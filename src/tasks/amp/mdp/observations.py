from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def amp_joint_state(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Return the AMP state used by the discriminator.

  The state is intentionally aligned with the motion loader: relative joint
  positions followed by relative joint velocities.
  """
  asset: Entity = env.scene[asset_cfg.name]
  default_joint_pos = asset.data.default_joint_pos
  default_joint_vel = asset.data.default_joint_vel
  assert default_joint_pos is not None
  assert default_joint_vel is not None

  joint_ids = asset_cfg.joint_ids
  joint_pos = asset.data.joint_pos[:, joint_ids] - default_joint_pos[:, joint_ids]
  joint_vel = asset.data.joint_vel[:, joint_ids] - default_joint_vel[:, joint_ids]
  return torch.cat((joint_pos, joint_vel), dim=-1)

