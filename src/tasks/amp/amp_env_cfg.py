"""AMP task environment helpers."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

import src.tasks.amp.mdp as mdp


def add_amp_observation_group(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  """Add discriminator-only AMP observations to an environment config."""
  cfg.observations["amp"] = ObservationGroupCfg(
    terms={
      "joint_state": ObservationTermCfg(
        func=mdp.amp_joint_state,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
      )
    },
    concatenate_terms=True,
    enable_corruption=False,
    history_length=1,
  )
  return cfg

