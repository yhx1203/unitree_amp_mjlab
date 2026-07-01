"""View a G1 motion CSV directly in the MuJoCo viewer."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Literal


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "motion_file",
    nargs="?",
    default="src/assets/motions/g1/walk1_subject1.csv",
    help="CSV motion file. Layout: root_pos(3), root_quat(4), joints(29).",
  )
  parser.add_argument(
    "--xml",
    default="src/assets/robots/unitree_g1/xmls/scene_g1.xml",
    help="MuJoCo scene XML path.",
  )
  parser.add_argument("--fps", type=float, default=30.0, help="CSV playback FPS.")
  parser.add_argument("--speed", type=float, default=1.0, help="Playback speed scale.")
  parser.add_argument(
    "--quat-order",
    choices=("xyzw", "wxyz"),
    default="xyzw",
    help="Quaternion order in the CSV.",
  )
  parser.add_argument("--start", type=int, default=0, help="Start frame index.")
  parser.add_argument("--end", type=int, default=None, help="End frame index, exclusive.")
  parser.add_argument("--once", action="store_true", help="Play once instead of looping.")
  return parser.parse_args()


def csv_quat_to_mujoco(
  quat,
  quat_order: Literal["xyzw", "wxyz"],
):
  import numpy as np

  if quat_order == "xyzw":
    quat = quat[[3, 0, 1, 2]]
  quat_norm = np.linalg.norm(quat)
  if quat_norm < 1.0e-8:
    raise ValueError("Encountered a near-zero root quaternion in the motion CSV.")
  return quat / quat_norm


def load_motion(path: Path, start: int, end: int | None):
  import numpy as np

  motion = np.loadtxt(path, delimiter=",", dtype=np.float64)
  if motion.ndim == 1:
    motion = motion[None, :]
  return motion[start:end]


def main() -> None:
  args = parse_args()
  import mujoco
  import mujoco.viewer

  xml_path = Path(args.xml)
  motion_path = Path(args.motion_file)
  if not xml_path.exists():
    raise FileNotFoundError(f"XML file not found: {xml_path}")
  if not motion_path.exists():
    raise FileNotFoundError(f"Motion CSV not found: {motion_path}")
  if args.fps <= 0.0:
    raise ValueError("--fps must be positive.")
  if args.speed <= 0.0:
    raise ValueError("--speed must be positive.")

  model = mujoco.MjModel.from_xml_path(str(xml_path))
  data = mujoco.MjData(model)
  motion = load_motion(motion_path, args.start, args.end)

  expected_cols = model.nq
  if motion.shape[1] != expected_cols:
    raise ValueError(
      f"Motion has {motion.shape[1]} columns, but model.nq is {expected_cols}. "
      "Expected root_pos(3), root_quat(4), then one value per joint qpos."
    )

  frame_dt = 1.0 / (args.fps * args.speed)
  print(
    f"Playing {motion_path} on {xml_path}: "
    f"{motion.shape[0]} frames, {args.fps:g} fps, speed {args.speed:g}x."
  )

  with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
      for frame in motion:
        if not viewer.is_running():
          break

        frame_start = time.perf_counter()
        data.qpos[0:3] = frame[0:3]
        data.qpos[3:7] = csv_quat_to_mujoco(frame[3:7], args.quat_order)
        data.qpos[7:] = frame[7:]

        mujoco.mj_forward(model, data)
        viewer.sync()

        sleep_time = frame_dt - (time.perf_counter() - frame_start)
        if sleep_time > 0.0:
          time.sleep(sleep_time)

      if args.once:
        break


if __name__ == "__main__":
  main()
