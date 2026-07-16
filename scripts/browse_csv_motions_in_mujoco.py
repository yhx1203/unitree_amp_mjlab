"""Browse a directory of G1 motion CSV files in the MuJoCo viewer."""

from __future__ import annotations

import argparse
import re
import threading
import time
from pathlib import Path

import numpy as np


KEY_SPACE = 32
KEY_R = 82
KEY_RIGHT = 262
KEY_LEFT = 263


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "motion_path",
    type=Path,
    nargs="?",
    default=Path("src/assets/motions/g1/retargeted"),
    help="A motion CSV or a directory searched recursively for CSV files.",
  )
  parser.add_argument(
    "--xml",
    type=Path,
    default=Path("src/assets/robots/unitree_g1/xmls/scene_g1.xml"),
    help="MuJoCo scene XML path.",
  )
  parser.add_argument("--fps", type=float, default=30.0, help="Motion frame rate.")
  parser.add_argument("--speed", type=float, default=1.0, help="Playback speed scale.")
  parser.add_argument(
    "--start-index",
    type=int,
    default=0,
    help="Initial motion index in natural filename order.",
  )
  parser.add_argument(
    "--filter",
    default=None,
    help="Only include paths containing this case-insensitive text.",
  )
  return parser.parse_args()


def natural_sort_key(path: Path) -> tuple[object, ...]:
  parts = re.split(r"(\d+)", path.as_posix().lower())
  return tuple(int(part) if part.isdigit() else part for part in parts)


def discover_motions(path: Path, name_filter: str | None = None) -> list[Path]:
  path = path.resolve()
  if path.is_file():
    if path.suffix.lower() != ".csv":
      raise ValueError(f"Motion file must have a .csv suffix: {path}")
    paths = [path]
  elif path.is_dir():
    paths = sorted(path.rglob("*.csv"), key=natural_sort_key)
  else:
    raise FileNotFoundError(f"Motion path does not exist: {path}")

  if name_filter:
    normalized_filter = name_filter.casefold()
    paths = [path for path in paths if normalized_filter in path.as_posix().casefold()]
  if not paths:
    raise FileNotFoundError(f"No matching CSV motions found under: {path}")
  return paths


def load_motion(path: Path, expected_columns: int) -> np.ndarray:
  motion = np.loadtxt(path, delimiter=",", dtype=np.float64)
  if motion.ndim == 1:
    motion = motion[None, :]
  if motion.shape[1] != expected_columns:
    raise ValueError(
      f"{path}: found {motion.shape[1]} columns, expected {expected_columns}. "
      "Expected root_pos(3), root_quat_xyzw(4), and one value per model joint."
    )
  if motion.shape[0] == 0 or not np.all(np.isfinite(motion)):
    raise ValueError(f"{path}: motion is empty or contains NaN/infinity")
  quaternion_norm = np.linalg.norm(motion[:, 3:7], axis=1)
  if np.any(quaternion_norm < 1.0e-8):
    raise ValueError(f"{path}: motion contains a near-zero root quaternion")
  return motion


class PlaybackControls:
  """Collect viewer key events for the playback loop."""

  def __init__(self) -> None:
    self._lock = threading.Lock()
    self._motion_delta = 0
    self._toggle_pause = False
    self._restart = False

  def handle_key(self, key: int) -> None:
    with self._lock:
      if key == KEY_LEFT:
        self._motion_delta -= 1
      elif key == KEY_RIGHT:
        self._motion_delta += 1
      elif key == KEY_SPACE:
        self._toggle_pause = not self._toggle_pause
      elif key == KEY_R:
        self._restart = True

  def consume(self) -> tuple[int, bool, bool]:
    with self._lock:
      result = (self._motion_delta, self._toggle_pause, self._restart)
      self._motion_delta = 0
      self._toggle_pause = False
      self._restart = False
    return result


def apply_frame(model, data, frame: np.ndarray) -> None:
  import mujoco

  data.qpos[0:3] = frame[0:3]
  # Dataset/project CSV quaternions are xyzw; MuJoCo qpos uses wxyz.
  quaternion_xyzw = frame[3:7]
  data.qpos[3:7] = quaternion_xyzw[[3, 0, 1, 2]] / np.linalg.norm(
    quaternion_xyzw
  )
  data.qpos[7:] = frame[7:]
  mujoco.mj_forward(model, data)


def describe_motion(index: int, paths: list[Path], motion: np.ndarray, fps: float) -> None:
  print(
    f"[MOTION {index + 1:03d}/{len(paths):03d}] {paths[index]} | "
    f"{motion.shape[0]} frames | {motion.shape[0] / fps:.2f}s"
  )


def main() -> None:
  args = parse_args()
  if args.fps <= 0.0:
    raise ValueError("--fps must be positive")
  if args.speed <= 0.0:
    raise ValueError("--speed must be positive")
  if not args.xml.exists():
    raise FileNotFoundError(f"MuJoCo XML does not exist: {args.xml}")

  import mujoco
  import mujoco.viewer

  model = mujoco.MjModel.from_xml_path(str(args.xml.resolve()))
  data = mujoco.MjData(model)
  paths = discover_motions(args.motion_path, args.filter)
  motion_index = args.start_index % len(paths)
  motion = load_motion(paths[motion_index], model.nq)
  frame_index = 0
  paused = False
  controls = PlaybackControls()
  frame_period = 1.0 / (args.fps * args.speed)

  apply_frame(model, data, motion[frame_index])
  describe_motion(motion_index, paths, motion, args.fps)
  print("[KEYS] Left/Right: previous/next | Space: pause/resume | R: restart")

  with mujoco.viewer.launch_passive(
    model,
    data,
    key_callback=controls.handle_key,
  ) as viewer:
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if pelvis_id >= 0:
      viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
      viewer.cam.trackbodyid = pelvis_id
      viewer.cam.distance = 3.0
      viewer.cam.azimuth = 135.0
      viewer.cam.elevation = -15.0

    viewer.sync()
    next_frame_time = time.perf_counter() + frame_period
    while viewer.is_running():
      motion_delta, toggle_pause, restart = controls.consume()
      if toggle_pause:
        paused = not paused
        print("[PAUSED]" if paused else "[PLAYING]")

      if motion_delta:
        motion_index = (motion_index + motion_delta) % len(paths)
        motion = load_motion(paths[motion_index], model.nq)
        frame_index = 0
        apply_frame(model, data, motion[frame_index])
        describe_motion(motion_index, paths, motion, args.fps)
        viewer.sync()
        next_frame_time = time.perf_counter() + frame_period
      elif restart:
        frame_index = 0
        apply_frame(model, data, motion[frame_index])
        print(f"[RESTART] {paths[motion_index].name}")
        viewer.sync()
        next_frame_time = time.perf_counter() + frame_period

      if paused:
        time.sleep(0.01)
        continue

      now = time.perf_counter()
      if now < next_frame_time:
        time.sleep(min(next_frame_time - now, 0.01))
        continue

      frame_index = (frame_index + 1) % motion.shape[0]
      apply_frame(model, data, motion[frame_index])
      viewer.sync()
      next_frame_time += frame_period
      if next_frame_time < now - frame_period:
        next_frame_time = now + frame_period


if __name__ == "__main__":
  main()
