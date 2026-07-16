"""Convert Unitree G1 retargeted PKL motions to this project's CSV format.

Both the older 23-DOF plain-pickle dataset and the compressed 29-DOF
Bones-SEED motion_lib files are supported.  Only load PKLs from a trusted
source because pickle/joblib is not a safe interchange format.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np


SOURCE_JOINT_NAMES_23 = (
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
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
)

TARGET_JOINT_NAMES = (
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

# Mapping for the older 23-DOF files, which use the 29-DOF model order with
# all six wrist joints omitted.
SOURCE_TO_TARGET = (
  0,
  1,
  2,
  3,
  4,
  5,
  6,
  7,
  8,
  9,
  10,
  11,
  12,
  13,
  14,
  15,
  16,
  17,
  18,
  22,
  23,
  24,
  25,
)

FILLED_WITH_ZERO = (
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "input",
    type=Path,
    nargs="?",
    default=Path("g1-retargeted-motions"),
    help="A PKL file or directory searched recursively for PKL files.",
  )
  parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path("src/assets/motions/g1/retargeted"),
    help="Output root for 36-column CSV files and manifest.json.",
  )
  parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Replace existing output CSV files.",
  )
  return parser.parse_args()


def load_motion(path: Path) -> tuple[str, dict[str, Any]]:
  payload = joblib.load(path)
  if not isinstance(payload, dict) or len(payload) != 1:
    raise ValueError(f"{path}: expected one motion in the outer dictionary")

  motion_key, motion = next(iter(payload.items()))
  if not isinstance(motion, dict):
    raise ValueError(f"{path}: motion value must be a dictionary")
  return str(motion_key), motion


def convert_motion(path: Path) -> tuple[np.ndarray, int, str, int]:
  motion_key, motion = load_motion(path)
  missing = {"root_trans_offset", "root_rot", "dof", "fps"} - motion.keys()
  if missing:
    raise ValueError(f"{path}: missing fields {sorted(missing)}")

  root_pos = np.asarray(motion["root_trans_offset"], dtype=np.float64)
  root_quat_xyzw = np.asarray(motion["root_rot"], dtype=np.float64)
  source_joint_pos = np.asarray(motion["dof"], dtype=np.float64)
  fps = int(np.asarray(motion["fps"]).reshape(-1)[0])

  num_frames = root_pos.shape[0]
  if root_pos.shape != (num_frames, 3):
    raise ValueError(f"{path}: root_trans_offset shape is {root_pos.shape}, expected (T, 3)")
  if root_quat_xyzw.shape != (num_frames, 4):
    raise ValueError(f"{path}: root_rot shape is {root_quat_xyzw.shape}, expected (T, 4)")
  source_dofs = source_joint_pos.shape[1] if source_joint_pos.ndim == 2 else -1
  if source_joint_pos.shape[0:1] != (num_frames,) or source_dofs not in (
    len(SOURCE_JOINT_NAMES_23),
    len(TARGET_JOINT_NAMES),
  ):
    raise ValueError(
      f"{path}: dof shape is {source_joint_pos.shape}, "
      f"expected (T, {len(SOURCE_JOINT_NAMES_23)}) or "
      f"(T, {len(TARGET_JOINT_NAMES)})"
    )
  if fps <= 0:
    raise ValueError(f"{path}: fps must be positive, got {fps}")

  quaternion_norm = np.linalg.norm(root_quat_xyzw, axis=1, keepdims=True)
  if np.any(quaternion_norm < 1.0e-8):
    raise ValueError(f"{path}: root_rot contains a near-zero quaternion")
  root_quat_xyzw = root_quat_xyzw / quaternion_norm

  if source_dofs == len(TARGET_JOINT_NAMES):
    # Bones-SEED motion_lib DOFs are already in the G1 MJCF/MuJoCo order.
    target_joint_pos = source_joint_pos
  else:
    # Older retargeted files omit the six wrist joints.
    target_joint_pos = np.zeros((num_frames, len(TARGET_JOINT_NAMES)), dtype=np.float64)
    target_joint_pos[:, SOURCE_TO_TARGET] = source_joint_pos
  output = np.concatenate((root_pos, root_quat_xyzw, target_joint_pos), axis=1)
  if output.shape != (num_frames, 36):
    raise AssertionError(f"Internal error: converted shape is {output.shape}")
  if not np.all(np.isfinite(output)):
    raise ValueError(f"{path}: converted motion contains NaN or infinity")
  return output, fps, motion_key, source_dofs


def find_inputs(input_path: Path) -> tuple[list[Path], Path]:
  input_path = input_path.resolve()
  if input_path.is_file():
    if input_path.suffix.lower() != ".pkl":
      raise ValueError(f"Input file must have a .pkl suffix: {input_path}")
    return [input_path], input_path.parent
  if not input_path.is_dir():
    raise FileNotFoundError(f"Input does not exist: {input_path}")
  paths = sorted(input_path.rglob("*.pkl"))
  if not paths:
    raise FileNotFoundError(f"No PKL files found under: {input_path}")
  return paths, input_path


def main() -> None:
  args = parse_args()
  input_paths, input_root = find_inputs(args.input)
  output_root = args.output_dir.resolve()
  manifest_entries = []

  for index, input_path in enumerate(input_paths, start=1):
    relative_path = input_path.relative_to(input_root).with_suffix(".csv")
    output_path = output_root / relative_path
    if output_path.exists() and not args.overwrite:
      raise FileExistsError(f"Output exists; pass --overwrite to replace it: {output_path}")

    motion, fps, motion_key, source_dofs = convert_motion(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, motion, delimiter=",", fmt="%.6f")
    manifest_entries.append(
      {
        "input": str(input_path.relative_to(input_root)),
        "output": str(relative_path),
        "motion_key": motion_key,
        "frames": int(motion.shape[0]),
        "fps": fps,
        "duration_s": motion.shape[0] / fps,
        "source_dofs": source_dofs,
      }
    )
    print(f"[{index:03d}/{len(input_paths):03d}] {relative_path} ({motion.shape[0]} frames)")

  manifest = {
    "format": "root_pos_xyz(3), root_quat_xyzw(4), joint_pos(29)",
    "supported_source_joint_counts": (
      len(SOURCE_JOINT_NAMES_23),
      len(TARGET_JOINT_NAMES),
    ),
    "source_joint_names_23": SOURCE_JOINT_NAMES_23,
    "target_joint_names": TARGET_JOINT_NAMES,
    "source_to_target_indices": SOURCE_TO_TARGET,
    "filled_with_zero": FILLED_WITH_ZERO,
    "motions": manifest_entries,
  }
  manifest_path = output_root / "manifest.json"
  manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
  total_frames = sum(entry["frames"] for entry in manifest_entries)
  print(
    f"Converted {len(manifest_entries)} motions ({total_frames} frames) to "
    f"{output_root}"
  )
  print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
  main()
