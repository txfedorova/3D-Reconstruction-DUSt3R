# Author: Tatiana Fedorova

# This script converts camera rig calibration from YAML format
# to COLMAP rig_config.json format.

from pathlib import Path
import json
import yaml
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--yaml_path", required=True, help="Path to input camera calibration YAML file")
parser.add_argument("--out_path", required=True, help="Path to output COLMAP rig_config.json")
parser.add_argument("--rig_name", default="rig0", help="Name of the COLMAP rig")
args = parser.parse_args()

YAML_PATH = Path(args.yaml_path)
OUT_PATH = Path(args.out_path)
RIG_NAME = args.rig_name

def rotmat_to_quat_wxyz(R: np.ndarray):
    """Rotation matrix -> quaternion in COLMAP order [qw, qx, qy, qz]."""
    q = np.empty(4, dtype=float)
    trace = np.trace(R)

    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2.0
        q[0] = 0.25 * s
        q[1] = (R[2, 1] - R[1, 2]) / s
        q[2] = (R[0, 2] - R[2, 0]) / s
        q[3] = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            q[0] = (R[2, 1] - R[1, 2]) / s
            q[1] = 0.25 * s
            q[2] = (R[0, 1] + R[1, 0]) / s
            q[3] = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            q[0] = (R[0, 2] - R[2, 0]) / s
            q[1] = (R[0, 1] + R[1, 0]) / s
            q[2] = 0.25 * s
            q[3] = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            q[0] = (R[1, 0] - R[0, 1]) / s
            q[1] = (R[0, 2] + R[2, 0]) / s
            q[2] = (R[1, 2] + R[2, 1]) / s
            q[3] = 0.25 * s

    q /= np.linalg.norm(q)
    return q.tolist()


if not YAML_PATH.exists():
    raise FileNotFoundError(f"YAML file not found: {YAML_PATH}")

with open(YAML_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cams = [f"cam{i}" for i in range(6)]

# Absolute camera poses relative to cam0.
# T_cn_cnm1 transforms current camera from previous camera.
T_abs = {"cam0": np.eye(4)}

for i in range(1, 6):
    T_rel = np.array(cfg[f"cam{i}"]["T_cn_cnm1"], dtype=float)
    T_abs[f"cam{i}"] = T_rel @ T_abs[f"cam{i-1}"]

rig_entry = {"cameras": []}

for cam_name in cams:
    cam = cfg[cam_name]
    fx, fy, cx, cy = cam["intrinsics"]
    k1, k2, k3, k4 = cam["distortion_coeffs"]

    entry = {
        "image_prefix": f"{RIG_NAME}/{cam_name}/",
        "camera_model_name": "OPENCV_FISHEYE",
        "camera_params": [fx, fy, cx, cy, k1, k2, k3, k4],
    }

    if cam_name == "cam0":
        entry["ref_sensor"] = True
    else:
        T = T_abs[cam_name]
        entry["cam_from_rig_rotation"] = rotmat_to_quat_wxyz(T[:3, :3])
        entry["cam_from_rig_translation"] = T[:3, 3].tolist()

    rig_entry["cameras"].append(entry)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump([rig_entry], f, indent=2)

print(f"Saved: {OUT_PATH}")
