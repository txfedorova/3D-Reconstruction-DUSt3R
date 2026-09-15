# Author: Tatiana Fedorova

# This script prepares a COLMAP-compatible directory structure for a multi-camera rig.
# It finds frame IDs that are present in all camera folders and creates files
# named frame_XXXX.jpg in output folders cam0...cam5.

from pathlib import Path
import re
import os
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--src_root", required=True, help="Path to source directory with cam0...cam5")
parser.add_argument("--dst_root", required=True, help="Path to output COLMAP rig directory")
args = parser.parse_args()

src_root = Path(args.src_root)
dst_root = Path(args.dst_root)

cams = [f"cam{i}" for i in range(6)]
pattern = re.compile(r"cam(\d+)_frame_(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)

frames_per_cam = {}

for cam in cams:
    cam_dir = src_root / cam
    ids = set()
    if not cam_dir.exists():
        raise FileNotFoundError(f"Camera directory not found: {cam_dir}")

    for p in cam_dir.iterdir():
        if not p.is_file():
            continue
        m = pattern.match(p.name)
        if m:
            ids.add(m.group(2))
    frames_per_cam[cam] = ids

common_ids = set.intersection(*(frames_per_cam[c] for c in cams))
common_ids = sorted(common_ids)

print(f"Common frame count across all cameras: {len(common_ids)}")
if not common_ids:
    raise RuntimeError("No common frame ids found across all 6 cameras.")

for cam in cams:
    (dst_root / cam).mkdir(parents=True, exist_ok=True)

for cam in cams:
    cam_dir = src_root / cam
    for frame_id in common_ids:
        candidates = list(cam_dir.glob(f"{cam}_frame_{frame_id}.*"))
        if len(candidates) != 1:
            print(f"[WARN] {cam} frame {frame_id}: expected 1 file, found {len(candidates)}")
            continue

        src_file = candidates[0]
        ext = src_file.suffix.lower()
        dst_file = dst_root / cam / f"frame_{frame_id}{ext}"

        if dst_file.exists():
            dst_file.unlink()

        try:
            os.link(src_file, dst_file)
        except OSError:
            shutil.copy2(src_file, dst_file)

print("Done.")
