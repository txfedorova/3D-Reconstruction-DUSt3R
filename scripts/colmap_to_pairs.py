# Author: Tatiana Fedorova

# This script creates image pairs for DUSt3R training from samples.json.
# It creates temporal pairs within one camera and cross-camera pairs
# between neighboring cameras at the same timestamp.

from pathlib import Path
import json
import re
import argparse
from collections import defaultdict


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, help="Path to COLMAP dataset directory")
args = parser.parse_args()

DATASET = Path(args.dataset)
SAMPLES_PATH = DATASET / "samples.json"
OUT_PATH = DATASET / "pairs.json"

FRAME_RE = re.compile(r"rig0/(cam\d+)/frame_(\d+)\.jpg")


def parse_sample(sample):
    m = FRAME_RE.fullmatch(sample["image_name"])
    if not m:
        return None
    cam = m.group(1)
    frame_id = int(m.group(2))
    return cam, frame_id


def main():
    with open(SAMPLES_PATH, "r", encoding="utf-8") as f:
        samples = json.load(f)

    by_cam = defaultdict(list)
    by_frame = defaultdict(dict)

    for s in samples:
        parsed = parse_sample(s)
        if parsed is None:
            continue
        cam, frame_id = parsed
        by_cam[cam].append((frame_id, s))
        by_frame[frame_id][cam] = s

    for cam in by_cam:
        by_cam[cam].sort(key=lambda x: x[0])

    pairs = []

    # 1) temporal pairs within each camera
    for cam, items in by_cam.items():
        n = len(items)
        for i in range(n):
            for delta in (1, 2):
                j = i + delta
                if j < n:
                    s1 = items[i][1]
                    s2 = items[j][1]
                    pairs.append({
                        "pair_type": f"temporal_d{delta}",
                        "sample1": s1,
                        "sample2": s2,
                    })

    # 2) cross-camera pairs at same timestamp
    cam_neighbors = [("cam0", "cam1"), ("cam1", "cam2"), ("cam2", "cam3"), ("cam3", "cam4"), ("cam4", "cam5")]

    for frame_id, cams in by_frame.items():
        for c1, c2 in cam_neighbors:
            if c1 in cams and c2 in cams:
                pairs.append({
                    "pair_type": "cross_camera",
                    "sample1": cams[c1],
                    "sample2": cams[c2],
                })

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)

    print(f"Total samples: {len(samples)}")
    print(f"Total pairs: {len(pairs)}")
    print(f"Saved to: {OUT_PATH}")


if __name__ == "__main__":
    main()
