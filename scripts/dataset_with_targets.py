# Author: Tatiana Fedorova

# This script loads image pairs, COLMAP depth maps and camera parameters.
# It prepares target 3D point maps for DUSt3R training.

from pathlib import Path
import json
import argparse
import numpy as np
from PIL import Image


def read_colmap_array(path: Path):
    with open(path, "rb") as f:
        width = b""
        while True:
            c = f.read(1)
            if c == b"&":
                break
            width += c
        height = b""
        while True:
            c = f.read(1)
            if c == b"&":
                break
            height += c
        channels = b""
        while True:
            c = f.read(1)
            if c == b"&":
                break
            channels += c

        width = int(width.decode("utf-8"))
        height = int(height.decode("utf-8"))
        channels = int(channels.decode("utf-8"))

        data = np.fromfile(f, np.float32)
        arr = data.reshape((width, height, channels), order="F")
        arr = np.transpose(arr, (1, 0, 2))
        if channels == 1:
            arr = arr[:, :, 0]
        return arr


def scale_K(K, sx, sy):
    K2 = np.array(K, dtype=np.float32).copy()
    K2[0, 0] *= sx
    K2[0, 2] *= sx
    K2[1, 1] *= sy
    K2[1, 2] *= sy
    return K2


def depth_to_pointmap(depth: np.ndarray, K: np.ndarray):
    h, w = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    z = depth.astype(np.float32)
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy

    pts = np.stack([x, y, z], axis=-1)
    invalid = ~np.isfinite(z) | (z <= 0)
    pts[invalid] = 0.0
    return pts, ~invalid


def transform_pointmap(points: np.ndarray, T_ab: np.ndarray):
    h, w, _ = points.shape
    pts = points.reshape(-1, 3)
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    pts_h = np.concatenate([pts, ones], axis=1)  # (N,4)
    out = (T_ab @ pts_h.T).T[:, :3]
    return out.reshape(h, w, 3)


class Dust3rTargetDataset:
    def __init__(self, pairs_json_path):
        with open(pairs_json_path, "r", encoding="utf-8") as f:
            self.pairs = json.load(f)

    def __len__(self):
        return len(self.pairs)

    def _load_view(self, sample):
        img = Image.open(sample["image_path"]).convert("RGB")
        depth = read_colmap_array(Path(sample["depth_path"]))

        orig_w, orig_h = img.size
        depth_h, depth_w = depth.shape
        sx = depth_w / orig_w
        sy = depth_h / orig_h

        img = img.resize((depth_w, depth_h), Image.BILINEAR)
        img = np.asarray(img, dtype=np.uint8)

        K = scale_K(sample["K"], sx, sy)
        T_cw = np.array(sample["T_cw"], dtype=np.float32)
        T_wc = np.array(sample["T_wc"], dtype=np.float32)

        points_cam, valid_mask = depth_to_pointmap(depth, K)

        return {
            "image_name": sample["image_name"],
            "image": img,
            "depth": depth.astype(np.float32),
            "K": K,
            "T_cw": T_cw,
            "T_wc": T_wc,
            "points_cam": points_cam,
            "valid_mask": valid_mask,
        }

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        v1 = self._load_view(pair["sample1"])
        v2 = self._load_view(pair["sample2"])

        gt_X1_1 = v1["points_cam"]
        T_12 = v1["T_cw"] @ v2["T_wc"]   # cam2 -> cam1
        gt_X2_1 = transform_pointmap(v2["points_cam"], T_12)

        return {
            "pair_type": pair["pair_type"],
            "view1": v1,
            "view2": v2,
            "gt_X1_1": gt_X1_1.astype(np.float32),
            "gt_X2_1": gt_X2_1.astype(np.float32),
            "valid1": v1["valid_mask"],
            "valid2": v2["valid_mask"],
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs_json", required=True, help="Path to pairs JSON file")
    args = parser.parse_args()

    ds = Dust3rTargetDataset(args.pairs_json)
    item = ds[0]

    print("dataset length:", len(ds))
    print("pair_type:", item["pair_type"])
    print("gt_X1_1:", item["gt_X1_1"].shape, item["gt_X1_1"].dtype)
    print("gt_X2_1:", item["gt_X2_1"].shape, item["gt_X2_1"].dtype)
    print("valid1:", int(item["valid1"].sum()))
    print("valid2:", int(item["valid2"].sum()))