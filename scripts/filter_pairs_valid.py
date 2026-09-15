# Author: Tatiana Fedorova

# This script filters training and validation pairs according to the number
# of valid depth values after resizing to the target training resolution.

from pathlib import Path
import json
import argparse
import numpy as np
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, help="Path to COLMAP dataset directory")
args = parser.parse_args()

ROOT = Path(args.dataset)

FILES = [
    ROOT / "pairs_train.json",
    ROOT / "pairs_val.json",
]

MIN_VALID_PIXELS = 1000
TARGET_LONG_SIDE = 224


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
        expected = width * height * channels
        if data.size != expected:
            raise ValueError(f"Unexpected data size: got {data.size}, expected {expected}")

        arr = data.reshape((width, height, channels), order="F")
        arr = np.transpose(arr, (1, 0, 2))
        if channels == 1:
            arr = arr[:, :, 0]
        return arr.astype(np.float32)


def round_to_multiple_of_16(x):
    return max(16, int(round(x / 16.0)) * 16)


def compute_target_size(orig_w, orig_h, target_long_side):
    scale = target_long_side / max(orig_w, orig_h)
    new_w = round_to_multiple_of_16(orig_w * scale)
    new_h = round_to_multiple_of_16(orig_h * scale)
    return int(new_w), int(new_h)


def valid_count_after_resize(sample):
    img = Image.open(sample["image_path"]).convert("RGB")
    depth = read_colmap_array(Path(sample["depth_path"]))

    orig_w, orig_h = img.size
    target_w, target_h = compute_target_size(orig_w, orig_h, TARGET_LONG_SIDE)

    depth_img = Image.fromarray(depth.astype(np.float32), mode="F")
    depth_img = depth_img.resize((target_w, target_h), Image.NEAREST)
    depth = np.array(depth_img, dtype=np.float32)

    valid = np.isfinite(depth) & (depth > 0)
    return int(valid.sum())


def process_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Pairs file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    good = []
    bad = []

    for i, pair in enumerate(pairs):
        v1 = valid_count_after_resize(pair["sample1"])
        v2 = valid_count_after_resize(pair["sample2"])

        if v1 >= MIN_VALID_PIXELS and v2 >= MIN_VALID_PIXELS:
            good.append(pair)
        else:
            bad.append({
                "idx": i,
                "pair_type": pair.get("pair_type"),
                "image1": pair["sample1"]["image_name"],
                "image2": pair["sample2"]["image_name"],
                "valid1": v1,
                "valid2": v2,
            })

    out_good = path.with_name(path.stem + "_filtered.json")
    out_bad = path.with_name(path.stem + "_rejected.json")

    out_good.write_text(json.dumps(good, ensure_ascii=False, indent=2), encoding="utf-8")
    out_bad.write_text(json.dumps(bad, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{path.name}:")
    print(f"  total    = {len(pairs)}")
    print(f"  kept     = {len(good)}")
    print(f"  rejected = {len(bad)}")
    print(f"  saved    = {out_good.name}")
    print(f"  rejected log = {out_bad.name}")


def main():
    for path in FILES:
        process_file(path)


if __name__ == "__main__":
    main()