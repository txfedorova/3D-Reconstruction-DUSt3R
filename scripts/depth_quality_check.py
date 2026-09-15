# Author: Tatiana Fedorova

# This script checks COLMAP depth maps, computes simple quality statistics
# and saves visual previews for manual inspection.

import json
from pathlib import Path
import argparse
import numpy as np
from PIL import Image, ImageDraw


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, help="Path to COLMAP dataset directory")
args = parser.parse_args()

ROOT = Path(args.dataset)
SAMPLES_JSON = ROOT / "samples.json"

OUT_DIR = ROOT / "depth_quality"
PREVIEW_DIR = OUT_DIR / "previews"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)


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
        return arr.astype(np.float32)


def depth_to_vis(depth: np.ndarray):
    valid = np.isfinite(depth) & (depth > 0)
    out = np.zeros_like(depth, dtype=np.uint8)

    if valid.sum() == 0:
        return Image.fromarray(out, mode="L")

    d = depth.copy()
    dmin = np.percentile(d[valid], 2)
    dmax = np.percentile(d[valid], 98)
    d = np.clip(d, dmin, dmax)
    d = (d - dmin) / (dmax - dmin + 1e-8)
    d[~valid] = 0.0
    out = (d * 255).astype(np.uint8)
    return Image.fromarray(out, mode="L")


def compute_metrics(depth: np.ndarray):
    valid = np.isfinite(depth) & (depth > 0)
    total = depth.size
    valid_count = int(valid.sum())
    valid_ratio = float(valid_count / total)

    if valid_count == 0:
        return {
            "valid_count": 0,
            "valid_ratio": 0.0,
            "depth_p05": None,
            "depth_p95": None,
            "depth_range": None,
            "col_diff_mean": None,
            "row_diff_mean": None,
            "banding_ratio": None,
            "score": 0.0,
        }

    vals = depth[valid]
    p05 = float(np.percentile(vals, 5))
    p95 = float(np.percentile(vals, 95))
    depth_range = float(p95 - p05)

    d = depth.copy()
    d[~valid] = np.nan

    col_diff = np.abs(np.diff(d, axis=1))
    row_diff = np.abs(np.diff(d, axis=0))

    col_diff_mean = float(np.nanmean(col_diff))
    row_diff_mean = float(np.nanmean(row_diff))
    banding_ratio = float(col_diff_mean / (row_diff_mean + 1e-8))

    score = (
        2.0 * valid_ratio
        + 0.2 * min(depth_range, 50.0) / 50.0
        + 0.3 * (1.0 / (1.0 + max(0.0, banding_ratio - 1.0)))
    )

    return {
        "valid_count": valid_count,
        "valid_ratio": valid_ratio,
        "depth_p05": p05,
        "depth_p95": p95,
        "depth_range": depth_range,
        "col_diff_mean": col_diff_mean,
        "row_diff_mean": row_diff_mean,
        "banding_ratio": banding_ratio,
        "score": float(score),
    }


def make_preview(depth: np.ndarray, text: str, out_path: Path):
    vis = depth_to_vis(depth).convert("RGB").resize((320, 180), Image.NEAREST)
    draw = ImageDraw.Draw(vis)
    draw.rectangle((0, 0, 320, 32), fill=(255, 255, 255))
    draw.text((5, 5), text, fill=(0, 0, 0))
    vis.save(out_path)


def main():
    if not SAMPLES_JSON.exists():
        raise FileNotFoundError(f"samples.json not found: {SAMPLES_JSON}")

    samples = json.loads(SAMPLES_JSON.read_text(encoding="utf-8"))
    results = []

    for i, sample in enumerate(samples):
        depth_path = Path(sample["depth_path"])
        image_name = sample["image_name"]

        depth = read_colmap_array(depth_path)
        metrics = compute_metrics(depth)

        result = {
            "idx": i,
            "image_name": image_name,
            "depth_path": str(depth_path),
            **metrics,
        }
        results.append(result)

        preview_name = f"{i:04d}_score{metrics['score']:.3f}_valid{metrics['valid_ratio']:.3f}.png"
        text = f"{i:04d} score={metrics['score']:.3f} valid={metrics['valid_ratio']:.3f} band={metrics['banding_ratio']}"
        make_preview(depth, text, PREVIEW_DIR / preview_name)

        if (i + 1) % 50 == 0:
            print(f"{i+1}/{len(samples)} processed")

    results_sorted = sorted(results, key=lambda x: x["score"], reverse=True)

    (OUT_DIR / "depth_quality.json").write_text(
        json.dumps(results_sorted, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    csv_lines = [
        "idx,image_name,valid_count,valid_ratio,depth_p05,depth_p95,depth_range,col_diff_mean,row_diff_mean,banding_ratio,score"
    ]
    for r in results_sorted:
        csv_lines.append(
            f"{r['idx']},\"{r['image_name']}\",{r['valid_count']},{r['valid_ratio']},"
            f"{r['depth_p05']},{r['depth_p95']},{r['depth_range']},"
            f"{r['col_diff_mean']},{r['row_diff_mean']},{r['banding_ratio']},{r['score']}"
        )

    (OUT_DIR / "depth_quality.csv").write_text("\n".join(csv_lines), encoding="utf-8")

    print("Saved:")
    print(OUT_DIR / "depth_quality.json")
    print(OUT_DIR / "depth_quality.csv")
    print(PREVIEW_DIR)


if __name__ == "__main__":
    main()