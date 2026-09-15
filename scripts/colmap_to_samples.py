# Author: Tatiana Fedorova

# This script converts COLMAP dense reconstruction outputs to samples.json.
# It saves image paths, depth map paths, camera intrinsics and camera poses
# for all images that have corresponding COLMAP depth maps.

from pathlib import Path
import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, help="Path to COLMAP dataset directory")
args = parser.parse_args()

DATASET = Path(args.dataset)
DENSE = DATASET / "dense" / "0"
SPARSE_TXT = DENSE / "sparse_txt"
IMAGES_DIR = DENSE / "images"
DEPTH_DIR = DENSE / "stereo" / "depth_maps"
OUT_PATH = DATASET / "samples.json"

def qvec_to_rotmat(qvec):
    qw, qx, qy, qz = qvec
    return [
        [
            1 - 2 * qy * qy - 2 * qz * qz,
            2 * qx * qy - 2 * qw * qz,
            2 * qx * qz + 2 * qw * qy,
        ],
        [
            2 * qx * qy + 2 * qw * qz,
            1 - 2 * qx * qx - 2 * qz * qz,
            2 * qy * qz - 2 * qw * qx,
        ],
        [
            2 * qx * qz - 2 * qw * qy,
            2 * qy * qz + 2 * qw * qx,
            1 - 2 * qx * qx - 2 * qy * qy,
        ],
    ]


def mat4_identity():
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def invert_pose(Tcw):
    R = [row[:3] for row in Tcw[:3]]
    t = [Tcw[0][3], Tcw[1][3], Tcw[2][3]]

    Rt = [
        [R[0][0], R[1][0], R[2][0]],
        [R[0][1], R[1][1], R[2][1]],
        [R[0][2], R[1][2], R[2][2]],
    ]

    twc = [
        -(Rt[0][0] * t[0] + Rt[0][1] * t[1] + Rt[0][2] * t[2]),
        -(Rt[1][0] * t[0] + Rt[1][1] * t[1] + Rt[1][2] * t[2]),
        -(Rt[2][0] * t[0] + Rt[2][1] * t[1] + Rt[2][2] * t[2]),
    ]

    Twc = mat4_identity()
    for i in range(3):
        for j in range(3):
            Twc[i][j] = Rt[i][j]
    Twc[0][3], Twc[1][3], Twc[2][3] = twc
    return Twc


def parse_cameras(path: Path):
    cameras = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = list(map(float, parts[4:]))

            if model == "PINHOLE":
                fx, fy, cx, cy = params
            elif model == "SIMPLE_PINHOLE":
                f, cx, cy = params
                fx, fy = f, f
            elif model == "SIMPLE_RADIAL":
                f, cx, cy, _k = params[:4]
                fx, fy = f, f
            elif model == "RADIAL":
                f, cx, cy, _k1, _k2 = params[:5]
                fx, fy = f, f
            elif model == "OPENCV":
                fx, fy, cx, cy = params[:4]
            elif model == "OPENCV_FISHEYE":
                fx, fy, cx, cy = params[:4]
            else:
                raise ValueError(f"Unsupported camera model: {model}")

            K = [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ]

            cameras[cam_id] = {
                "model": model,
                "width": width,
                "height": height,
                "params": params,
                "K": K,
            }
    return cameras


def parse_images(path: Path):
    images = []
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue

        parts = line.split()
        image_id = int(parts[0])
        qvec = list(map(float, parts[1:5]))
        tvec = list(map(float, parts[5:8]))
        camera_id = int(parts[8])
        name = parts[9]

        R = qvec_to_rotmat(qvec)
        Tcw = mat4_identity()
        for r in range(3):
            for c in range(3):
                Tcw[r][c] = R[r][c]
        Tcw[0][3], Tcw[1][3], Tcw[2][3] = tvec

        images.append({
            "image_id": image_id,
            "camera_id": camera_id,
            "name": name,
            "T_cw": Tcw,
            "T_wc": invert_pose(Tcw),
        })

        i += 2
    return images


def main():
    cameras = parse_cameras(SPARSE_TXT / "cameras.txt")
    images = parse_images(SPARSE_TXT / "images.txt")

    samples = []
    missing_image = 0
    missing_depth = 0

    for img in images:
        rel = Path(img["name"])
        image_path = IMAGES_DIR / rel
        depth_path = DEPTH_DIR / (str(rel) + ".photometric.bin")

        if not image_path.exists():
            missing_image += 1
            continue
        if not depth_path.exists():
            missing_depth += 1
            continue

        cam = cameras[img["camera_id"]]
        samples.append({
            "image_id": img["image_id"],
            "image_name": img["name"],
            "image_path": str(image_path),
            "depth_path": str(depth_path),
            "camera_id": img["camera_id"],
            "camera_model": cam["model"],
            "width": cam["width"],
            "height": cam["height"],
            "K": cam["K"],
            "T_cw": img["T_cw"],
            "T_wc": img["T_wc"],
        })

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"Total images in images.txt: {len(images)}")
    print(f"Valid samples: {len(samples)}")
    print(f"Missing image files: {missing_image}")
    print(f"Missing depth files: {missing_depth}")
    print(f"Saved to: {OUT_PATH}")


if __name__ == "__main__":
    main()
