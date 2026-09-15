# Author: Tatiana Fedorova

# This script splits pairs.json into training, validation and small overfit subsets.

from pathlib import Path
import json
import random
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, help="Path to COLMAP dataset directory")
args = parser.parse_args()

SRC = Path(args.dataset) / "pairs.json"
OUT_DIR = SRC.parent

OVERFIT_N = 40
VAL_RATIO = 0.1
SEED = 42

with open(SRC, "r", encoding="utf-8") as f:
    pairs = json.load(f)

random.Random(SEED).shuffle(pairs)

pairs_overfit = pairs[:OVERFIT_N]

n_val = int(len(pairs) * VAL_RATIO)
pairs_val = pairs[:n_val]
pairs_train = pairs[n_val:]

(OUT_DIR / "pairs_overfit.json").write_text(
    json.dumps(pairs_overfit, ensure_ascii=False, indent=2), encoding="utf-8"
)
(OUT_DIR / "pairs_train.json").write_text(
    json.dumps(pairs_train, ensure_ascii=False, indent=2), encoding="utf-8"
)
(OUT_DIR / "pairs_val.json").write_text(
    json.dumps(pairs_val, ensure_ascii=False, indent=2), encoding="utf-8"
)

print("total:", len(pairs))
print("overfit:", len(pairs_overfit))
print("train:", len(pairs_train))
print("val:", len(pairs_val))