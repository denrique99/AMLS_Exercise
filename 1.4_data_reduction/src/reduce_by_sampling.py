import random
import json
import argparse
import pickle
from collections import defaultdict


def stratified_indices(y, ratio, seed=0):
    rng = random.Random(seed)
    by_cl = defaultdict(list)
    for idx, label in enumerate(y):
        by_cl[label].append(idx)

    keep = []
    for cls, idxs in by_cl.items():
        k = max(1, int(len(idxs) * ratio))
        keep.extend(rng.sample(idxs, k))
    rng.shuffle(keep)
    return keep

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio", type=float, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--split_data", type=str, default="../data/split_data.pkl")
    args = parser.parse_args()

    with open(args.split_data, "rb") as f:
        X_train_split, X_val_split, y_train_split, y_val_split = pickle.load(f)

    idxs = stratified_indices(y_train_split, args.ratio)
    json.dump(idxs, open(args.out, "w"))
    print(f"Saved {len(idxs)} indices → {args.out}")
