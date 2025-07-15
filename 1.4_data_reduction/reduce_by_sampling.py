import random, json, argparse
from collections import defaultdict
from dataset import ECGDataset         # same parser as before

def stratified_indices(dataset, ratio, seed=0):
    rng   = random.Random(seed)
    by_cl = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        by_cl[label].append(idx)

    keep = []
    for cls, idxs in by_cl.items():
        k = max(1, int(len(idxs)*ratio))
        keep.extend(rng.sample(idxs, k))
    rng.shuffle(keep)
    return keep

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio", type=float, required=True)
    parser.add_argument("--out",   type=str,   required=True)
    args   = parser.parse_args()

    full = ECGDataset(split="train_full")
    idxs = stratified_indices(full, args.ratio)
    json.dump(idxs, open(args.out, "w"))
    print(f"Saved {len(idxs)} indices → {args.out}")
