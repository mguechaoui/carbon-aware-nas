"""Extract the Pareto front from NAS results, plot it, pick K models spread along it."""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=5)
args = ap.parse_args()

rows = json.load(open("results/nas_trials.json"))
obj = np.array([[1 - r["val_acc"], r["gpu_energy_mj"], r["gpu_latency_ms"]] for r in rows])


def pareto_mask(o):
    m = np.ones(len(o), bool)
    for i in range(len(o)):
        dom = np.all(o <= o[i], axis=1) & np.any(o < o[i], axis=1)
        if dom.any():
            m[i] = False
    return m


mask = pareto_mask(obj)
front = [r for r, k in zip(rows, mask) if k]
print(f"{len(front)} Pareto-optimal of {len(rows)} trials")

plt.figure(figsize=(6, 4))
plt.scatter(obj[~mask, 1], 1 - obj[~mask, 0], c="lightgray", label="dominated")
plt.scatter(obj[mask, 1], 1 - obj[mask, 0], c="tab:green", label="Pareto-optimal")
plt.xlabel("GPU energy per inference (mJ, idle-subtracted)")
plt.ylabel("validation accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("results/pareto.png", dpi=150)

front.sort(key=lambda r: r["gpu_energy_mj"])
idx = sorted(set(np.linspace(0, len(front) - 1, min(args.k, len(front))).round().astype(int)))
json.dump([front[i] for i in idx], open("results/selected.json", "w"), indent=1)
print("selected architectures:")
for i in idx:
    print(" ", front[i]["arch"], f"acc={front[i]['val_acc']:.3f}", f"E={front[i]['gpu_energy_mj']:.4f} mJ")
