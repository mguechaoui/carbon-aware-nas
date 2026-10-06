"""How reliable is a cheap accuracy proxy? Train random architectures at SHORT and LONG epoch budgets
(fresh model each time, 45k train / 5k val, test set untouched) and compare the rankings."""
import argparse
import json
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import kendalltau, spearmanr

from common import Net, evaluate, load_cifar, sample_arch, train

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=24)
ap.add_argument("--short", type=int, default=2)
ap.add_argument("--long", type=int, default=8)
args = ap.parse_args()

xtr, ytr, _, _ = load_cifar("cuda")
path = "results/proxy_study.json"
rows = json.load(open(path)) if os.path.exists(path) else []  # resumable
done = {tuple(r["arch"].values()) for r in rows}

rng, archs, seen = random.Random(123), [], set()
while len(archs) < args.n:
    a = sample_arch(rng)
    if tuple(a.values()) not in seen:
        seen.add(tuple(a.values()))
        archs.append(a)

for i, a in enumerate(archs):
    if tuple(a.values()) in done:
        continue
    accs = {}
    for name, ep in (("short", args.short), ("long", args.long)):
        torch.manual_seed(i)
        model = Net(a).cuda()
        train(model, xtr[:45000], ytr[:45000], ep)
        accs[name] = evaluate(model, xtr[45000:], ytr[45000:])
        del model
        torch.cuda.empty_cache()
    rows.append(dict(arch=a, acc_short=accs["short"], acc_long=accs["long"]))
    json.dump(rows, open(path, "w"), indent=1)
    print(f"{len(rows)}/{args.n} {a} short={accs['short']:.3f} long={accs['long']:.3f}")

s = np.array([r["acc_short"] for r in rows])
l = np.array([r["acc_long"] for r in rows])
k = min(5, len(rows))
top_long = set(np.argsort(-l)[:k])
top_short = set(np.argsort(-s)[:2 * k])
out = dict(n=len(rows), short_epochs=args.short, long_epochs=args.long,
           spearman=float(spearmanr(s, l)[0]), kendall=float(kendalltau(s, l)[0]),
           top_k_recall=f"{len(top_long & top_short)}/{k} of the best-{k} (long) are within the top-{2 * k} (short)")
json.dump(out, open("results/proxy_study_summary.json", "w"), indent=1)
print(json.dumps(out, indent=1))
plt.figure(figsize=(4.5, 4.5))
plt.scatter(s, l, s=14)
plt.xlabel(f"val accuracy, {args.short} epochs")
plt.ylabel(f"val accuracy, {args.long} epochs")
plt.title(f"Spearman = {out['spearman']:.2f}")
plt.tight_layout()
plt.savefig("results/proxy_study.png", dpi=150)