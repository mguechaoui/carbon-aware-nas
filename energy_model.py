"""Measure random (untrained) architectures and fit an energy predictor. Latency/energy do not depend on weights."""
import argparse
import json
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--skip_measure", action="store_true")
args = ap.parse_args()

if not args.skip_measure:
    from common import BASES, DEPTH, EXPANDS, KERNELS, RESOLUTIONS, Net, count_macs, sample_arch
    total = (DEPTH[1] - DEPTH[0] + 1) * len(BASES) * len(KERNELS) * len(EXPANDS) * len(RESOLUTIONS)
    if args.n > total:
        print(f"search space has only {total} architectures; capping --n")
        args.n = total
    from measure import gpu_measure, idle_power_w
    idle = idle_power_w()
    rng, seen, rows = random.Random(0), set(), []
    while len(rows) < args.n:
        a = sample_arch(rng)
        key = tuple(a.values())
        if key in seen:
            continue
        seen.add(key)
        model = Net(a).cuda()
        macs, params = count_macs(model)
        m = gpu_measure(model, idle, seconds=2.0, reps=2)
        rows.append(dict(arch=a, macs=macs, params=params, **m))
        if len(rows) % 20 == 0:
            print(f"{len(rows)}/{args.n}")
            json.dump(rows, open("results/energy_dataset.json", "w"))
    json.dump(rows, open("results/energy_dataset.json", "w"))

from common import features
rows = json.load(open("results/energy_dataset.json"))
X = np.array([features(r["arch"], r["macs"], r["params"]) for r in rows])
y = np.array([r["gpu_energy_mj"] for r in rows])
mac_only = np.array([[r["macs"]] for r in rows])
idx = np.arange(len(rows))
tr, te = train_test_split(idx, test_size=0.2, random_state=0)  # held-out architectures


def report(name, pred):
    mae = mean_absolute_error(y[te], pred)
    return dict(model=name, MAE_mj=mae, RMSE_mj=float(np.sqrt(mean_squared_error(y[te], pred))),
                NMAE=mae / float(np.mean(y[te])), MAPE=mean_absolute_percentage_error(y[te], pred),
                R2=r2_score(y[te], pred))


lin = LinearRegression().fit(mac_only[tr], y[tr])
gb = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=0).fit(X[tr], y[tr])
lin_all = LinearRegression().fit(X[tr], y[tr])
results = [report("linear(MACs only)", lin.predict(mac_only[te])),
           report("linear(all features)", lin_all.predict(X[te])), report("gradient boosting", gb.predict(X[te]))]
noise = float(np.median([r["energy_spread"] for r in rows]))
# learning curve: error vs number of measured architectures (same held-out test set)
curve = []
for size in [s for s in (20, 50, 100, 200, 300, 500) if s <= len(tr)] + ([len(tr)] if len(tr) not in (20, 50, 100, 200, 300, 500) else []):
    g = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=0).fit(X[tr[:size]], y[tr[:size]])
    curve.append(dict(n_train=int(size), MAPE=mean_absolute_percentage_error(y[te], g.predict(X[te]))))
out = dict(learning_curve=curve, results=results, median_measurement_spread=noise, n_train=len(tr), n_test=len(te))
json.dump(out, open("results/energy_model.json", "w"), indent=1)
print(json.dumps(out, indent=1))

plt.figure(figsize=(4.5, 4.5))
plt.scatter(y[te], gb.predict(X[te]), s=12)
lim = [0, max(y[te].max(), gb.predict(X[te]).max())]
plt.plot(lim, lim, "k--", lw=1)
plt.xlabel("measured energy (mJ)")
plt.ylabel("predicted energy (mJ)")
plt.tight_layout()
plt.savefig("results/energy_pred.png", dpi=150)
plt.figure(figsize=(4.5, 3.5))
plt.plot([c["n_train"] for c in curve], [100 * c["MAPE"] for c in curve], "o-")
plt.xlabel("measured architectures used for training")
plt.ylabel("MAPE on held-out architectures (%)")
plt.tight_layout()
plt.savefig("results/energy_learning_curve.png", dpi=150)
