"""24-hour runtime simulation: policies choose among the model library under a carbon signal + latency budgets.
Carbon signal is SYNTHETIC unless --signal points to a CSV with 24 numbers (gCO2/kWh), one per line."""
import argparse
import csv
import glob
import json
import math

ap = argparse.ArgumentParser()
ap.add_argument("--signal", default=None)
ap.add_argument("--req_per_hour", type=int, default=1_000_000)
ap.add_argument("--max_drop", type=float, default=0.05, help="max accuracy drop allowed vs best feasible model")
args = ap.parse_args()

lib = []
for p in sorted(glob.glob("models/*/metadata.json")):
    m = json.load(open(p))
    m["name"] = p.split("/")[-2]
    lib.append(m)
assert len(lib) >= 2, "need at least 2 models in models/"

if args.signal:
    ci = [float(x) for x in open(args.signal).read().split()][:24]
else:  # synthetic: solar dip around noon, evening peak
    ci = [300 - 150 * math.exp(-((h - 13) ** 2) / 18) + 120 * math.exp(-((h - 19) ** 2) / 6) for h in range(24)]
thr = sorted(ci)[len(ci) // 2]
lat_med = sorted(m["gpu_latency_ms"] for m in lib)[len(lib) // 2]
budget = [lat_med if (8 <= h < 10 or 17 <= h < 19) else float("inf") for h in range(24)]


def feasible(b):
    f = [m for m in lib if m["gpu_latency_ms"] <= b]
    return f or [min(lib, key=lambda m: m["gpu_latency_ms"])]


def pick(policy, h):
    f = feasible(budget[h])
    if policy == "always_largest":
        return max(lib, key=lambda m: m["accuracy"])
    if policy == "always_smallest":
        return min(lib, key=lambda m: m["gpu_energy_mj"])
    if policy == "latency_only":
        return max(f, key=lambda m: m["accuracy"])
    best = max(m["accuracy"] for m in f)
    ok = [m for m in f if m["accuracy"] >= best - args.max_drop]
    if policy == "energy_aware":  # ignores the carbon signal
        return min(ok, key=lambda m: m["gpu_energy_mj"])
    # carbon_aware_heuristic: hand-designed threshold rule, not an optimal controller
    return min(ok, key=lambda m: m["gpu_energy_mj"]) if ci[h] >= thr else max(ok, key=lambda m: m["accuracy"])


rows = []
for pol in ["always_largest", "always_smallest", "latency_only", "energy_aware", "carbon_aware_heuristic"]:
    energy_j = carbon_g = acc = viol = down = 0.0
    switches, prev = 0, None
    for h in range(24):
        m = pick(pol, h)
        R = args.req_per_hour
        e = R * m["gpu_energy_mj"] / 1000
        energy_j += e
        carbon_g += e / 3.6e6 * ci[h]
        acc += m["accuracy"] * R
        viol += R if m["gpu_latency_ms"] > budget[h] else 0
        if prev is not None and prev != m["name"]:
            switches += 1
            down += m["model_load_switch_cost_ms"]
        prev = m["name"]
    n = 24 * args.req_per_hour
    rows.append(dict(policy=pol, mean_accuracy=acc / n, energy_J=energy_j, carbon_mg=carbon_g * 1000,
                     latency_violation_pct=100 * viol / n, switches=switches, switch_downtime_ms=down))

base = rows[0]["carbon_mg"]
for r in rows:
    r["carbon_vs_largest_pct"] = 100 * (r["carbon_mg"] / base - 1)
with open("results/runtime_sim.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
for r in rows:
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
print("NOTE: accuracy values are test accuracies of each model; carbon uses", "provided" if args.signal else "SYNTHETIC", "signal.")
