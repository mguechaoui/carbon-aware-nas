"""Multi-objective search (NSGA-II) over accuracy, GPU energy and GPU latency."""
import argparse
import json
import os

import optuna
import torch

from common import Net, count_macs, evaluate, load_cifar, suggest_arch, train
from measure import cpu_latency_ms, gpu_measure, idle_power_w

ap = argparse.ArgumentParser()
ap.add_argument("--trials", type=int, default=40)
ap.add_argument("--epochs", type=int, default=3)
args = ap.parse_args()

os.makedirs("results", exist_ok=True)
xtr, ytr, _, _ = load_cifar("cuda")  # test set is NOT used during search
idle = idle_power_w()
print(f"idle GPU power: {idle:.2f} W")


def objective(trial):
    a = suggest_arch(trial)
    torch.manual_seed(trial.number)
    model = Net(a).cuda()
    macs, params = count_macs(model)
    train(model, xtr[:45000], ytr[:45000], args.epochs)
    acc = evaluate(model, xtr[45000:], ytr[45000:])
    m = gpu_measure(model, idle)
    cpu = cpu_latency_ms(model)
    for k, v in dict(arch=a, macs=macs, params=params, val_acc=acc, cpu_latency_ms=cpu, **m).items():
        trial.set_user_attr(k, v)
    print(f"trial {trial.number}: {a} acc={acc:.3f} E={m['gpu_energy_mj']:.4f} mJ lat={m['gpu_latency_ms']:.4f} ms")
    del model
    torch.cuda.empty_cache()
    return 1 - acc, m["gpu_energy_mj"], m["gpu_latency_ms"]


def dump(study, trial=None):
    rows = [{**t.user_attrs, "trial": t.number} for t in study.trials if t.state.name == "COMPLETE"]
    json.dump(rows, open("results/nas_trials.json", "w"), indent=1)
    return rows


study = optuna.create_study(
    directions=["minimize"] * 3,
    sampler=optuna.samplers.NSGAIISampler(population_size=10, seed=0),
    storage="sqlite:///results/nas.db", study_name="nas", load_if_exists=True)
study.optimize(objective, n_trials=args.trials, callbacks=[dump])

rows = dump(study)
print(f"saved {len(rows)} trials to results/nas_trials.json")