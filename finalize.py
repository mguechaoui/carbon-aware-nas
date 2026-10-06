"""Retrain selected architectures longer, evaluate on the TEST set, measure, save model + metadata."""
import argparse
import json
import os
import time

import torch

from common import Net, count_macs, evaluate, load_cifar, train
from measure import cpu_latency_ms, gpu_measure, idle_power_w

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=15)
args = ap.parse_args()

xtr, ytr, xte, yte = load_cifar("cuda")
idle = idle_power_w()
selected = json.load(open("results/selected.json"))

for i, r in enumerate(selected):
    a = r["arch"]
    torch.manual_seed(i)
    model = Net(a).cuda()
    macs, params = count_macs(model)
    train(model, xtr, ytr, args.epochs)
    acc = evaluate(model, xte, yte)
    m = gpu_measure(model, idle)
    cpu = cpu_latency_ms(model)
    d = f"models/model_{i:02d}"
    os.makedirs(d, exist_ok=True)
    torch.save(model.state_dict(), f"{d}/model.pt")

    # switching cost: build + load weights onto GPU + one warm-up forward
    torch.cuda.synchronize()
    t = time.perf_counter()
    m2 = Net(a).cuda()
    m2.load_state_dict(torch.load(f"{d}/model.pt"))
    m2.eval()
    with torch.no_grad():
        m2(torch.randn(1, 3, 32, 32, device="cuda"))
    torch.cuda.synchronize()
    switch_ms = (time.perf_counter() - t) * 1000

    meta = dict(arch=a, accuracy=acc, params=params, macs=macs, cpu_latency_ms=cpu, model_load_switch_cost_ms=switch_ms,
                hardware=torch.cuda.get_device_name(0),
                measurement="batch=64, fp16, idle-subtracted NVML power; latency/energy per sample",
                **m)
    json.dump(meta, open(f"{d}/metadata.json", "w"), indent=1)
    print(f"{d}: test acc={acc:.4f} E={m['gpu_energy_mj']:.4f} mJ switch={switch_ms:.1f} ms")
    del model, m2
    torch.cuda.empty_cache()
