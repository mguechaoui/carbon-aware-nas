"""GPU latency + energy measurement through NVML, CPU latency through wall-clock.
Run `python measure.py --check` FIRST to verify that power readings work in your setup."""
import argparse
import copy
import statistics
import threading
import time

import torch

import pynvml

_h = None


def _handle():
    global _h
    if _h is None:
        pynvml.nvmlInit()
        _h = pynvml.nvmlDeviceGetHandleByIndex(0)
    return _h


def read_power_w():
    return pynvml.nvmlDeviceGetPowerUsage(_handle()) / 1000.0


class Sampler:
    def __init__(self, dt=0.02):
        self.dt, self.s, self.stop = dt, [], threading.Event()

    def _run(self):
        while not self.stop.is_set():
            self.s.append(read_power_w())
            time.sleep(self.dt)

    def __enter__(self):
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()
        return self

    def __exit__(self, *a):
        self.stop.set()
        self.t.join()

    def mean(self):
        return statistics.fmean(self.s)

    def std(self):
        return statistics.pstdev(self.s)


def idle_power_w(seconds=3.0):
    torch.cuda.synchronize()
    time.sleep(1.0)
    with Sampler() as s:
        time.sleep(seconds)
    return s.mean()


@torch.no_grad()
def gpu_measure(model, idle_w, bs=64, seconds=3.0, reps=3):
    """Per-sample latency (ms) and idle-subtracted energy (mJ). Batch bs, fp16 autocast."""
    model = model.cuda().eval()
    x = torch.randn(bs, 3, 32, 32, device="cuda")
    runs = []
    with torch.autocast("cuda", dtype=torch.float16):
        for _ in range(30):
            model(x)
        torch.cuda.synchronize()
        for _ in range(reps):
            n = 0
            with Sampler() as s:
                t0 = time.perf_counter()
                while time.perf_counter() - t0 < seconds:
                    for _ in range(10):
                        model(x)
                    n += 10
                torch.cuda.synchronize()
                dt = time.perf_counter() - t0
            p = s.mean()
            runs.append(dict(lat=dt / n / bs * 1000, p=p, sd=s.std(), e=max(p - idle_w, 0.0) * dt / (n * bs) * 1000))
    med = lambda k: statistics.median(r[k] for r in runs)
    es = [r["e"] for r in runs]
    spread = (max(es) - min(es)) / max(med("e"), 1e-9)
    return dict(gpu_latency_ms=med("lat"), gpu_power_w=med("p"), gpu_power_std_w=med("sd"), gpu_energy_mj=med("e"), energy_spread=spread, gpu_idle_power_w=idle_w)


@torch.no_grad()
def cpu_latency_ms(model, runs=50):
    m = copy.deepcopy(model).cpu().eval()
    x = torch.randn(1, 3, 32, 32)
    for _ in range(10):
        m(x)
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        m(x)
        ts.append((time.perf_counter() - t) * 1000)
    return statistics.median(ts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.parse_args()
    try:
        print("power now (W):", read_power_w())
        print("idle power (W):", idle_power_w())
        print("OK: NVML power readings work.")
    except Exception as e:  # noqa
        print("NVML power readout FAILED:", repr(e))
        print("See PLAN.md > Troubleshooting (run from native Windows Python instead of WSL).")
