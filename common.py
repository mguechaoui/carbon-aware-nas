"""Search space, model builder, MAC counter, data + training helpers."""
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASES = [16, 24, 32, 48]
KERNELS = [3, 5]
EXPANDS = [1, 2, 4]
RESOLUTIONS = [24, 32, 40]
DEPTH = (2, 6)


def sample_arch(rng: random.Random):
    return dict(depth=rng.randint(*DEPTH), base=rng.choice(BASES),
                kernel=rng.choice(KERNELS), expand=rng.choice(EXPANDS),
                res=rng.choice(RESOLUTIONS))


def suggest_arch(trial):
    return dict(depth=trial.suggest_int("depth", *DEPTH),
                base=trial.suggest_categorical("base", BASES),
                kernel=trial.suggest_categorical("kernel", KERNELS),
                expand=trial.suggest_categorical("expand", EXPANDS),
                res=trial.suggest_categorical("res", RESOLUTIONS))


class Block(nn.Module):
    """MobileNet-style inverted residual block."""

    def __init__(self, cin, cout, k, e, stride):
        super().__init__()
        mid = cin * e
        layers = []
        if e != 1:
            layers += [nn.Conv2d(cin, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.ReLU6(inplace=True)]
        layers += [nn.Conv2d(mid, mid, k, stride, k // 2, groups=mid, bias=False),
                   nn.BatchNorm2d(mid), nn.ReLU6(inplace=True),
                   nn.Conv2d(mid, cout, 1, bias=False), nn.BatchNorm2d(cout)]
        self.f = nn.Sequential(*layers)
        self.res = stride == 1 and cin == cout

    def forward(self, x):
        y = self.f(x)
        return x + y if self.res else y


class Net(nn.Module):
    def __init__(self, a, num_classes=10):
        super().__init__()
        self.res = a["res"]
        c = a["base"]
        self.stem = nn.Sequential(nn.Conv2d(3, c, 3, 1, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU6(inplace=True))
        blocks = []
        for i in range(a["depth"]):
            stride = 2 if i in (1, 3) else 1
            cout = c * 2 if stride == 2 else c
            blocks.append(Block(c, cout, a["kernel"], a["expand"], stride))
            c = cout
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Conv2d(c, 128, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU6(inplace=True),
                                  nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, num_classes))

    def forward(self, x):
        if x.shape[-1] != self.res:  # resolution is part of the search space
            x = F.interpolate(x, size=self.res, mode="bilinear", align_corners=False)
        return self.head(self.blocks(self.stem(x)))


def count_macs(model, size=32):
    """Returns (multiply-accumulates for one image, parameter count)."""
    total = [0]

    def conv_hook(m, i, o):
        total[0] += o.shape[1] * o.shape[2] * o.shape[3] * (m.in_channels // m.groups) \
                    * m.kernel_size[0] * m.kernel_size[1]

    def lin_hook(m, i, o):
        total[0] += m.in_features * m.out_features

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(lin_hook))
    dev = next(model.parameters()).device
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, size, size, device=dev))
    for h in hooks:
        h.remove()
    model.train(was_training)
    return total[0], sum(p.numel() for p in model.parameters())


def features(a, macs, params):
    """Feature vector for the energy predictor."""
    return [a["depth"], a["base"], a["kernel"], a["expand"], a["res"], math.log(macs), math.log(params)]


# ---------------- data + training (whole CIFAR-10 lives on the GPU) -------------
def load_cifar(device="cuda", root="data"):
    import torchvision
    tr = torchvision.datasets.CIFAR10(root, train=True, download=True)
    te = torchvision.datasets.CIFAR10(root, train=False, download=True)
    f = lambda d: torch.from_numpy(d).permute(0, 3, 1, 2).contiguous().to(device)
    return f(tr.data), torch.tensor(tr.targets).to(device), f(te.data), torch.tensor(te.targets).to(device)


def prep(x):
    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616], device=x.device).view(1, 3, 1, 1)
    return (x.float() / 255 - mean) / std


def train(model, x, y, epochs, bs=256, lr=3e-3):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    steps = epochs * math.ceil(len(x) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    scaler = torch.cuda.amp.GradScaler()
    for _ in range(epochs):
        perm = torch.randperm(len(x), device=x.device)
        for i in range(0, len(x), bs):
            idx = perm[i:i + bs]
            xb = prep(x[idx])
            flip = torch.rand(len(idx), device=x.device) < 0.5
            xb = torch.where(flip[:, None, None, None], xb.flip(3), xb)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = F.cross_entropy(model(xb), y[idx])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()


@torch.no_grad()
def evaluate(model, x, y, bs=1000):
    model.eval()
    correct = 0
    for i in range(0, len(x), bs):
        with torch.autocast("cuda", dtype=torch.float16):
            correct += (model(prep(x[i:i + bs])).argmax(1) == y[i:i + bs]).sum().item()
    return correct / len(x)
