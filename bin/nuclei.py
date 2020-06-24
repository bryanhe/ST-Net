#!/usr/bin/env python3

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import torch
import torchvision
import numpy as np
import time
import collections
import sys
import scipy
sys.path.append(".")
import stnet
from stnet.utils.util import latexify
import tqdm
import pathlib
pathlib.Path("fig").mkdir(parents=True, exist_ok=True)

mean = np.array([0.54, 0.51, 0.68])
std = np.array([0.25, 0.21, 0.16])
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(mean=mean, std=std)])
dataset = stnet.datasets.Spatial(transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=8, shuffle=True)

normal = []
tumor = []
for (X, t, *_) in tqdm.tqdm(loader):
    f = stnet.models.feature.features(X)
    normal += list(f[:, 6][t[:, 0] == 0].numpy())
    tumor += list(f[:, 6][t[:, 0] == 1].numpy())
    # print(sorted(normal)[len(normal) // 2])
    # print(sorted(tumor)[len(tumor) // 2])

    # if min(len(normal), len(tumor)) > 10:
    #     break

latexify()
fig = plt.figure(figsize=(3, 3))
bins = int(max(max(normal), max(tumor)))
print(bins)
plt.hist((normal, tumor), bins=range(bins + 1), density=True)
plt.xlabel("# Nuclei")
plt.ylabel("Fraction of Patches")
plt.title("")
plt.legend(["Normal", "Tumor"], loc="best")
plt.tight_layout()
plt.savefig("fig/nuclei.pdf")
plt.close(fig)
