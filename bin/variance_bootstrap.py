#!/usr/bin/env python3
"""This script tries to get an estimate of the upper bound of what amount of
variance can be predicted.

The main idea is that the counts are not completely reliable; if you could
remeasure the same tissue twice, the counts wouldn't match exactly. To
approximate this value, for each spot, we can resample the same number of
points, with the same distribution, which I think is an upper bound."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import torch
import numpy as np
import time
import collections
import sys
import scipy
sys.path.append(".")
import stnet
from stnet.utils.util import latexify
import pathlib
pathlib.Path("fig").mkdir(parents=True, exist_ok=True)


patient = sorted(list(set(map(lambda x: x.split("/")[-2], glob.glob("{}/*/*/*.npz".format(stnet.config.SPATIAL_PROCESSED_ROOT))))))

gene_names = stnet.datasets.Spatial(gene_filter=250).gene_names
gene_filter = stnet.datasets.Spatial(gene_filter=250).gene_filter

dataset = {}
loader = {}
for p in patient:
    dataset[p] = stnet.datasets.Spatial(patient=[p], gene_filter=None, gene_transform=None, load_image=False)
    loader[p] = torch.utils.data.DataLoader(dataset[p], batch_size=256, num_workers=8, shuffle=False)

def load(loader):
    gene = []
    tumor = []
    coord = []
    sec = []
    t = time.time()
    for (i, (X, y, g, c, _, _, s, *_)) in enumerate(loader):
        print("{:8d} / {:d}:    {:4.0f} / {:4.0f} seconds".format(i + 1, len(loader), time.time() - t, (time.time() - t) * len(loader) / (i + 1)), end="\r", flush=True)
        gene.append(g)
        tumor.append(y)
        coord.append(c)
        sec.append(s)
    print()
    gene = np.concatenate(gene)
    tumor = np.concatenate(tumor)
    coord = np.concatenate(coord)
    sec = np.concatenate(sec)
    return gene, tumor, coord, sec

def neighbor(x, c, s):
    ans = np.empty(x.shape)
    for i in range(x.shape[0]):
        mask = np.logical_and(s == s[i], np.sum(np.abs(c - c[i, :]), 1) == 1)
        pred = x[mask, :]
        ans[i, :] = np.mean(pred, 0)
    return ans

pearson = np.zeros((len(patient), 250))
for (i, p) in enumerate(patient):
    g, t, c, s = load(loader[p])
    raw = g
    pred = neighbor(raw, c, s)
    # print(pred)
    n = pred.shape[1]
    pred = np.log((1 + pred) / (n + np.sum(pred, 1, keepdims=True)))
    raw = np.log((1 + raw) / (n + np.sum(raw, 1, keepdims=True)))
    for (j, k) in enumerate([x for (x, y) in enumerate(gene_filter) if y]):
        mask = ~np.isnan(pred[:, k])
        # print(raw[:, k])
        # print(pred[:, k])
        coef, _ = scipy.stats.pearsonr(raw[mask, k], pred[mask, k])
        print(coef)
        pearson[i, j] = coef
    # print(pearson)
    # print()

print("Median correlation | # Consistent Patients | Gene Name")
print(sorted(zip(np.median(pearson, 0), np.sum(pearson > 0, 0), gene_names))[::-1])
exit(0)

def bootstrap(x):
    ans = np.empty(x.shape)
    for i in range(x.shape[0]):
        n = int(sum(x[i, :]))
        samples = np.random.choice(x.shape[1], size=n, replace=True, p=x[i, :] / n)
        ans[i, :] = np.bincount(samples, minlength=x.shape[1])
    return ans

pearson = np.zeros((len(patient), 250))
for (i, p) in enumerate(patient):
    g, t, c, s = load(loader[p])
    raw = g
    pred = bootstrap(raw)
    print(pred.shape)
    n = pred.shape[1]
    pred = np.log((1 + pred) / (n + np.sum(pred, 1, keepdims=True)))
    raw = np.log((1 + raw) / (n + np.sum(raw, 1, keepdims=True)))
    for (j, k) in enumerate([x for (x, y) in enumerate(gene_filter) if y]):
        coef, _ = scipy.stats.pearsonr(raw[:, k], pred[:, k])
        pearson[i, j] = coef
    print(pearson)

print("Median correlation | # Consistent Patients | Gene Name")
print(sorted(zip(np.median(pearson, 0), np.sum(pearson > 0, 0), gene_names))[::-1])
