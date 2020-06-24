#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

"""This module provides statistics about the spatial dataset."""

import argparse
import argcomplete

parser = argparse.ArgumentParser()
parser.add_argument("--figroot", default="fig/", type=str)
parser.add_argument("--workers", default=8, type=int)
parser.add_argument("--batch", default=256, type=int)

argcomplete.autocomplete(parser)
args = parser.parse_args()

import sys
sys.path.append(".")
import stnet
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import os
import pathlib
import tqdm
import glob
import collections
import code

stnet.utils.util.latexify()
pathlib.Path(os.path.dirname(args.figroot)).mkdir(parents=True, exist_ok=True)

patients = stnet.utils.util.get_spatial_patients()
filenames = glob.glob("{}/*/*/*.npz".format(stnet.config.SPATIAL_PROCESSED_ROOT))
ps = [(f.split("/")[-2], f.split("/")[-1].split("_")[0]) for f in filenames]
num_spots = collections.Counter(ps)
num_spots = sorted([sorted(num_spots[(p, s)] for s in patients[p]) for p in patients])
print("Number of spots: ", num_spots)

width = 1
space = 2
fig = plt.figure(figsize=(5.5, 1.5))
for i in range(3):
    if i == 0:
        color = "darkgray"
    elif i == 1:
        color = "dimgray"
    elif i == 2:
        color = "black"
    plt.bar((3 * width + space) * np.array(range(23)) + width * i, [x[i] for x in num_spots], width=width, color=color, bottom=0)

plt.ylim([0, 800])
ax = plt.gca()
ax.set_xticks((3 * width + space) * np.array(range(23)) + 1.5 * width)
ax.set_xticklabels((i + 1 for i in range(23)))
ax.set_yticks([0, 200, 400, 600, 800])
# ax.set_xticklabels(("" for i in range(23)))
plt.xlabel("Patient")
plt.ylabel("# Spots")
plt.tight_layout()
plt.savefig(args.figroot + "num_spots.pdf")
plt.close(fig)

dataset = stnet.datasets.Spatial(gene_filter=None, load_image=False, gene_transform=None)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, num_workers=args.workers, shuffle=True)
spots = []
genes = None
count = []
n = 0
largest = 0
for (X, tumor, y, coord, index, patient, section, pixel, *_) in tqdm.tqdm(dataloader):
    spots.extend(torch.sum(y, 1).tolist())
    if genes is None:
        genes = torch.zeros(y.shape[1])
    genes += torch.sum(y, 0)
    count.append(y.numpy())
    n += y.shape[0]
    largest = max(largest, y.max())
count = np.concatenate(count)
print("largest single gene-spot count: ", largest.item())
print("number of genes: ", y.shape[1])

genes /= n

fig = plt.figure(figsize=(2.0, 1.5))
plt.semilogy(np.array(range(len(spots))) / len(spots), np.array(sorted(spots)), linewidth=1, color="k")
plt.xlim([0, 1])
plt.ylim([1, 10000])
plt.xlabel("Spot (percentile)")
plt.ylabel("Counts per spot")
ax = plt.gca()
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticks([1, 10, 100, 1000, 10000])
plt.tight_layout()
fig.savefig(args.figroot + "counts_per_spot.pdf")
plt.close(fig)

fig = plt.figure(figsize=(2.0, 1.5))
xmin = 100
xmax = 100000
bins = 16
logbins = np.logspace(np.log10(xmin), np.log10(xmax), bins)
plt.hist(spots, bins=logbins)
plt.xscale("log")
plt.xlabel("Counts per spot")
plt.ylabel("# Spots")
ax = plt.gca()
ax.set_xticks([100, 1000, 10000, 100000])
ax.set_yticks([0, 2500, 5000, 7500])
plt.xlim([xmin, xmax])
# plt.ylim([0, 7500])
# ax.set_yticks()
plt.tight_layout()
fig.savefig(args.figroot + "count_histogram.pdf")
plt.close(fig)

fig = plt.figure(figsize=(2.0, 1.5))
xmin = 100
xmax = 10000
bins = 11
logbins = np.logspace(np.log10(xmin), np.log10(xmax), bins)
plt.hist(np.sum(count != 0, 1), bins=logbins)
plt.xscale("log")
plt.xlabel("Distinct genes per spot")
plt.ylabel("# Spots")
ax = plt.gca()
ax.set_xticks([100, 1000, 10000])
ax.set_yticks([0, 2500, 5000, 7500, 10000])
plt.xlim([xmin, xmax])
# plt.ylim([0, 7500])
# ax.set_yticks()
plt.tight_layout()
fig.savefig(args.figroot + "nnz_histogram.pdf")
plt.close(fig)

fig = plt.figure(figsize=(2.0, 1.5))
plt.semilogy(np.array(range(count.shape[0])) / count.shape[0], sorted(np.sum(count != 0, 1)), linewidth=1, color="k")
plt.xlim([0, 1])
plt.ylim([1, 10000])
plt.xlabel("Spot (percentile)")
plt.ylabel("# Distinct Genes")
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticks([1, 10, 100, 1000, 10000])
plt.tight_layout()
fig.savefig(args.figroot + "nnz_per_spot.pdf")
plt.close(fig)

perm = torch.argsort(genes).numpy()[::-1]
# index = [0, 99, 199, 299, 399, 499]
# index = [0, 49, 99, 149, 199, 249]
index = [0, 249, 499]
count = count[:, perm]
gene_names = np.array(dataset.gene_names)[perm]
gene_names = gene_names[index]
gene_names[gene_names == "__ambiguous[ENSG00000185883+ENSG00000260272]"] = "ATP6V0C"

count = count[:, index]
count = np.sort(count, 0)
fig = plt.figure(figsize=(2.0, 1.5))
plt.semilogy(np.array(range(count.shape[0])) / count.shape[0], count[:, 0], linewidth=1, color="black",     label=gene_names[0], zorder=5)
plt.semilogy(np.array(range(count.shape[0])) / count.shape[0], count[:, 1], linewidth=1, color="dimgray",   label=gene_names[1], zorder=4)
plt.semilogy(np.array(range(count.shape[0])) / count.shape[0], count[:, 2], linewidth=1, color="gray",      label=gene_names[2], zorder=3)
# plt.semilogy(np.array(range(count.shape[0])) / count.shape[0], count[:, 3], linewidth=1, color="darkgray",  label=gene_names[3], zorder=2)
# plt.semilogy(np.array(range(count.shape[0])) / count.shape[0], count[:, 4], linewidth=1, color="lightgray", label=gene_names[4], zorder=1)
# plt.semilogy(np.array(range(count.shape[0])) / count.shape[0], count[:, 5], linewidth=1, color="gainsboro", label=gene_names[5], zorder=0)
plt.xlim([0, 1])
plt.ylim([1, 1500])
plt.xlabel("Spot (percentile)")
plt.ylabel("Single gene counts")
ax = plt.gca()
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticks([1, 10, 100, 1000])
plt.legend(loc="best", fontsize=4)
plt.tight_layout()
fig.savefig(args.figroot + "gene_counts_per_spot.pdf")
plt.close(fig)

fig = plt.figure(figsize=(2.5, 2.0))
xmin = 1
xmax = 1000
bins = 16
logbins = np.logspace(np.log10(xmin), np.log10(xmax), bins)
plt.hist([count[:, i] for i in range(count.shape[1])], bins=logbins, label=list(map(lambda x: "{} ({})".format(x[0], x[1] + 1), zip(gene_names, index))))
plt.xscale("log")
plt.xlabel("Total counts")
plt.ylabel("# Spots")
ax = plt.gca()
ax.set_xticks([1, 10, 100, 1000, 10000])
plt.xlim([xmin, xmax])
plt.legend(loc="best", fontsize=6)
# ax.set_yticks()
plt.tight_layout()
fig.savefig(args.figroot + "gene_count_histogram.pdf")
plt.close(fig)

fig = plt.figure(figsize=(2.5, 2.0))
print("highest mean expression of single gene: ", max(genes).item())
plt.plot(range(1, len(genes) + 1), sorted(genes)[::-1], linewidth=1, color="k")
plt.xlim([1, 500])
plt.ylim([0, 32])
ax = plt.gca()
ax.set_xticks([1, 250, 500])
ax.set_yticks([0, 10, 20, 30])
plt.xlabel("Gene")
plt.ylabel("Mean count")
plt.tight_layout()
fig.savefig(args.figroot + "mean_expression_by_gene.pdf")
plt.close(fig)
