#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle
import glob
import math
import scipy.stats
import scipy.special
import sklearn.metrics
import sys
sys.path.append(".")
import stnet
from stnet.utils.util import latexify
import pathlib
import argparse
import code
import statsmodels.stats.multitest
import code
import collections

### Config ###
parser = argparse.ArgumentParser(
    "Generates plots for gene prediction."
)
parser.add_argument("root")
parser.add_argument("epoch")
args = parser.parse_args()

root = args.root
epoch = args.epoch

### Loading results ###
def load(root, epoch):
    patients = []
    pred = []
    count = []
    expression = []
    pathologist = []
    tumor = []
    gene_names = None
    ensg_names = None
    for filename in sorted(glob.glob("{}/???????_{}.npz".format(root, epoch))):
        patient = filename.split("/")[-1].split("_")[0]
        if patient in patients:
            continue
        patients.append(patient)
        # print(filename)
        try:
            data = np.load(filename)
            gene_names = data["gene_names"] # TODO: check consistent
            ensg_names = data["ensg_names"]
            # print(list(data.keys()))
            p = data["predictions"]
            c = data["counts"]
            if p.shape[0] == c.shape[0] and p.shape[1] != c.shape[1]:
                gene_names = ["count"]
                # For total count prediction
                c = np.sum(c, 1, keepdims=True)
            if p.shape != c.shape:
                # For binary prediction
                p = p.reshape(p.shape[0], -1, 2)
                p = p[:, :, 1] - p[:, :, 0]
            pred.append(p)
            count.append(c)

            mean_tumor = data["mean_expression_tumor"]
            mean_normal = data["mean_expression_normal"]
            tumor.append(data["tumor"])
            expression.append(data["mean_expression"])
            pathologist.append(tumor[-1] * mean_tumor + (1 - tumor[-1]) * mean_normal)
        except FileNotFoundError:
            # Symlink for cross-validation may point to a file that is not generated yet; just skip until finished
            pass
    return patients, pred, count, expression, pathologist, tumor, gene_names, ensg_names

patients, pred, count, expression, pathologist, tumor, gene_names, ensg_names = load(root, epoch)

print("Number of patients: {}".format(len(count)))

if count == []:
    print("No patients found.")
    exit(0)

### Basic setup for figures ###
pathlib.Path("fig").mkdir(parents=True, exist_ok=True)
latexify()

### Computing Pearson Correlations ###
number = [0 for i in range(len(count) + 1)]
number_pathologist = [0 for i in range(len(count) + 1)]
number_tumor = [0 for i in range(len(count) + 1)]
number_normal = [0 for i in range(len(count) + 1)]
pearson = np.zeros((len(count), count[0].shape[1]))
pearsonp = np.zeros((len(count), count[0].shape[1]))
pearson_pathologist = np.zeros((len(count), count[0].shape[1]))
pearson_tumor = np.zeros((len(count), count[0].shape[1]))
pearson_normal = np.zeros((len(count), count[0].shape[1]))
std = np.zeros((len(count), count[0].shape[1]))
print(len(count))
print(count[0].shape)
for i in range(len(count)):
    for j in range(count[i].shape[1]):
        coef, p = scipy.stats.pearsonr(count[i][:, j], pred[i][:, j])
        pearson[i, j] = coef
        pearsonp[i, j] = p / 2 if coef >= 0 else 1 - p / 2
        std[i, j] = np.std(count[i][:, j])
        pearson_pathologist[i, j], _ = scipy.stats.pearsonr(count[i][:, j], pathologist[i][:, j])

        pearson_tumor[i, j], _ = scipy.stats.pearsonr(count[i][np.squeeze(tumor[i] == 1, 1), j], pred[i][np.squeeze(tumor[i] == 1, 1), j])
        if any(tumor[i] == 0):
            pearson_normal[i, j], _ = scipy.stats.pearsonr(count[i][np.squeeze(tumor[i] == 0, 1), j], pred[i][np.squeeze(tumor[i] == 0, 1), j])
        else:
            pearson_normal[i, j] = float("nan")


### Patient-level correlations ###
x = []
for j in range(count[0].shape[1]):
    coef, p = scipy.stats.pearsonr([count[i][:, j].mean() for i in range(len(count))], [pred[i][:, j].mean() for i in range(len(count))])
    x.append(coef)
print(list(zip(np.median(pearson, 0), np.sum(pearson > 0, 0), gene_names)))

# Consistent patients vs gene figure (Fig 2a)
fig = plt.figure(figsize=(2.5, 2.5))
plt.scatter(range(1, pearson.shape[1] + 1), sorted(x)[::-1], s=1, color="k", zorder=100)
plt.xlabel("Gene")
plt.ylabel("Correlation")
plt.tight_layout()
plt.savefig("fig/patient_correlation.pdf")
plt.close(fig)


### Statistical testing for main results ###
def pvalue(k, n=len(patients)):
    ans = 0
    for i in range(k, n + 1):
        ans += scipy.special.comb(n, i)
    return ans / (2 ** n)

alpha = 0.1
reject, pvals_corrected, *_ = statsmodels.stats.multitest.multipletests(list(pvalue(sum(pearson[:, i] > 0)) for i in range(pearson.shape[1])), method="holm", alpha=alpha)
print("CNN:")
print("    Genes:    ", sum(reject))
if sum(reject) != 0:
    print("    FDR:      ", max(pvals_corrected[reject]))
    print("    Patients: ", sorted(sum(pearson[:, i] > 0) for i in range(pearson.shape[1]))[::-1][sum(reject) - 1])

feature = [16, 21, 18, 17, 18, 19, 16, 18, 14, 16, 17, 17, 17, 17, 15, 20, 22, 12, 13, 19, 22, 21, 16, 17, 21, 14, 19, 18, 12, 17, 18, 15, 17, 14, 12, 17, 23, 20, 16, 13, 15, 17, 18, 22, 19, 18, 17, 15, 15, 18, 17, 13, 13, 17, 18, 17, 19, 16, 15, 18, 15, 20, 18, 19, 17, 18, 19, 16, 13, 21, 13, 20, 18, 18, 17, 18, 15, 21, 18, 14, 16, 18, 18, 15, 18, 17, 14, 21, 19, 17, 19, 16, 12, 16, 17, 15, 18, 15, 17, 17, 18, 15, 15, 18, 17, 15, 22, 11, 17, 15, 17, 17, 15, 16, 16, 17, 19, 16, 15, 21, 16, 18, 16, 17, 13, 17, 16, 19, 15, 11, 20, 16, 16, 17, 20, 17, 19, 17, 14, 20, 20, 18, 19, 18, 14, 19, 20, 19, 19, 14, 18, 16, 17, 14, 17, 18, 18, 17, 17, 18, 18, 18, 17, 16, 15, 16, 16, 18, 20, 15, 16, 15, 16, 18, 16, 20, 13, 18, 17, 16, 19, 14, 8, 15, 14, 19, 17, 16, 21, 15, 18, 14, 19, 18, 10, 17, 14, 15, 17, 22, 11, 15, 17, 20, 18, 21, 16, 14, 15, 17, 17, 15, 20, 16, 16, 13, 17, 14, 18, 15, 16, 17, 16, 18, 20, 14, 21, 15, 17, 17, 7, 22, 16, 22, 16, 19, 14, 17, 20, 18, 18, 13, 17, 20, 16, 15, 17, 17, 14, 17]
print(list(zip(gene_names, x)))
# Consistent patients vs gene figure (Fig 2a)
fig = plt.figure(figsize=(1.93, 1.93))
plt.scatter(range(1, pearson.shape[1] + 1), sorted(sum(pearson[:, i] > 0) for i in range(pearson.shape[1]))[::-1], s=1, zorder=100)
plt.scatter(range(1, pearson_pathologist.shape[1] + 1), sorted(sum(np.logical_or(pearson_pathologist[:, i] > 0, np.isnan(pearson_pathologist[:, i]))) for i in range(pearson_pathologist.shape[1]))[::-1], s=1, zorder=50)
plt.scatter(range(1, 250 + 1), sorted(feature)[::-1], s=1, zorder=50)
ngenes = pearson.shape[1]
plt.scatter(range(1, pearson.shape[1] + 1), scipy.stats.binom.ppf((np.array(range(ngenes)) + 0.5)[::-1] / ngenes, 23, 0.5), s=1)
plt.legend(["ST-Net", "Tumor/Normal", "Feature", "Null"], prop={"size": 6})
plt.xlabel("Gene")
plt.ylabel("# Consistent Patients")
plt.tight_layout()
plt.ylim([0, 23.5])
plt.yticks([0, 5, 10, 15, 20, 23])
plt.savefig("fig/number.pdf")
plt.close(fig)

### Statistical testing for pathologist-based baseline ###
alpha = 0.4
reject, pvals_corrected, *_ = statsmodels.stats.multitest.multipletests(list(pvalue(sum(pearson_pathologist[:, i] > 0)) for i in range(pearson_pathologist.shape[1])), method="holm", alpha=alpha)
print("Pathologist Baseline:")
print("    Genes:    ", sum(reject))
if sum(reject) != 0:
    print("    FDR:      ", max(pvals_corrected[reject]))
    print("    Patients: ", sorted(sum(pearson_pathologist[:, i] > 0) for i in range(pearson_pathologist.shape[1]))[::-1][sum(reject) - 1])
print(pearson_pathologist.shape)
p_p = pearson_pathologist[~np.any(np.isnan(pearson_pathologist), 1), :]
print(list(zip(np.median(p_p, 0), np.sum(p_p > 0, 0), gene_names)))
asd

### Statistical testing for intra-tumor and intra-normal ###

alpha = 0.1
reject, pvals_corrected, *_ = statsmodels.stats.multitest.multipletests(list(pvalue(sum(pearson_tumor[:, i] > 0)) for i in range(pearson_tumor.shape[1])), method="holm", alpha=alpha)
print("Tumor:")
print("    Genes:    ", sum(reject))
if sum(reject) != 0:
    print("    FDR:      ", max(pvals_corrected[reject]))
    print("    Patients: ", sorted(sum(pearson_tumor[:, i] > 0) for i in range(pearson_tumor.shape[1]))[::-1][sum(reject) - 1])

alpha = 0.4
reject, pvals_corrected, *_ = statsmodels.stats.multitest.multipletests(list(pvalue(sum(pearson_normal[:, i] > 0)) for i in range(pearson_normal.shape[1])), method="holm", alpha=alpha)
print("Normal:")
print("    Genes:    ", sum(reject))
if sum(reject) != 0:
    print("    FDR:      ", max(pvals_corrected[reject]))
    print("    Patients: ", sorted(sum(pearson_normal[:, i] > 0) for i in range(pearson_normal.shape[1]))[::-1][sum(reject) - 1])

### Statistical testing for oracle ###
# TODO
reject, pvals_corrected, *_ = statsmodels.stats.multitest.multipletests(list(pvalue(sum(pearson_pathologist[:, i] > 0)) for i in range(pearson_pathologist.shape[1])), method="holm", alpha=alpha)
print("Oracle:")
print("    Genes:    ", sum(reject))
if sum(reject) != 0:
    print("    FDR:      ", max(pvals_corrected[reject]))
    print("    Patients: ", sorted(sum(pearson_pathologist[:, i] > 0) for i in range(pearson_pathologist.shape[1]))[::-1][sum(reject) - 1])

### List of genes ###
reject, pvals_corrected, *_ = statsmodels.stats.multitest.multipletests(list(pvalue(sum(pearson[:, i] > 0)) for i in range(pearson.shape[1])), method="holm", alpha=alpha)
print()
print("Gene list for DAVID")
print(ensg_names[reject])


### Genes by performance ###
print()
print("Median correlation | # Consistent Patients | Gene Name")
print(sorted(zip(np.median(pearson, 0), np.sum(pearson > 0, 0), gene_names))[::-1])
print(list(zip(np.median(pearson, 0), np.sum(pearson > 0, 0), gene_names)))


### Performance on FASN (used to select slides for Figure 2b) ###
print()
print("FASN Correlation | Patient | Rank")
fasn = list(zip(sorted(zip(np.squeeze(pearson[:, gene_names == "FASN"]), patients)), range(len(patients), 0, -1)))
print(fasn[::-1])
