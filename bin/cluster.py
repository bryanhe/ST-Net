#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

import argparse
import argcomplete

parser = argparse.ArgumentParser()
parser.add_argument("patient", nargs="*", default=None, type=str)
parser.add_argument("--figroot", default="fig/cluster/", type=str)
parser.add_argument("--batch", type=int, default=128, help="batch size")
parser.add_argument("--workers", type=int, default=4, help="number of workers for dataloader")


argcomplete.autocomplete(parser)
args = parser.parse_args()

import sys
sys.path.append(".")
import stnet
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import os
import pathlib
import scipy.stats
import scipy
import sklearn.manifold
import torch
import torchvision
import tqdm
import umap

device = "cuda"

stnet.utils.util.latexify()
pathlib.Path(os.path.dirname(args.figroot)).mkdir(parents=True, exist_ok=True)
fig = plt.figure(figsize=(1.00, 0.50))
cmap_bin = plt.get_cmap("binary")
plt.scatter([-1], [-1], label="Tumor",  s=25, linewidth=0.2, edgecolors="k")
plt.scatter([-1], [-1], label="Normal", s=25, linewidth=0.2, edgecolors="k")
plt.axis([0, 1, 0, 1])
plt.gca().axis("off")
plt.gca().set_aspect("equal")
plt.xticks([])
plt.yticks([])
plt.legend(loc="center")
plt.tight_layout()
fig.savefig(args.figroot + "tumor_legend.pdf")
plt.close(fig)

if args.patient == []:
    args.patient = list(stnet.utils.util.get_spatial_patients().keys())

mean = torch.tensor([0.54, 0.51, 0.68])
std = torch.tensor([0.25, 0.21, 0.16])

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(mean=mean, std=std)])

gene_names = stnet.datasets.Spatial(gene_filter=250).gene_names
gene_filter = stnet.datasets.Spatial(gene_filter=250).gene_filter

try:
    data = np.load(os.path.join("cache", "hidden.npz"))
    P = data["P"]
    Y = data["Y"]
    H = data["H"]
    G = data["G"]
    coord = data["C"]
    S = data["S"]
    N = data["N"]
except FileNotFoundError:
    P = []
    Y = []
    H = []
    G = []
    coord = []
    S = []
    N = []

    for (i, p) in enumerate(args.patient):
        print("{} / {}: {}".format(i + 1, len(args.patient), p))
    
        model = torchvision.models.densenet121(num_classes=250)
        model = torch.nn.DataParallel(model)
        model.to(device)
        model.load_state_dict(torch.load(os.path.join("output", "densenet121_224", "top_250_ft_13", "{}_model.pt".format(p)))["model"])
    
        model.eval()
        torch.set_grad_enabled(False)
    
        model.module.classifier = stnet.utils.nn.Identity()
    
        dataset = stnet.datasets.Spatial([p], transform=transform, gene_filter=None, gene_transform=None)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, num_workers=args.workers, shuffle=False, pin_memory=True)
    
        for (X, y, gene, c, _, _, s, *_) in tqdm.tqdm(dataloader):
            X = X.to(device)
            hidden = model(X).to("cpu").detach().numpy()
            P.extend(hidden.shape[0] * [i])
            H.append(hidden)
            Y.append(y.numpy())
            G.append(gene[:, np.nonzero(gene_filter)].numpy())
            coord.extend(c.numpy())
            S.extend(s)
            N.append(np.sum(gene.numpy(), 1))
    S = np.array(S)
    coord = np.array(coord)
    pathlib.Path("cache").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(os.path.join("cache", "hidden.npz"), P=P, Y=Y, H=H, G=G, N=N, C=coord, S=S)

H = np.concatenate(H)
Y = np.squeeze(np.concatenate(Y))
P = np.array(P)
G = np.concatenate(G)
N = np.concatenate(N)

pearson_close = np.zeros((len(args.patient), 250))
pearson_far = np.zeros((len(args.patient), 250))
for (i, p) in tqdm.tqdm(enumerate(args.patient)):
    mask = (P == i)
    h = H[mask, :]
    g = G[mask, 0, :]
    s = S[mask]
    c = coord[mask, :]
    total = N[mask]
    n = sum(P == i)
    real = []
    pred_close = []
    pred_far = []
    t = []

    for j in range(n):
        mask = np.logical_and(s == s[j], np.sum(np.abs(c - c[j, :]), 1) == 1)
        hj = h[mask, :]
        gj = g[mask, :]

        d = np.sum((hj - h[j, :]) ** 2, 1)
        if d.shape != (0, ):
            real.append(g[j, :].reshape(1, -1))
            pred_close.append(gj[np.argmin(d), :].reshape(1, -1))
            pred_far.append(gj[np.argmax(d), :].reshape(1, -1))
            t.append(total[j])
    real = np.concatenate(real)
    pred_close = np.concatenate(pred_close)
    pred_far = np.concatenate(pred_far)
    t = np.array(t)
    real = np.log((1 + real) / (26949 + t.reshape(-1, 1)))
    pred_close = np.log((1 + pred_close) / (26949 + t.reshape(-1, 1)))
    pred_far = np.log((1 + pred_far) / (26949 + t.reshape(-1, 1)))
    for j in range(250):
        pearson_close[i, j], _ = scipy.stats.pearsonr(real[:, j], pred_close[:, j])
        pearson_far[i, j], _ = scipy.stats.pearsonr(real[:, j], pred_far[:, j])
    print("close")
    print(sorted(np.median(pearson_close[:(i + 1), :], 0)))
    print("far")
    print(sorted(np.median(pearson_far[:(i + 1), :], 0)))
    print()

print("Median correlation | # Consistent Patients | Gene Name")
print(sorted(zip(np.median(pearson_close, 0), np.sum(pearson_close > 0, 0), gene_names))[::-1])
print(sorted(zip(np.median(pearson_far, 0), np.sum(pearson_far > 0, 0), gene_names))[::-1])

fig = plt.figure(figsize=(2.5, 2.5))
plt.plot(range(250), sorted(np.median(pearson_close, 0)), label="Similar")
plt.plot(range(250), sorted(np.median(pearson_far, 0)), label="Distinct")
plt.legend(loc="best")
plt.xlabel("Gene")
plt.ylabel("Median Correlation ")
plt.tight_layout()
plt.savefig(os.path.join(args.figroot, "latent.pdf"))
plt.close(fig)

d = []
rand = []
for (i, p) in tqdm.tqdm(enumerate(args.patient)):
    mask = (P == i)
    h = H[mask, :]
    c = coord[mask, :]
    n = sum(P == i)

    for j in range(n):
        hj = np.concatenate((h[:j, :], h[(j + 1):, :]))

        dist = np.sum((hj - h[j, :]) ** 2, 1)
        d.append(np.linalg.norm(c[np.argmin(dist), :] - c[j, :]))
        rand.append(np.linalg.norm(c[np.random.randint(n), :] - c[j, :]))

fig = plt.figure(figsize=(2.5, 2.5))
plt.hist(d, density=True, bins=50, cumulative=True)
plt.xlabel("Distance (# of spots)")
plt.ylabel("Fraction of Patches")
axis = list(plt.axis())
print(axis)
plt.plot([np.mean(rand)] * 2, [axis[2], axis[3]], linewidth=1, color="k")
print(np.mean(rand))
plt.axis(axis)
plt.tight_layout()
plt.savefig(os.path.join(args.figroot, "distance.pdf"))
plt.close(fig)

pearson = np.zeros((len(args.patient), 250))
C = np.array([matplotlib.colors.get_named_colors_mapping()["tab:blue"] if y == 1 else matplotlib.colors.get_named_colors_mapping()["tab:orange"] for y in Y])
for (i, p) in tqdm.tqdm(enumerate(args.patient)):
    # if p != "BC23269" and p != "BC23272" and p != "BC23810":
    #     continue
    fig = plt.figure(figsize=(1.5, 1.5))
    mask = (P == i)

    try:
        embed = np.load(os.path.join("cache", "embed_{}.npz".format(p)))["embed"]
    except FileNotFoundError:
        embed = umap.UMAP(n_components=2).fit_transform(H[mask, :])
        np.savez_compressed(os.path.join("cache", "embed_{}.npz".format(p)), embed=embed)

    g = G[mask, 0, :]
    total = N[mask]
    y = Y[mask]
    n = embed.shape[0]

    # pred = []
    # for j in range(n):
    #     # d = np.sum((np.concatenate((embed[:j, :], embed[(j + 1):, :])) - embed[j, :]) ** 2, 1)
    #     d = np.sum((np.concatenate((H[mask, :][:j, :], H[mask, :][(j + 1):, :])) - H[mask, :][j, :]) ** 2, 1)
    #     d = sorted(zip(d, list(range(j)) + list(range(j + 1, n))))
    #     d, ind = zip(*d)
    #     pred.append(np.mean(g[ind[:4], :], 0).reshape(1, -1))
    # pred = np.concatenate(pred)
    # pred = np.log((1 + pred) / (26949 + total.reshape(-1, 1)))
    # g = np.log((1 + g) / (26949 + total.reshape(-1, 1)))
    # for j in range(250):
    #     coef, _ = scipy.stats.pearsonr(g[:, j], pred[:, j])
    #     pearson[i, j] = coef
    # print(pearson)
    # print(sorted(np.median(pearson[:(i + 1), :], 0)))
    # for j in range(n):


    # isolate_tumor  = (-float("inf"), None)
    # isolate_normal = (-float("inf"), None)
    # central_tumor  = (-float("inf"), None)
    # central_normal = (-float("inf"), None)
    # for j in range(n):
    #     # d = np.sum((np.concatenate((embed[:j, :], embed[(j + 1):, :])) - embed[j, :]) ** 2, 1)
    #     d = np.sum((np.concatenate((embed[:j, :], embed[(j + 1):, :])) - embed[j, :]) ** 2, 1)
    #     try:
    #         closest_same = min(d[y[j] == np.concatenate((y[:j], y[(j + 1):]))])
    #         if y[j] == 0:
    #             isolate_normal = max(isolate_normal, (closest_same, j))
    #         else:
    #             isolate_tumor = max(isolate_tumor, (closest_same, j))
    #     except ValueError:
    #         pass
    #     try:
    #         closest_diff = min(d[y[j] != np.concatenate((y[:j], y[(j + 1):]))])
    #         if y[j] == 0:
    #             central_normal = max(central_normal, (closest_diff, j))
    #         else:
    #             central_tumor = max(central_tumor, (closest_diff, j))
    #     except ValueError:
    #         pass

    isolate_tumor  = ( float("inf"), None)
    isolate_normal = ( float("inf"), None)
    central_tumor  = (-float("inf"), None)
    central_normal = (-float("inf"), None)
    for j in range(n):
        # d = np.sum((np.concatenate((embed[:j, :], embed[(j + 1):, :])) - embed[j, :]) ** 2, 1)
        d = np.sqrt(np.sum((np.concatenate((embed[:j, :], embed[(j + 1):, :])) - embed[j, :]) ** 2, 1))
        nearby = np.concatenate((y[:j], y[(j + 1):]))[d < 1.5]
        frac_same = np.mean(y[j] == nearby)
        if y[j] == 0:
            isolate_normal = min(isolate_normal, (frac_same, j))
            central_normal = max(central_normal, (frac_same, j))
        else:
            isolate_tumor = min(isolate_tumor, (frac_same, j))
            central_tumor = max(central_tumor, (frac_same, j))

    perm = np.random.permutation(sum(mask))
    plt.scatter(embed[:, 0][perm], embed[:, 1][perm], c=list(C[mask][perm]), s=2, linewidth=0, edgecolors="none")
    dataset = stnet.datasets.Spatial([p], gene_filter=None, gene_transform=None)
    text = []
    if central_tumor[1] is not None:
        plt.scatter(embed[central_tumor[1], 0], embed[central_tumor[1], 1], c="tab:blue", s=6, linewidth=0.5, edgecolors="k")
        if p == "BC23272":
            text.append(plt.text(embed[central_tumor[1], 0] + 0.1, embed[central_tumor[1], 1], "a", horizontalalignment="center", verticalalignment="center"))
        elif p == "BC23810":
            text.append(plt.text(embed[central_tumor[1], 0], embed[central_tumor[1], 1] - 0.5, "a", horizontalalignment="center", verticalalignment="center"))
        else:
            text.append(plt.text(embed[central_tumor[1], 0], embed[central_tumor[1], 1], "a", horizontalalignment="center", verticalalignment="center"))
        dataset[central_tumor[1]][0].save(os.path.join(args.figroot, "{}_ct.jpg").format(p))
    if isolate_tumor[1] is not None:
        plt.scatter(embed[isolate_tumor[1], 0], embed[isolate_tumor[1], 1], c="tab:blue", s=6, linewidth=0.5, edgecolors="k")
        text.append(plt.text(embed[isolate_tumor[1], 0], embed[isolate_tumor[1], 1], "b", horizontalalignment="center", verticalalignment="center"))
        dataset[isolate_tumor[1]][0].save(os.path.join(args.figroot, "{}_it.jpg").format(p))
    if central_normal[1] is not None:
        plt.scatter(embed[central_normal[1], 0], embed[central_normal[1], 1], c="tab:orange", s=6, linewidth=0.5, edgecolors="k")
        text.append(plt.text(embed[central_normal[1], 0], embed[central_normal[1], 1], "c", horizontalalignment="center", verticalalignment="center"))
        dataset[central_normal[1]][0].save(os.path.join(args.figroot, "{}_cn.jpg").format(p))
    if isolate_normal[1] is not None:
        plt.scatter(embed[isolate_normal[1], 0], embed[isolate_normal[1], 1], c="tab:orange", s=6, linewidth=0.5, edgecolors="k")
        text.append(plt.text(embed[isolate_normal[1], 0], embed[isolate_normal[1], 1], "d", horizontalalignment="center", verticalalignment="center"))
        dataset[isolate_normal[1]][0].save(os.path.join(args.figroot, "{}_in.jpg").format(p))

    xmin, xmax, ymin, ymax = plt.axis()
    print()
    print(xmin, xmax, ymin, ymax)
    margin = 1.5
    for t in text:
        print(t.set_position)
        xmin = min(xmin, t.get_position()[0] - margin)
        xmax = max(xmax, t.get_position()[0] + margin)
        ymin = min(ymin, t.get_position()[1] - margin)
        ymax = max(ymax, t.get_position()[1] + margin)

        x = np.array(t.get_position())
        c = np.array(((xmin + xmax) / 2, (ymin + ymax) / 2))
        d = x - c
        d /= np.linalg.norm(d)
        x += 0.5 * d
        # t.set_position(list(x))
    plt.axis((xmin, xmax, ymin, ymax))
    print(xmin, xmax, ymin, ymax)

    import adjustText
    print(adjustText.adjust_text(text, embed[:, 0], embed[:, 1], force_text=(0.01, 0.01), force_points=(0.01, 0.01), avoid_self=False, lim=5000))

    plt.xticks([], [])
    plt.yticks([], [])
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.tight_layout()
    plt.savefig(os.path.join(args.figroot, "{}.pdf".format(p)))
    plt.close(fig)
