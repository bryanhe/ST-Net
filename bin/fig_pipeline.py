#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

import argparse
import argcomplete

parser = argparse.ArgumentParser()
parser.add_argument("patient", nargs="?", default="BC23377", type=str)
parser.add_argument("section", nargs="?", default="C1", type=str)
parser.add_argument("index", nargs="?", default=429, type=int)
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
import PIL

stnet.utils.util.latexify()
pathlib.Path(os.path.dirname(args.figroot)).mkdir(parents=True, exist_ok=True)

dataset = stnet.datasets.Spatial([(args.patient, args.section)], gene_filter=None, load_image=False, gene_transform=None)
image_filename = os.path.join(stnet.config.SPATIAL_RAW_ROOT, dataset.subtype[args.patient], args.patient, "{}_{}.tif".format(args.patient, args.section))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, num_workers=args.workers, shuffle=False)
image = plt.imread(image_filename)
print(image.shape)

fig = plt.figure(figsize=(2, 2))
plt.imshow(image, aspect="equal", interpolation="nearest")
#plt.imshow(image, aspect="equal", interpolation="none")
fig.patch.set_visible(False)
plt.gca().axis("off")
plt.gca().set_aspect("equal")

spots = []
genes = None
n = 0
xmin, xmax, ymin, ymax = float("inf"), -float("inf"), float("inf"), -float("inf")
for (X, tumor, y, coord, index, patient, section, pixel, *_) in tqdm.tqdm(dataloader):
    for p in pixel:
        alpha = 1.0 if n == args.index else 0.5
        color = "red" if n == args.index else "black"
        lw = 0.4 if n == args.index else 0.2
        plt.plot(p[0].item() + 112 * np.array([-1, -1, 1, 1, -1]), p[1].item() + 112 * np.array([-1, 1, 1, -1, -1]), linewidth=lw, color=color, alpha=alpha)
        # plt.gca().annotate(str(n), (p[0].item(), p[1].item()), fontsize=2, ha="center")
        xmin = min(xmin, p[0].item())
        xmax = max(xmax, p[0].item())
        ymin = min(ymin, p[1].item())
        ymax = max(ymax, p[1].item())
        n += 1
margin = 112 + 50
x = xmin + 300
y = ymin + 300
plt.gca().annotate("1 mm", (x + 743, y + 200), fontsize=8, ha="center")
plt.plot([x, x + 1486],        [y, y],             color="black", linewidth=0.3)
plt.plot([x, x],               [y - 100, y + 100], color="black", linewidth=0.3)
plt.plot([x + 1486, x + 1486], [y - 100, y + 100], color="black", linewidth=0.3)
plt.axis([xmin - margin, xmax + margin, ymin - margin, ymax + margin])



plt.xticks([])
plt.yticks([])
# plt.gca().invert_yaxis()
plt.tight_layout()
fig.savefig("{}pipeline_{}_{}.jpg".format(args.figroot, args.patient, args.section), quality=100, dpi=600)
fig.savefig("{}pipeline_{}_{}.pdf".format(args.figroot, args.patient, args.section), quality=100, dpi=600)
plt.close(fig)

dataset.load_image=True
dataset[args.index][0].transpose(PIL.Image.FLIP_TOP_BOTTOM).save(args.figroot + "pipeline_patch.jpg")

dataset = stnet.datasets.Spatial([(args.patient, args.section)], gene_filter=None, load_image=True, gene_transform=None, window=448)
dataset[args.index][0].transpose(PIL.Image.FLIP_TOP_BOTTOM).save(args.figroot + "pipeline_patch_448.jpg")

dataset = stnet.datasets.Spatial([(args.patient, args.section)], gene_filter=None, load_image=True, gene_transform=None, window=512)
dataset[args.index][0].transpose(PIL.Image.FLIP_TOP_BOTTOM).save(args.figroot + "pipeline_patch_512.jpg")
