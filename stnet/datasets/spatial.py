from functools import partial

import os
import os.path as osp

from typing import Any, Callable, Optional, NamedTuple, Type

import yaml

import numpy as np

import pandas as pd

from PIL.Image import open as read_image

import torch as t
from torch.utils.data import DataLoader, Dataset

from torchvision import transforms

import stnet

import torch

import PIL
import torchvision
import glob
import openslide
import pickle
import logging
import pathlib
import statistics
import collections

# TODO: make this a visiondataset
class Spatial(torch.utils.data.Dataset):
    def __init__(self,
                 patient=None,
                 transform=None,
                 window=224,
                 cache=False,
                 root=stnet.config.SPATIAL_PROCESSED_ROOT,
                 gene_filter="tumor",
                 load_image=True,
                 gene_transform="log",
                 downsample=1,
                 norm=None,
                 feature=False):

        self.dataset = sorted(glob.glob("{}/*/*/*.npz".format(root)))
        if patient is not None:
            # Can specify (patient, section) or only patient (take all sections)
            self.dataset = [d for d in self.dataset if ((d.split("/")[-2] in patient) or ((d.split("/")[-2], d.split("/")[-1].split("_")[0]) in patient))]

            # TODO: if patient == [], then many things downstream bug out
            # how to handle this case?
            # Could just throw an error

        # TODO: filter things that are too close to edge?

        self.transform = transform
        self.window = window
        self.downsample = downsample
        self.cache = cache
        self.root = root
        self.load_image = load_image
        self.gene_transform = gene_transform
        self.norm = norm
        self.feature = feature

        with open(root + "/subtype.pkl", "rb") as f:
            self.subtype = pickle.load(f)
        with open(root + "/gene.pkl", "rb") as f:
            self.ensg_names = pickle.load(f)

        self.gene_names = list(map(lambda x: stnet.utils.ensembl.symbol[x], self.ensg_names))
        self.mean_expression = np.load(stnet.config.SPATIAL_PROCESSED_ROOT + "/mean_expression.npy")
        self.median_expression = np.load(stnet.config.SPATIAL_PROCESSED_ROOT + "/median_expression.npy")

        self.slide = collections.defaultdict(dict)
        # TODO: this can be parallelized
        for (patient, section) in set([(d.split("/")[-2], d.split("/")[-1].split("_")[0]) for d in self.dataset]):
            self.slide[patient][section] = openslide.open_slide("{}/{}/{}/{}_{}.tif".format(stnet.config.SPATIAL_RAW_ROOT, self.subtype[patient], patient, patient, section))

        if gene_filter is None or gene_filter == "none":
            self.gene_filter = None
        elif gene_filter == "high":
            self.gene_filter = np.array([m > 1. for m in self.mean_expression])
        elif gene_filter == "tumor":
            # These are the 10 genes with the largest difference in expression between tumor and normal tissue
            # Printed by save_counts.py
            self.tumor_genes = ["PABPC1", "GNAS", "HSP90AB1", "TFF3", "ATP1A1", "COX6C", "B2M", "FASN", "ACTG1", "HLA-B"]
            self.gene_filter = np.array([g in self.tumor_genes for g in self.gene_names])
        elif isinstance(gene_filter, list):
            self.gene_filter = np.array([g in gene_filter or e in gene_filter for (g, e) in zip(self.gene_names, self.ensg_names)])
        elif isinstance(gene_filter, int):
            keep = set(list(zip(*sorted(zip(self.mean_expression, range(self.mean_expression.shape[0])))[::-1][:gene_filter]))[1])
            self.gene_filter = np.array([i in keep for i in range(len(self.gene_names))])
        else:
            raise ValueError()

        if self.gene_filter is not None:
            self.ensg_names = [n for (n, f) in zip(self.ensg_names, self.gene_filter) if f]
            self.gene_names = [n for (n, f) in zip(self.gene_names, self.gene_filter) if f]
            self.mean_expression = self.mean_expression[self.gene_filter]
            self.median_expression = self.median_expression[self.gene_filter]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        npz = np.load(self.dataset[index])

        count   = npz["count"]
        tumor   = npz["tumor"]
        pixel   = npz["pixel"]
        patient = npz["patient"][0]
        section = npz["section"][0]
        coord   = npz["index"]

        feature_filename = os.path.splitext(self.dataset[index])[0] + "_feature.npy"
        need_image = self.feature and not os.path.isfile(feature_filename)

        if self.load_image or need_image:
            orig_window = self.window * self.downsample
            cached_image = "{}/{}/{}/{}_{}_{}_{}_{}_{}.tif".format(self.root, self.subtype[patient], patient, patient, section, orig_window, self.downsample, coord[0], coord[1])
            if self.cache and pathlib.Path(cached_image).exists():
                X = PIL.Image.open(cached_image)
            else:
                slide = self.slide[patient][section]
                X = slide.read_region((pixel[0] - orig_window // 2, pixel[1] - orig_window // 2), 0, (orig_window, orig_window))
                X = X.convert("RGB")

                if self.cache:
                    X.save(cached_image)

                # TODO: check downsample
                if self.downsample != 1:
                    X = torchvision.transforms.Resize((self.window, self.window))(X)

            if self.transform is not None:
                X = self.transform(X)

        if self.feature:
            try:
                f = np.load(feature_filename)
            except FileNotFoundError:
                asd
                f = stnet.util.histology.features(X).numpy()
                np.save(feature_filename, f)
            f = torch.Tensor(f[0, :])
        else:
            f = []

        if not self.load_image:
            X = []

        Z = np.sum(count)
        n = count.shape[0]
        if self.gene_filter is not None:
            count = count[self.gene_filter]
        y = torch.as_tensor(count, dtype=torch.float)

        tumor = torch.as_tensor([1 if tumor else 0])
        coord = torch.as_tensor(coord)
        index = torch.as_tensor([index])

        if self.norm is None or self.norm == "none":
            if self.gene_transform == "log":
                y = torch.log(1 + y)
            elif self.gene_transform == "sqrt":
                y = torch.sqrt(y)
            else:
                assert(self.gene_transform is None or self.gene_transform == "none")
        elif self.norm == "norm":
            if self.gene_transform == "log":
                y = torch.log((1 + y) / (n + Z))
            else:
                assert(self.gene_transform is None or self.gene_transform == "none")
                y = y / Z
        elif self.norm == "normfilter":
            if self.gene_transform == "log":
                y = torch.log((1 + y) / torch.sum(1 + y))
            else:
                assert(self.gene_transform is None or self.gene_transform == "none")
                y = y / torch.sum(y)
        elif self.norm == "normpat":
            if self.gene_transform == "log":
                y = torch.log((1 + y) / self.p_median[patient])
            else:
                assert(self.gene_transform is None or self.gene_transform == "none")
                y = y / self.ps_median[(patient, section)]
        elif self.norm == "normsec":
            if self.gene_transform == "log":
                y = torch.log((1 + y) / self.ps_median[(patient, section)])
            else:
                assert(self.gene_transform is None or self.gene_transform == "none")
                y = y / self.ps_median[(patient, section)]
        else:
            raise ValueError()

        return X, tumor, y, coord, index, patient, section, pixel, f
