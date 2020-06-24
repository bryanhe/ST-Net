import logging
import torch
import pickle
import pathlib
import numpy as np
import os
import datetime
import distutils
import torchvision
import glob
import collections
import stnet


# https://stackoverflow.com/questions/4623446/how-do-you-sort-files-numerically
def tryint(s):
    try:
        return int(s)
    except:
        return s


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]



# Based on https://nipunbatra.github.io/blog/2014/latexify.html
def latexify():
    import matplotlib
    params = {'backend': 'pdf',
          'axes.titlesize':  8,
          'axes.labelsize':  8,
          'font.size':       8,
          'legend.fontsize': 8,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          # 'text.usetex': True,
          'font.family': 'DejaVu Serif',
          'font.serif': 'Computer Modern',
    }
    matplotlib.rcParams.update(params)


def newer_than(file1, file2):
    """
    Returns True if file1 is newer than file2.
    A typical use case is if file2 is generated using file1.
    For example:

    if newer_than(file1, file2):
        # update file2 based on file1
    """
    return os.path.isfile(file1) and (not os.path.isfile(file2) or os.path.getctime(file1) > os.path.getctime(file2))


# TODO: move functions that are specific to histology to other files?
#       possibly another file for things specific to this project, like the parser?
def contains_tissue(X, color_threshold=200, percentage_threshold=0.6):
    return np.mean(np.mean(X, axis=2).flatten() > color_threshold) < percentage_threshold


def get_spatial_patients():
    """
    Returns a dict of patients to sections.

    The keys of the dict are patient names (str), and the values are lists of
    section names (str).
    """
    patient_section = map(lambda x: x.split("/")[-1].split(".")[0].split("_"), glob.glob(stnet.config.SPATIAL_RAW_ROOT + "/*/*/*.jpg"))
    patient = collections.defaultdict(list)
    for (p, s) in patient_section:
        patient[p].append(s)
    return patient
