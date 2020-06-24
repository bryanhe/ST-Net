#!/usr/bin/env python3

import argparse
import argcomplete

parser = argparse.ArgumentParser("Cross validation for experiments.")

parser.add_argument("root", type=str)
parser.add_argument("folds", type=int)
parser.add_argument("epochs", type=int)
parser.add_argument("testpatients", nargs="+", type=str)

argcomplete.autocomplete(parser)
args, unknown = parser.parse_known_args()

import os
import sys
sys.path.append(".")
import stnet
import pathlib
import logging
import numpy as np
import scipy.stats
import shutil

# TODO: logging at this level gets really mangled (inner calls to logging config overwrite this)
# simplest option is to just not log here (very little meaningful stuff anyways)
logfile = os.path.join(args.root + "cv", "cv.log")

pathlib.Path(os.path.dirname(logfile)).mkdir(parents=True, exist_ok=True)

loglevel = logging.DEBUG
cfg = dict(
      version=1,
      formatters={
          "f": {"()":
                    "stnet.utils.logging.MultilineFormatter",
                "format":
                    "%(levelname)-8s [%(asctime)s] %(message)s",
                "datefmt":
                    "%m/%d %H:%M:%S"}
          },
      handlers={
          "s": {"class": "logging.StreamHandler",
                "formatter": "f",
                "level": loglevel},
          "f": {"class": "logging.FileHandler",
                "formatter": "f",
                # "level": logging.DEBUG,
                "level": loglevel,
                "filename": logfile}
          },
      root={
          "handlers": ["s", "f"],
          "level": logging.NOTSET
          },
      disable_existing_loggers=False,
  )
logging.config.dictConfig(cfg)

logger = logging.getLogger(__name__)
logger.info(args)
logger.info(unknown)

pathlib.Path(os.path.dirname(args.root)).mkdir(parents=True, exist_ok=True)
patients = sorted(stnet.utils.util.get_spatial_patients().keys())
folds_to_run = args.folds

for p in args.testpatients:
    assert(p in patients)
    patients.remove(p)

fold = [patients[f::args.folds] for f in range(args.folds)]

def cross_validate(root, extra_args=[]):
    for f in range(folds_to_run):
        logger.info("Fold #{}".format(f))
    
        train = [fold[i] for i in range(args.folds) if i != f]
        train = [i for j in train for i in j]
        test = fold[f]
    
        if os.path.isfile(root + "{}_{}.npz".format(f, args.epochs)):
            logger.info("Already completed by previous run.")
        else:
            stnet.main(["run_spatial", "--gene"] +
                       ["--logfile", root + "{}.log".format(f)] +
                       ["--epochs", str(args.epochs)] +
                       ["--checkpoint", os.path.join(root + "{}_checkpoints".format(f), "epoch_")] +
                       ["--pred_root", root + "{}_".format(f)] +
                       ["--trainpatients"] + train +
                       ["--testpatients"] + test +
                       unknown + extra_args
            )
        try:
            shutil.rmtree(os.path.join(root + "{}_checkpoints".format(f)))
        except FileNotFoundError:
            pass

    best_loss = float("inf")
    best_epoch = 0
    for epoch in range(args.epochs):
        try:
            patient = []
            pred = []
            count = []
            mean = []
            for f in range(folds_to_run):
                data = np.load(os.path.join(root + "{}_{}.npz".format(f, epoch + 1)))
                pred.append(data["predictions"])
                count.append(data["counts"])
                mean.append(data["mean_expression"])
                patient.append(data["patient"])
    
            mean = np.concatenate([np.repeat(np.expand_dims(m, 0), c.shape[0], 0) for (m, c) in zip(mean, count)], 0)
            pred = np.concatenate(pred)
            count = np.concatenate(count)
            patient = np.concatenate(patient)

            loss = np.sum((pred - count) ** 2)

            if loss < best_loss:
                best_loss = loss
                best_epoch = epoch + 1
        except FileNotFoundError as e:  # TODO: epochs sometimes missing
                                   # this happens when a job crashes during test time
                                   # checkpoint is already saved, but prediction file doesn't exist yet
            # pass
            raise e
    
    logger.info("Best loss: {}".format(best_loss))
    logger.info("Best epoch: {}".format(best_epoch))

    return best_loss, best_epoch

logger.info("Searching for epochs")
_, best_epoch = cross_validate(os.path.join(args.root + "cv", "cv_"))

try:
    os.symlink("{}{}.npz".format(os.path.basename(args.root), best_epoch), "{}cv.npz".format(args.root))
    os.symlink(os.path.join(os.path.basename(args.root) + "checkpoints", "epoch_{}.pt".format(best_epoch)), "{}model.pt".format(args.root))
except FileExistsError:
    pass

if os.path.isfile("{}{}.npz".format(args.root, args.epochs)):
    logger.info("Final run already completed.")
else:
    stnet.main(["run_spatial", "--gene"] +
               ["--logfile", (args.root + "gene.log")] +
               ["--epochs", str(args.epochs)] +
               ["--checkpoint", os.path.join(args.root + "checkpoints", "epoch_")] +
               # ["--save_pred_every", str(best_epoch)] +
               ["--pred_root", args.root] +
               ["--trainpatients"] + patients +
               ["--testpatients"] + args.testpatients +
               ["--keep_checkpoints", str(best_epoch)] +
               unknown
    )
