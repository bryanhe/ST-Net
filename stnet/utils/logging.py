""" logging.py
"""

import logging
import logging.config
import pathlib
import datetime
import os

import sys
from logging import DEBUG, INFO, WARNING, ERROR


# Based on https://mail.python.org/pipermail/python-list/2010-November/591474.html
class MultilineFormatter(logging.Formatter):
    def format(self, record):
        str = logging.Formatter.format(self, record)
        header, footer = str.split(record.message)
        str = str.replace('\n', '\n' + ' '*len(header))
        return str

def setup_logging(logfile=None, loglevel=logging.DEBUG):
    if logfile is None:
        logfile = "log/" + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')

    pathlib.Path(os.path.dirname(logfile)).mkdir(parents=True, exist_ok=True)

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
