# WARNING: parser must be called very early on for argcomplete.
# argcomplete evaluates the package until the parser is constructed before it
# can generate completions. Because of this, the parser must be constructed
# before the full package is imported to behave in a usable way. Note that
# running
# > python3 -m stnet
# will actually import the entire package (along with dependencies like
# pytorch, numpy, and pandas), before running __main__.py, which takes
# about 0.5-1 seconds.
# See Performance section of https://argcomplete.readthedocs.io/en/latest/

from .parser import parser
parser()

import pandas as _

from stnet.__version__ import __version__
from stnet.config import config
from stnet.main import main

import stnet.datasets as datasets
import stnet.cmd as cmd
import stnet.utils as utils
import stnet.transforms as transforms
