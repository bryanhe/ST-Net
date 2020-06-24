def parser():
    """Returns parser for stnet."""

    import argparse
    import argcomplete
    from datetime import datetime as dt
    from . import __version__
    from .config import config

    parser = argparse.ArgumentParser(
        description='\n'.join([
            "Histonet",
            "--------",
            "Version: {}".format(__version__),
        ]),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '-v',
        '--version',
        action='version',
        version=__version__,
    )
    parser.add_argument(
        '--seed',
        '-s',
        type=int,
        default=0,
        help='RNG seed',
    )

    subparsers = parser.add_subparsers()

    spatial_parser = subparsers.add_parser("run_spatial", help="train model")

    add_model_arguments(spatial_parser)

    def patient_or_section(name):
        if "_" in name:
            return tuple(name.split("_"))
        return name

    spatial_parser.add_argument("--test", type=float, default=0.25, help="fraction of data as test set")
    spatial_parser.add_argument("--testpatients", nargs="*", type=patient_or_section, default=None,
                                   help="list of data points as test set"
                                        "(--test is ignored if this is set)")
    spatial_parser.add_argument("--trainpatients", nargs="*", type=patient_or_section, default=None,
                                   help="list of data points as train set"
                                        "(defaults to all patients not in test set if not set)")
    add_logging_arguments(spatial_parser)
    add_window_arguments(spatial_parser)
    add_device_arguments(spatial_parser)
    add_training_arguments(spatial_parser)
    add_augmentation_arguments(spatial_parser)

    group = spatial_parser.add_mutually_exclusive_group()
    group.add_argument("--tumor", action="store_const", const="tumor", dest="task", default="tumor",
                       help="Tumor prediction")
    group.add_argument("--gene", action="store_const", const="gene", dest="task", help="Gene count prediction")
    group.add_argument("--count", action="store_const", const="count", dest="task", help="Total count prediction")
    group.add_argument("--geneb", action="store_const", const="geneb", dest="task",
                       help="Gene count binary prediction (high/low)")

    add_gene_filter_arguments(spatial_parser)

    # This is used to allow the cross validation version to still have the right dims
    def binary_str(x):
        if not isinstance(x, str):
            raise TypeError()
        for i in x:
            if i != "0" and i != "1":
                raise ValueError()
        print([int(i) for i in x])
        return [int(i) for i in x]

    spatial_parser.add_argument("--gene_mask", type=binary_str, default=None,
                                   help="binary string with a length matching number of genes")
    spatial_parser.add_argument("--gene_transform", dest="gene_transform", choices=["none", "log"], default="log",
                                   help="transform for gene count")

    add_normalization_arguments(spatial_parser)

    spatial_parser.add_argument("--finetune", type=int, nargs="?", const=1, default=None,
                                   help="fine tune last n layers")
    spatial_parser.add_argument("--randomize", action="store_true",
                                   help="randomize weights in layers to be fined tuned")
    spatial_parser.add_argument("--average", action="store_true", help="average between rotations and reflections")

    spatial_parser.add_argument("--save_pred_every", type=int, default=None,
                                   help="how frequently to save predictions")
    spatial_parser.add_argument("--pred_root", type=str, default=None, help="root for prediction outputs")
    spatial_parser.set_defaults(func="stnet.cmd.run_spatial")

    patients_parser = subparsers.add_parser(
        'patients',
        help='print patients',
    )
    patients_parser.set_defaults(func="stnet.cmd.print_spatial.print_spatial_patients")

    sections_parser = subparsers.add_parser(
        'sections',
        help='print sections',
    )
    sections_parser.set_defaults(func="stnet.cmd.print_spatial.print_spatial_sections")

    ensg_parser = subparsers.add_parser(
        'ensg',
        help='print ENSG name',
    )
    ensg_parser.add_argument("index", nargs="+", type=int, help="specify list of genes to print")
    ensg_parser.set_defaults(func="stnet.cmd.print_spatial.print_ensg_name")

    prepare_subparser = subparsers.add_parser("prepare", help="run preprocessing for dataset").add_subparsers(title="dataset", dest="dataset")
    prepare_subparser.required = True

    prepare_spatial_parser = prepare_subparser.add_parser("spatial", help="preprocessing for spatial")
    add_logging_arguments(prepare_spatial_parser)
    prepare_spatial_parser.add_argument("--root", type=str, default=config.SPATIAL_RAW_ROOT, help="directory containing raw data")
    prepare_spatial_parser.add_argument("--dest", type=str, default=config.SPATIAL_PROCESSED_ROOT, help="destination for patch info")
    prepare_spatial_parser.set_defaults(func="stnet.cmd.prepare.spatial")

    argcomplete.autocomplete(parser)
    return parser


def add_model_arguments(parser):
    parser.add_argument("--model", "-m", default="vgg11",
                        # choices=sorted(name for name in torchvision.models.__dict__ if name.islower() and not name.startswith("__") and callable(torchvision.models.__dict__[name])),  TODO: autocomplete speed issue
                        help="model architecture")
    parser.add_argument("--pretrained", action="store_true",
                        help="use ImageNet pretrained weights")
    parser.add_argument("--load", type=str, default=None, help="weights to load")

    parser.add_argument("--checkpoint_every", type=int, default=1, help="how frequently to save checkpoints")
    parser.add_argument("--checkpoint", type=str, default=None, help="root for checkpoints")
    parser.add_argument("--restart", action="store_true", help="automatically reload checkpoint")
    parser.add_argument("--keep_checkpoints", nargs="*", type=int, help="which checkpoints to keep (defaults to all)")


def add_window_arguments(parser, default=224):
    parser.add_argument("--window", type=int, default=default, help="window size")
    parser.add_argument("--window_raw", type=int, default=None, help="window size before scaling and cropping")
    parser.add_argument("--downsample", type=int, default=1, help="ratio to downsample by")


def add_augmentation_arguments(parser):
    parser.add_argument("--brightness", type=float, default=0, help="how much to jitter brightness")
    parser.add_argument("--contrast", type=float, default=0, help="how much to jitter contrast")
    parser.add_argument("--saturation", type=float, default=0, help="how much to jitter saturation")
    parser.add_argument("--hue", type=float, default=0, help="how much to jitter hue")


def add_device_arguments(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--gpu", action="store_const", const=True, dest="gpu", default=True, help="use GPU")
    group.add_argument("--cpu", action="store_const", const=False, dest="gpu", default=False, help="use CPU")


def add_training_arguments(parser):
    parser.add_argument("--optim", default="SGD",
                        # choices=sorted(name for name in torchvision.models.__dict__ if name.islower() and not name.startswith("__") and callable(torchvision.models.__dict__[name])),  TODO: autocomplete speed issue and change to optim instead of model
                        help="optimizer")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum for SGD")
    parser.add_argument("--weight_decay", type=float, default=0, help="weight decay for SGD")

    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch", type=int, default=64, help="training batch size")
    parser.add_argument("--test-batch", type=int, default=None, help="test batch size (defaults to match training batch size)")

    parser.add_argument("--workers", type=int, default=4, help="number of workers for dataloader")
    parser.add_argument("--iters-per-epoch", type=int, default=None, help="number of iterations per epoch")


def add_gene_filter_arguments(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--gene_filter", dest="gene_filter", choices=["none", "high", "tumor"], default="tumor",
                       help="special gene filters")
    group.add_argument("--gene_list", dest="gene_filter", nargs="+", type=str, help="specify list of genes to look at")
    group.add_argument("--gene_n", dest="gene_filter", type=int, nargs="?", default=250,
                       help="specify number of genes to look at (top mean expressed) defaults to 250")


def add_normalization_arguments(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--norm", action="store_const", const="norm", dest="norm", default=None,
                       help="normalize gene counts per spot")
    group.add_argument("--normfilter", action="store_const", const="normfilter", dest="norm", default=None,
                       help="normalize gene counts per spot (filtered only)")
    group.add_argument("--normpat", action="store_const", const="normpat", dest="norm", default=None,
                       help="normalize by median in patient")
    group.add_argument("--normsec", action="store_const", const="normsec", dest="norm", default=None,
                       help="normalize by median in section")


def add_logging_arguments(parser):
    import logging

    def loglevel(level):
        numeric_level = getattr(logging, level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError('Invalid log level: %s' % loglevel)
        return numeric_level

    parser.add_argument("--loglevel", "-l", type=loglevel,
                        default=logging.DEBUG, help="logging level")
    parser.add_argument("--logfile", type=str,
                        default=None,
                        help="file to store logs")

def add_task_arguments(parser):
    parser.add_argument("--tumor",    dest="tasks", action="append_const", const="Tumor")
    parser.add_argument("--gender",   dest="tasks", action="append_const", const="Gender")
    parser.add_argument("--age",      dest="tasks", action="append_const", const="Age")
    parser.add_argument("--stage",    dest="tasks", action="append_const", const="PathologicStage")
    parser.add_argument("--project",  dest="tasks", action="append_const", const="Project")
    parser.add_argument("--neighbor", dest="tasks", action="append_const", const="Neighbor")
