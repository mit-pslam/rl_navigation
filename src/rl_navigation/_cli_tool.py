"""Module handling all the command line options."""
import rl_navigation
import rl_navigation.subcommands.train as rlnav_train
import rl_navigation.subcommands.plot as rlnav_plot
import argparse


class ArgumentDispatcher:
    """Class to help separate between cli and actual code during testing."""

    def __init__(self, target_function):
        """Assign to a target function."""
        self.target_function = target_function

    def __call__(self, args):
        """Run the target functions."""
        self.target_func(**vars(args))


def make_train_subparser(subparsers):
    """Make a subparser for training."""
    train_parser = subparsers.add_parser("train", help="run training")
    train_parser.set_defaults(func=ArgumentDispatcher(rlnav_train.run_train))


def make_plot_subparser(subparsers):
    """Make a subparser for showing training progress."""
    plot_parser = subparsers.add_parser("plot", help="visualize training")
    plot_parser.set_defaults(func=ArgumentDispatcher(rlnav_plot.run_plot))


def main():
    """Run everything."""
    parser = argparse.ArgumentParser(
        prog="rl_navigation", description="utility to facilitate training in flight goggles"
    )
    parser.add_argument(
        "--version",
        action="version",
        version="rl_navigation version: {}".format(rl_navigation.__version__),
    )

    subparsers = parser.add_subparsers(help="sub-command help", dest="subparser_name")
    make_train_subparser(subparsers)
    make_plot_subparser(subparsers)

    args = parser.parse_args()
    if args.subparser_name is not None:
        args.func(args)
    else:
        parser.print_usage()
        parser.exit()
