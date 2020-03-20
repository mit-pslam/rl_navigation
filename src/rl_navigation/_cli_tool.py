"""Module handling all the command line options."""
import argparse


def main():
    """Run everything."""
    parser = argparse.ArgumentParser(description="utility to facilitate training in flight goggles")

    args = parser.parse_args()
    print(args)
