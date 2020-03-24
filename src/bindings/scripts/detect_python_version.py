"""Script to hand meson information about target python install."""
import argparse
import site
import sys
import os


def main():
    """Get sysconfig info and print to console."""
    parser = argparse.ArgumentParser(description="utility for getting package path")
    parser.add_argument("--user", action='store_true', help='use user path instead')
    args = parser.parse_args()

    if args.user:
        path_list = site.getusersitepackages()
    else:
        path_list = site.getsitepackages()

    if len(path_list) == 0:
        print("failed to get paths", file=os.stderr)
        sys.exit(1)

    if isinstance(path_list, list):
        print(path_list[0])
    else:
        print(path_list)


if __name__ == "__main__":
    main()
