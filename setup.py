"""Setuptools description for the rl_navigation packages."""
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import subprocess
import versioneer
import pathlib
import tempfile
import sys
import os


class BuildMesonExtenstions(build_ext):
    """A custom build extension for adding compiler-specific options."""

    def build_extensions(self):
        interpreter = str(sys.executable)
        install_path = pathlib.Path(".").absolute() / self.build_lib
        binding_directory = pathlib.Path("src") / "bindings"
        print(binding_directory)
        print(pathlib.Path(".").absolute())
        print(list(pathlib.Path(".").iterdir()))
        print(list((pathlib.Path(".") / "src").iterdir()))

        build_dir = self.build_temp
        ret = subprocess.run(
            [
                "meson",
                "setup",
                build_dir,
                str(binding_directory),
                "-Dinterpreter_path={}".format(interpreter),
                "-Dbuild_result_path={}".format(install_path),
            ], capture_output=True, encoding='utf-8'
        )

        if ret.returncode != 0:
            print("Meson out: {}".format(ret.stdout))
            print("Meson error: {}".format(ret.stderr))
            sys.exit(1)

        subprocess.run(["ninja", "install"], cwd=build_dir)


setup(
    name="rl_navigation",
    version=versioneer.get_version(),
    cmdclass=dict(versioneer.get_cmdclass(), **{"build_ext": BuildMesonExtenstions}),
    description="package that contains code to train RL-policies in a FlightGoggles enviornment",
    license="TBD",
    author="MIT AIIA sUAS Disaster Response",
    python_requires=">=3.7",
    package_dir={"": "src"},
    packages=["rl_navigation", "rl_navigation.subcommands"],
    entry_points={"console_scripts": ["rl_navigation=rl_navigation._cli_tool:main"]},
    ext_modules=[Extension("sophus", [])],
    install_requires=[
        "numpy",
        "stable-baselines",
        "tensorflow-gpu<2",
        "zmq",
        "opencv-python",
        "transforms3d",
        "yacs",
    ],
    classifiers=["Programming Language :: Python :: 3.7"],
    extras_require={
        "doc": ["sphinx", "sphinx_rtd_theme"],
        "test": ["tox"],
    },
)
