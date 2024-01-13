"""Setuptools description for the rl_navigation packages."""
from setuptools import setup, Extension, find_namespace_packages, find_packages
from setuptools.command.build_ext import build_ext
import subprocess
import versioneer
import pathlib
import sys


class BuildMesonExtenstions(build_ext):
    """A custom build extension for adding compiler-specific options."""

    def build_extensions(self):
        interpreter = str(sys.executable)
        install_path = pathlib.Path(".").absolute() / self.build_lib
        binding_directory = pathlib.Path("src") / "bindings"

        build_dir = self.build_temp

        # Removing folder prevents ninja below from dying, if folder exists
        subprocess.run(["rm", "-r", build_dir])

        ret = subprocess.run(
            [
                "meson",
                "setup",
                build_dir,
                str(binding_directory),
                "-Dinterpreter_path={}".format(interpreter),
                "-Dbuild_result_path={}".format(install_path),
            ],
            capture_output=True,
            encoding="utf-8",
        )

        if ret.returncode != 0:
            print("setup error: {}".format(str(ret.stdout)))
            sys.exit(1)

        ret = subprocess.run(["ninja", "install"], cwd=build_dir)
        if ret.returncode != 0:
            print("build error: {}".format(str(ret.stdout)))
            sys.exit(1)


setup(
    name="rl_navigation",
    version=versioneer.get_version(),
    cmdclass=dict(versioneer.get_cmdclass(), **{"build_ext": BuildMesonExtenstions}),
    description="package that contains code to train RL-policies in a FlightGoggles enviornment",
    license="TBD",
    author="MIT AIIA sUAS Disaster Response",
    python_requires=">=3.7",
    package_dir={"": "src"},
    packages=find_namespace_packages(include=["rl_navigation_models.*"], where="src")
    + find_packages(where="src"),
    include_package_data=True,
    ext_modules=[Extension("sophus", [])],
    install_requires=[
        "numpy<1.20",
        "gym<=0.21.0",
        "importlib-metadata<5.0",  # required to support gym properly
        "scipy>=1.4.0",
        "zmq",
        "opencv-python==4.1.2.30",
        "transforms3d",
        "yacs",
        "matplotlib",
        "tesse-gym@git+https://git@github.com/MIT-TESSE/tesse-gym.git@master#egg=tesse-gym",
        "flightgoggles@git+https://git@github.com/mit-pslam/pyFlightGoggles.git@opencv4-compatible#egg=flightgoggles",
    ],
    
    dependency_links=[
        "git+https://git@github.com/MIT-TESSE/tesse-gym.git@master#egg=tesse-gym",
        "git+https://git@github.com/mit-pslam/pyFlightGoggles.git@opencv4-compatible#egg=flightgoggles",
    ],
    classifiers=["Programming Language :: Python :: 3.7"],
    extras_require={
        "doc": ["sphinx", "sphinx_rtd_theme"],
        "test": ["tox"],
        "rllib": ["ray[default,rllib]==1.13.0", "aiohttp<3.8.0", "aioredis==1.3.1"],
        "fast_depth": [
            "fast-depth-estimation@git+ssh://git@github.mit.edu/aiia-suas-disaster-response/fast-depth-estimation.git@master#egg=fast-depth-estimation",
        ],
        "ros": [
            "rl_navigation_ros@git+ssh://git@github.mit.edu/aiia-suas-disaster-response/rl_navigation_ros#egg=rl_navigation_ros"
        ],
    },
)
