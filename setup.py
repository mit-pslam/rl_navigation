"""Setuptools description for the rl_navigation packages."""
from setuptools import setup
import versioneer


setup(
    name="rl_navigation",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="package that contains code to train RL-policies in a FlightGoggles enviornment",
    license="TBD",
    author="Mark Mazumder",
    python_requires=">=3.7",
    package_dir={"": "src"},
    packages=["rl_navigation", "rl_navigation.subcommands"],
    entry_points={"console_scripts": ["rl_navigation=rl_navigation._cli_tool:main"]},
    install_requires=[
        "numpy",
        "stable-baselines",
        "tensorflow<2",
        "zmq",
        "opencv-python",
        "transforms3d",
        "yacs",
    ],
    classifiers=["Porgramming Language :: Python :: 3.7"],
    extras_require={
        "doc": ["sphinx", "sphinx_rtd_theme"],
        "test": ["tox"],
        "gpu": ["tesnorflow-gpu<2"],
    },
)
