"""Setuptools description for the rl_navigation packages."""
from setuptools import setup

version = {}
with open("src/rl_navigation/version.py") as fp:
    exec(fp.read(), version)

setup(
    name="rl_navigation",
    version=version["__version__"],
    author="Mark Mazumder",
    python_requires=">=3.7",
    package_dir={"": "src"},
    packages=["rl_navigation"],
    license="TBD",
    description="package that contains code to train RL-policies in a FlightGoggles enviornment",
    entry_points={"console_scripts": ["rl_navigation = rl_navigation._cli_tool:main"]},
    install_requires=[
        "fs",
        "numpy",
        "scipy",
        "tqdm",
        "pandas",
        "scikit-image",
        "seaborn",
        "opencv_python",
        "pyyaml",
    ],
    extras_require={"test": ["pytest", "hypothesis", "pytest-cov", "coverage"]},
)
