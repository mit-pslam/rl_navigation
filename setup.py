"""Setuptools description for the rl_navigation packages."""
from setuptools import setup
import versioneer

requirements = ["numpy"]
doc_requirements = ["sphinx", "sphinx_rtd_theme"]
test_requirements = ["pytest", "pytest-cov", "coverage"]

setup(
    name="rl_navigation",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="package that contains code to train RL-policies in a FlightGoggles enviornment",
    license="TBD",
    author="Mark Mazumder",
    python_requires=">=3.7",
    packages=["rl_navigation", "rl_navigation.subcommands"],
    entry_points={"console_scripts": ["rl_navigation=rl_navigation._cli_tool:main"]},
    install_requires=requirements,
    classifiers=["Porgramming Language :: Python :: 3.7"],
    extras_require={"doc": doc_requirements, "test": test_requirements},
)
