"""Setuptools description for the rl_navigation packages."""
from setuptools import setup
#  from setuptools.commands.build_ext import build_ext
import versioneer


#  class BuildMesonExtenstions(build_ext):
    #  """A custom build extension for adding compiler-specific options."""

    #  def build_extensions(self):
        #  ct = self.compiler.compiler_type
        #  opts = self.c_opts.get(ct, [])
        #  link_opts = self.l_opts.get(ct, [])
        #  if ct == "unix":
            #  opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            #  opts.append(cpp_flag(self.compiler))
            #  if has_flag(self.compiler, "-fvisibility=hidden"):
                #  opts.append("-fvisibility=hidden")
        #  elif ct == "msvc":
            #  opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        #  for ext in self.extensions:
            #  ext.extra_compile_args = opts
            #  ext.extra_link_args = link_opts
        #  build_ext.build_extensions(self)


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
        "gpu": ["tensorflow-gpu<2"],
    },
)
