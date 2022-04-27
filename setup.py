from setuptools import setup, find_packages
from os import path
import itertools
import braingeneers.configure as config


here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# DEPENDENCIES:
#   Dependencies are listed in configure.py, they are imported from braingeneers.configure.dependencies
#   dynamically. Please edit dependencies in configure.py, not here. This was done so that the dependency
#   list is not duplicated. It is also necessary because
#
setup(
    name='braingeneerspy',
    version='0.1.0',
    python_requires='>=3.6.0',
    description='Braingeneers Python utilities',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/braingeneers/braingeneerspy',
    author='Braingeneers',
    include_package_data=True,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Braingeneers, collaborators',
        'Programming Language :: Python :: 3',
        'License :: MIT',
    ],
    packages=find_packages(exclude=()),
    # Edit these fields using the `dependencies` variable above, no changes here are necessary
    install_requires=config.DEPENDENCIES['minimal'],
    extras_require={
        'all': list(itertools.chain(*config.DEPENDENCIES.values())),              # all dependencies included
        **{k: v for k, v in config.DEPENDENCIES.items() if k != 'minimal'},    # each dependency group
    }
)
