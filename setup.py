from setuptools import setup, find_packages
from os import path
import itertools


"""
Define all groups of dependencies here. `minimal` dependencies are always installed with
the braingeneerspy package, all other groups of dependencies will be installed
braingeneerspy[all] or braingeneerspy[specific_dependency_group]
Maintainers are encouraged to move dependencies to specific dependency groups and keep
`minimal` as small as possible. `minimal` would typically be used in a RaspberryPI
environment for example, where large dependencies can be very slow to build.
"""
dependencies = {
    # Minimal dependencies are always installed with a pip install of the repo
    'minimal': [
        'matplotlib',
        'requests',
        'numpy',
        'scipy',
        'tenacity',
        'awswrangler',
        'sortedcontainers',
        'powerlaw',
        # 'boto3==1.17.96',  # depends on awscrt==0.11.22
        'boto3',  # if version conflicts occur revert back to above version
        'smart_open>=5.1.0',
    ],
    # Specific dependency groups allow unnecessary (and often large) dependencies to be skipped
    # add dependency groups here, changes will be dynamically added to setup(...)
    'iot': [
        'awsiotsdk==1.6.0',
        'redis',
    ],
    'ml': [
        'torch',
    ],
    'hengenlab': [
        'neuraltoolkit @ git+https://github.com/hengenlab/neuraltoolkit.git',
    ],
}

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

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
    install_requires=dependencies['minimal'],
    extras_require={
        'all': list(itertools.chain(*dependencies.values())),              # all dependencies included
        **{k: v for k, v in dependencies.items() if k != 'minimal'},    # each dependency group
    }
)
