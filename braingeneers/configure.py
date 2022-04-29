""" Global package functions and helpers for Braingeneers specific configuration. """
import functools
import os
from typing import List, Tuple, Union, Iterable, Iterator
import re
import itertools
import importlib


"""
Preconfigure remote filesystem access
Default S3 endpoint/bucket_name, note this can be overridden with the environment variable ENDPOINT_URL
    or by calling set_default_endpoint(...). Either an S3 endpoint or local path can be used.
"""
DEFAULT_ENDPOINT = "https://s3-west.nrp-nautilus.io"  # PRP/S3/CEPH default
CURRENT_ENDPOINT = None
_open = None  # a reference to the current smart_open function that is configured this should not be used directly

"""
Define all groups of dependencies here. `minimal` dependencies are always installed with
the braingeneerspy package, all other groups of dependencies will be installed
braingeneerspy[all] or braingeneerspy[specific_dependency_group]
Maintainers are encouraged to move dependencies to specific dependency groups and keep
`minimal` as small as possible. `minimal` would typically be used in a RaspberryPI
environment for example, where large dependencies can be very slow to build.
"""
DEPENDENCIES = {
    # Minimal dependencies are always installed with a pip install of the repo
    'minimal': [
        'requests',
        'numpy',
        'tenacity',
        'awswrangler',
        'sortedcontainers',
        # 'boto3==1.17.96',  # depends on awscrt==0.11.22
        'boto3',  # if version conflicts occur revert back to above version
        'smart_open>=5.1.0',
        'h5py',
    ],
    # Specific dependency groups allow unnecessary (and often large) dependencies to be skipped
    # add dependency groups here, changes will be dynamically added to setup(...)
    'iot': [
        'awsiotsdk==1.6.0',
        'redis',
    ],
    'analysis': [
        'scipy',
        'powerlaw',
        'matplotlib',
    ],
    'ml': [
        'torch',
        'sklearn',
    ],
    'hengenlab': [
        'neuraltoolkit @ git+https://github.com/hengenlab/neuraltoolkit.git',
        'sahara_work @ git+https://github.com/hengenlab/sahara_work.git',
    ],
}


def get_default_endpoint() -> str:
    """
    Returns the current default (S3) endpoint. By default this will point to the standard
    S3 location where data files are stored. Use set_default_endpoint(...) to change.

    :return: str: the current endpoint
    """
    return CURRENT_ENDPOINT


def set_default_endpoint(endpoint: str = None) -> None:
    """
    Sets the default S3 endpoint and (re)configures braingeneers.utils.smart_open and
    braingeneers.utils.s3wrangler to utilize the new endpoint. This endpoint may also be set
    to the local filesystem with a relative or absolute local path.

    Examples:
      PRP/S3/CEPH:      "https://s3-west.nrp-nautilus.io"
      PRP/S3/SeaweedFS: "https://swfs-s3.nrp-nautilus.io"
      Local Filesystem: "/home/user/project/files/" (absolute) or "project/files/" (relative)

    :param endpoint: S3 or local-path endpoint as shown in examples above, if None will look for ENDPOINT
        environment variable, then default to DEFAULT_ENDPOINT if not found.
    """
    # lazy loading of imports is necessary so that we don't import these classes with braingeneers root
    # these imports can cause SSL warning messages for some users, so it's especially important to avoid
    # importing them unless S3 access is needed.
    import boto3
    import smart_open
    import awswrangler

    global _open
    endpoint = endpoint if endpoint is not None else os.environ.get('ENDPOINT', DEFAULT_ENDPOINT)

    # smart_open
    if endpoint.startswith('http'):
        transport_params = {'client': boto3.Session().client('s3', endpoint_url=endpoint)}
        _open = functools.partial(smart_open.open, transport_params=transport_params)
    else:
        _open = smart_open.open

    # s3wrangler - only update s3wrangler if the endpoint is S3 based, s3wrangler doesn't support local
    if endpoint.startswith('http'):
        awswrangler.config.s3_endpoint_url = endpoint

    global CURRENT_ENDPOINT
    CURRENT_ENDPOINT = endpoint


@functools.lru_cache(maxsize=None)
def verify_optional_extras(optional_extras: (str, List[str]), raise_exception: bool = True):
    """
    Verifies whether one or more extra dependencies have been installed,
    extras are defined in setup.py->dependencies.

    This function is normally called at the beginning, or initialization of a code that relies
    on an optional package.

    This function can be called repeatedly, it will cache the results after the first call for efficiency.

    Example usage:
        # Verify hengenlab dependencies are installed
        braingeneers.verify_optional_extras('hengenlab')

        # Verify iot and ml dependencies
        braingeneers.verify_optional_extras(['iot', 'ml'])

    :param optional_extras: String or list of strings of "extras" packages to verify are installed.
    :param raise_exception: boolean default == True, raises an exception when missing packages,
        else the function will simply return True|False
    :raise ModuleNotFoundError:
    :return: None if all dependencies are met else a list of missing dependencies
    """
    optional_extras = set([optional_extras] if isinstance(optional_extras, str) else optional_extras)  # ensure set
    required_packages = set(itertools.chain(*[v for k, v in DEPENDENCIES.items() if k in optional_extras]))
    missing_packages = [
        package
        for package in required_packages
        if importlib.util.find_spec(re.split(r'[^a-z^A-Z^0-9^_^-]+', package)[0]) is None
    ]

    if len(missing_packages) > 0 and raise_exception:
        exception_message = f'Package dependencies are missing, ' \
                            f'see README.md for optional package installation instructions ' \
                            f'packages: {",".join(map(str, optional_extras))}, ' \
                            f'missing packages: {",".join(map(str, missing_packages))}.'
        raise ModuleNotFoundError(exception_message)
    else:
        return missing_packages if len(missing_packages) > 0 else None
