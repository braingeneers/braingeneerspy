""" Global package functions and helpers for Braingeneers specific configuration. """
import functools
import os

# Preconfigure remote filesystem access
# Default S3 endpoint/bucket_name, note this can be overridden with the environment variable ENDPOINT_URL
#   or by calling set_default_endpoint(...). Either an S3 endpoint or local path can be used.
DEFAULT_ENDPOINT = "https://s3-west.nrp-nautilus.io"  # PRP/S3/CEPH default
CURRENT_ENDPOINT = None
_open = None  # a reference to the current smart_open function that is configured this should not be used directly


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
