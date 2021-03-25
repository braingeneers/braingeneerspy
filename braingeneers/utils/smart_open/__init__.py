"""
Configures smart_open for braingeneers user on PRP/S3

Usage example:
    import braingeneers.utils.smart_open as smart_open
    with smart_open.open('s3://braingeneersdev/test_file.txt', 'r') as f:
        print(f.read())
"""
from smart_open import *
import functools
import os

# noinspection PyShadowingBuiltins
open = functools.partial(open, transport_params={
    'resource_kwargs': {'endpoint_url': os.environ.get('ENDPOINT_URL', 'https://s3.nautilus.optiputer.net')}
})
