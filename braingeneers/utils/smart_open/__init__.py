"""
Configures smart_open for braingeneers use on PRP/S3
This replaces your python default `open` function with smart_open.open
which supports both local and remote files.

Basic usage example (copy/paste this to test your setup), if it works you will see a helpful bit of advice printed to the screen:
    ```
    from braingeneers.utils import smart_open

    with smart_open.open('s3://braingeneersdev/test_file.txt', 'r') as f:
        print(f.read())
    ```

You may also replace python's default open with smart_open.open:
    ```
    from braingeneers.utils import smart_open

    open = smart_open.open
    ```

To use the PRP internal S3 endpoint, which is faster than the default external
endpoint, add the following environment variable to your job YAML file.
This will set the environment variable ENDPOINT_URL which overrides the
default external PPR/S3 endpoint, which is used if you don't set this variable.
Setting this environment variable can also be used to set an endpoint other than the PRP/S3.

Job YAML:
=========
spec:
  template:
    spec:
      containers:
      - name: ...
        command: ...
        args: ...
        env:
          - name: "ENDPOINT_URL"
            value: "http://rook-ceph-rgw-nautiluss3.rook"


"""
import os
import boto3
import functools
from smart_open import *


endpoint_url = os.environ.get('ENDPOINT_URL', 'https://s3.nautilus.optiputer.net')
transport_params = {'client': boto3.Session().client('s3', endpoint_url=endpoint_url)}
open = functools.partial(open, transport_params=transport_params)
