"""
Extends the awswrangler.s3 package for Braingeneers/PRP access.
See documentation: https://aws-data-wrangler.readthedocs.io/en/2.4.0-docs/api.html#amazon-s3

Usage examples:
    import braingeneers.utils.s3wrangler as wr
    uuids = wr.list_directories('s3://braingeneers/ephys/')
    print(uuids)
"""
import awswrangler
from awswrangler import config
from awswrangler.s3 import *
import braingeneers
import botocore

awswrangler.config.botocore_config = botocore.config.Config(
        retries={"max_attempts": 10}, 
        connect_timeout=20, 
        max_pool_connections=50)

braingeneers.set_default_endpoint()
