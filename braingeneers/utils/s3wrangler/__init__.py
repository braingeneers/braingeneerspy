"""
Extends the awswrangler.s3 package for Braingeneers/PRP access.
See documentation: https://aws-data-wrangler.readthedocs.io/en/2.4.0-docs/api.html#amazon-s3

Usage examples:
    import braingeneers.utils.s3wrangler as wr
    uuids = wr.list_directories('s3://braingeneers/ephys/')
    print(uuids)
"""
import os
import awswrangler
from awswrangler.s3 import *


prp_s3_endpoint = 'https://s3-west.nrp-nautilus.io'
awswrangler.config.s3_endpoint_url = os.environ.get('ENDPOINT_URL', prp_s3_endpoint)
