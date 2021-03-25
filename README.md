# Braingeneers Python Utilities

[This package][github] is supposed to collect, as well as make installable
through Pip, all of the Python code and utilities that we develop as
part of the Braingeneers project. There are five subpackages:
  * `braingeneers.analysis` contains code for data analysis.
  * `braingeneers.datasets_electrophysiology` contains methods which load and manipulate ephys data.
  * `braingeneers.datasets_fluidics` contains methods which load and manipulate fluidics data.
  * `braingeneers.datasets_imaging` contains methods which load and manipulate imaging data.
  * `braingeneers.drylab` contains code for neuronal simulations.'
  * `braigeneers.utils`  
    * `braingeneers.utils.messaging` a single interface for all messaging and inter-device data transfer functions (MQTT, redis, device state, etc.). A wetAI tutorial on this package exists.
    * `braingeneers.utils.s3wrangler` a wrapper of `awswrangler.s3` for accessing PRP/S3. See section below for the documentation and examples.
    * `braingeneers.utils.smart_open` a wrapper of `smart_open` for opening files on PRP/S3. See section below for the documentation and examples.

[github]: https://www.github.com/braingeneers/braingeneerspy

## braingeneers.utils.s3wrangler
Extends the `awswrangler.s3 package` for Braingeneers/PRP access.
See API documentation: https://aws-data-wrangler.readthedocs.io/en/2.4.0-docs/api.html#amazon-s3

Usage examples:
```python
import braingeneers.utils.s3wrangler as wr

# get all UUIDs from s3://braingeneers/ephys/
uuids = wr.list_directories('s3://braingeneers/ephys/')
print(uuids)
```

## braingeneers.utils.smart_open
Configures smart_open for braingeneers use on PRP/S3. When importing this version of `smart_open` 
braingeneers defaults will be auto configured. Note that `smart_open` supports both local and S3 files so 
it can be used for all files, not just S3 file access.

Usage example:
```python
import braingeneers.utils.smart_open as smart_open

with smart_open.open('s3://braingeneersdev/test_file.txt', 'r') as f:
    print(f.read())
```
