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
    * `braingeneers.utils.smart_open_braingeneers` a wrapper of `smart_open` for opening files on PRP/S3. See section below for the documentation and examples.

[github]: https://www.github.com/braingeneers/braingeneerspy

## How to Upgrade Package

To publish changes made to the **braingeneerspy** package on github, please follow these steps. Update the `version` variable in `setup.py`. To then receive the udpated **braingeneerspy** package on your personal computer, run the following command locally:
```
pip install --upgrade git+https://github.com/braingeneers/braingeneerspy.git # install braingeneers python package
```

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

## braingeneers.utils.smart_open_braingeneers
Configures smart_open for braingeneers use on PRP/S3. When importing this version of `smart_open` 
braingeneers defaults will be autoconfigured. Note that `smart_open` supports both local and S3 files, 
so it can be used for all files, not just S3 file access.

Basic usage example (copy/paste this to test your setup), if it works you will see a helpful bit of advice printed to the screen:

```python
import braingeneers.utils.smart_open_braingeneers as smart_open

with smart_open.open('s3://braingeneersdev/test_file.txt', 'r') as f:
    print(f.read())
```

You may also safely replace Python's default `open` function with `smart_open.open`, 
`smart_open` supports both local and remote files:

```python
import braingeneers.utils.smart_open_braingeneers as smart_open

open = smart_open.open
```
### Non-standard S3 endpoints:

`smart_open` and `s3wrangler` are pre-configured by default to the standard braingeneers S3 endpoint,
no configuration is necessary. If you would like to utilize a different S3 service you can specify a
new custom `ENDPOINT`, this can be a local path or an endpoint URL for another S3 service (s3wrangler
only supports S3 services, not local paths, `smart_open` supports local paths).

- Set an environment variable `ENDPOINT` with the new endpoint. Unix based example:`export ENDPOINT="https://s3-west.nrp-nautilus.io"`
- Call `braingeneers.set_default_endpoint(endpoint: str)` and `braingeneers.get_default_endpoint()`. 
  These functions will update both `smart_open` and `s3wrangler` (if it's an S3 endpoint, 
  local path endpoints are ignored by s3wrangler)

When running a job on the PRP you can use the PRP internal S3 endpoint,
which is faster than the default external endpoint (this will only work on jobs run in the PRP 
environment). Add the following environment variable to your job YAML file.
This will set the environment variable ENDPOINT_URL which overrides the
default external PPR/S3 endpoint, which is used if you don't set this variable.
Setting this environment variable can also be used to set an endpoint other than the PRP/S3.

```yaml
spec:
  template:
    spec:
      containers:
      - name: ...
        command: ...
        args: ...
        env:
          - name: "ENDPOINT"
            value: "http://rook-ceph-rgw-nautiluss3.rook"
```

Notes:
- There were version conflicts between 4.2.0 and 5.1.0 of smart_open. This configuration has been tested to work with 5.1.0.
