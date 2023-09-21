# Braingeneers Python Utilities

[![ssec](https://img.shields.io/badge/SSEC-Project-purple?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA0AAAAOCAQAAABedl5ZAAAACXBIWXMAAAHKAAABygHMtnUxAAAAGXRFWHRTb2Z0d2FyZQB3d3cuaW5rc2NhcGUub3Jnm+48GgAAAMNJREFUGBltwcEqwwEcAOAfc1F2sNsOTqSlNUopSv5jW1YzHHYY/6YtLa1Jy4mbl3Bz8QIeyKM4fMaUxr4vZnEpjWnmLMSYCysxTcddhF25+EvJia5hhCudULAePyRalvUteXIfBgYxJufRuaKuprKsbDjVUrUj40FNQ11PTzEmrCmrevPhRcVQai8m1PRVvOPZgX2JttWYsGhD3atbHWcyUqX4oqDtJkJiJHUYv+R1JbaNHJmP/+Q1HLu2GbNoSm3Ft0+Y1YMdPSTSwQAAAABJRU5ErkJggg==&style=plastic)](https://escience.washington.edu/wetai/)
[![BSD License](https://badgen.net/badge/license/BSD-3-Clause/blue)](LICENSE)

## Getting Started

Welcome to the **Braingeneers Python Utilities** repository! This package collects and provides various Python code and utilities developed as part of the Braingeneers project. The package adheres to the Python Package Authority (PyPA) standards for package structure and organization.

# Development/Contribution

To get started with your development (or fork), click the "Open with GitHub Codespaces" button below to launch a fully configured development environment with all the necessary tools and extensions.

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/braingeneers/braingeneerspy?quickstart=1)

Instruction on how to contribute to this project can be found in the CONTRIBUTION.md

## Installation

You can install `braingeneerspy` using `pip` with the following commands:

### Install from GitHub (Recommended)

```bash
pip install --force-reinstall git+https://github.com/braingeneers/braingeneerspy.git
```

### Install from a Wheel (PyPI)

If you prefer to install a pre-built wheel, you can find the latest release on [PyPI](https://pypi.org/project/braingeneerspy/). Please replace `<version>` with the specific version you want to install.

```bash
pip install braingeneerspy==<version>
```

### Install with Optional Dependencies

You can install `braingeneerspy` with specific optional dependencies based on your needs. Use the following command examples:

- Install with IoT, analysis, and data access functions (skips machine learning and lab-specific dependencies):

```bash
pip install --force-reinstall git+https://github.com/braingeneers/braingeneerspy.git#egg=braingeneerspy[iot,analysis,data]
```

- Install with all optional dependencies:

```bash
pip install --force-reinstall git+https://github.com/braingeneers/braingeneerspy.git#egg=braingeneerspy[all]
```

Please note that macOS users may need to wrap the GitHub URL in quotes if they encounter issues during installation, as shown in the examples above.

## Optional Dependency Groups

Dependencies in `braingeneerspy` are organized into optional groups of requirements. You can install all dependencies with `all`, or you can install a specific set of dependencies. Here are the optional dependency groups:

- *Unspecified*: Minimal packages for data access will be installed.
- `all`: All optional dependencies will be included.
- `iot`: IoT dependencies such as AWS and Redis packages will be installed.
- `analysis`: Dependencies for data analysis routines, plotting tools, math libraries, etc.
- `ml`: Machine learning dependencies such as `torch` will be installed.
- `hengenlab`: Hengenlab data loader-specific packages such as `neuraltoolkit` will be installed.

## Committing Changes to the Repo

If you plan to make changes to the `braingeneerspy` package and publish them on GitHub, please follow these steps:

1. Update the `version` variable in `setup.py`.
2. To receive the updated `braingeneerspy` package on your local machine, run one of the pip install commands mentioned earlier.

## Modules and Subpackages

`braingeneerspy` includes several subpackages and modules, each serving a specific purpose within the Braingeneers project:

- `braingeneers.analysis`: Contains code for data analysis.
- `braingeneers.data`: Provides code for basic data access, including subpackages for handling electrophysiology, fluidics, and imaging data.
- `braingeneers.iot`: Offers code for Internet of Things (IoT) communication, including a messaging interface.
- `braingeneers.ml`: Contains code related to machine learning, such as a high-performance PyTorch data loader for electrophysiology data.
- `braingeneers.utils`: Provides utility functions, including S3 access and smart file opening.

## S3 Access and Configuration

### `braingeneers.utils.s3wrangler`

This module extends the `awswrangler.s3 package` for Braingeneers/PRP access. For API documentation and usage examples, please visit the [official documentation](https://aws-data-wrangler.readthedocs.io/en/2.4.0-docs/api.html#amazon-s3).

Here's a basic usage example:

```python
import braingeneers.utils.s3wrangler as wr

# Get all UUIDs from s3://braingeneers/ephys/
uuids = wr.list_directories('s3://braingeneers/ephys/')
print(uuids)
```

### `braingeneers.utils.smart_open_braingeneers`

This module configures `smart_open` for Braingeneers use on PRP/S3. When importing this version of `smart_open`, Braingeneers defaults will be autoconfigured. Note that `smart_open` supports both local and S3 files, so it can be used for all files, not just S3 file access.

Here's a basic usage example:

```python
import braingeneers.utils.smart_open_braingeneers as smart_open

with smart_open.open('s3://braingeneersdev/test_file.txt', 'r') as f:
    print(f.read())
```

You can also safely replace Python's default `open` function with `smart_open.open`:

```python
import braingeneers.utils.smart_open_braingeneers as smart_open

open = smart_open.open
```

## Customizing S3 Endpoints

By default, `smart_open` and `s3wrangler` are pre-configured for the standard Braingeneers S3 endpoint. However, you can specify a custom `ENDPOINT` if you'd like to use a different S3 service. This can be a local path or an endpoint URL for another S3 service (note that `s3wrangler` only supports S3 services, not local paths, while `smart_open` supports local paths).

To set a custom endpoint, follow these steps:

1. Set an environment variable `ENDPOINT` with the new endpoint. For example, on Unix-based systems:

   ```bash
   export ENDPOINT="https://s3-west.nrp-nautilus.io"
   ```

2. Call `braingeneers.set_default_endpoint(endpoint: str)` and `braingeneers.get_default_endpoint()`. These functions will update both `smart_open` and `s3wrangler` (if it's an S3 endpoint, local path endpoints are ignored by `s3wrangler`).

### Using the PRP Internal S3 Endpoint

When running a job on the PRP, you can use the PRP internal S3 endpoint, which is faster than the default external endpoint. To do this, add the following environment variable to your job YAML file:

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

Please note that this will only work on jobs run in the PRP environment. Setting the `ENDPOINT` environment variable can also be used to specify an endpoint other than the PRP/S3.

### Notes

- `braingeneerspy` is compatible with `smart_open` version 5.1.0. If you encounter issues, make sure to use this version for compatibility.
