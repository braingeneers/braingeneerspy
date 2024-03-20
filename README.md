# Braingeneers Python Utilities

[![ssec](https://img.shields.io/badge/SSEC-Project-purple?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA0AAAAOCAQAAABedl5ZAAAACXBIWXMAAAHKAAABygHMtnUxAAAAGXRFWHRTb2Z0d2FyZQB3d3cuaW5rc2NhcGUub3Jnm+48GgAAAMNJREFUGBltwcEqwwEcAOAfc1F2sNsOTqSlNUopSv5jW1YzHHYY/6YtLa1Jy4mbl3Bz8QIeyKM4fMaUxr4vZnEpjWnmLMSYCysxTcddhF25+EvJia5hhCudULAePyRalvUteXIfBgYxJufRuaKuprKsbDjVUrUj40FNQ11PTzEmrCmrevPhRcVQai8m1PRVvOPZgX2JttWYsGhD3atbHWcyUqX4oqDtJkJiJHUYv+R1JbaNHJmP/+Q1HLu2GbNoSm3Ft0+Y1YMdPSTSwQAAAABJRU5ErkJggg==&style=plastic)](https://escience.washington.edu/wetai/)
[![MIT License](https://badgen.net/badge/license/MIT/blue)](LICENSE)
[![Documentation Status](https://readthedocs.org/projects/braingeneers/badge/?version=latest)](https://braingeneers.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/166130153.svg)](https://zenodo.org/badge/latestdoi/166130153)

## Getting Started

Welcome to the **Braingeneers Python Utilities** repository! This package collects and provides various Python code and utilities developed as part of the Braingeneers project. The package adheres to the Python Package Authority (PyPA) standards for package structure and organization.

## Contribution

We welcome contributions from collaborators and researchers interested in our work. If you have improvements, suggestions, or new findings to share, please submit a pull request. Your contributions help advance our research and analysis efforts.

To get started with your development (or fork), click the "Open with GitHub Codespaces" button below to launch a fully configured development environment with all the necessary tools and extensions.

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/braingeneers/braingeneerspy?quickstart=1)

Instruction on how to contribute to this project can be found in the [CONTRIBUTION.md](https://github.com/braingeneers/braingeneerspy/blob/development/.github/CONTRIBUTING.md).

## Installation

You can install `braingeneers` using `pip` with the following commands:

### Install from PyPI (Recommended)

```bash
pip install braingeneers
```

### Install from GitHub

```bash
pip install --force-reinstall git+https://github.com/braingeneers/braingeneerspy.git
```

### Install with Optional Dependencies

You can install `braingeneers` with specific optional dependencies based on your needs. Use the following command examples:

- Install with machine-learning dependencies:

```bash
pip install "braingeneers[ml]"
```

- Install with Hengen lab dependencies:

```bash
pip install "braingeneers[hengenlab]"
```

- Install with developer dependencies (running tests and building sphinx docs):

```bash
pip install "braingeneers[dev]"
```

- Install with all optional dependencies:

```bash
pip install "braingeneers[all]"
```

## Committing Changes to the Repo

To make changes and publish them on GitHub, please refer to the [CONTRIBUTING.md](https://github.com/braingeneers/braingeneerspy/blob/master/.github/CONTRIBUTING.md) file for up-to-date guidelines.

## Modules and Subpackages

`braingeneers` includes several subpackages and modules, each serving a specific purpose within the Braingeneers project:

- `braingeneers.analysis`: Contains code for data analysis.
- `braingeneers.data`: Provides code for basic data access, including subpackages for handling electrophysiology, fluidics, and imaging data.
- `braingeneers.iot`: Offers code for Internet of Things (IoT) communication, including a messaging interface.
- `braingeneers.ml`: Contains code related to machine learning, such as a high-performance PyTorch data loader for electrophysiology data.
- `braingeneers.utils`: Provides utility functions, including S3 access and smart file opening.

## S3 Access and Configuration

### `braingeneers.utils.s3wrangler`

This module extends the `awswrangler.s3 package` for Braingeneers/PRP access. For API documentation and usage examples, please visit the [official documentation](https://aws-sdk-pandas.readthedocs.io/en/stable/).

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

## Documentation

The docs directory has been set up using `sphinx-build -M html docs/source/ docs/build/` to create a base project Documentation structure. You can add inline documentation (NumPy style) to further enrich our project's documentation. To render the documentation locally, navigate to the `docs/build/html` folder in the terminal and run `python3 -m http.server`.

## Working in Codespaces

### Project Structure

- **src/:** This folder contains scripts and notebooks representing completed work by the team.

- **pyproject.toml:** This file follows the guidelines from [PyPA](https://packaging.python.org/tutorials/packaging-projects/) for documenting project setup information.

### Customizing the Devcontainer

The `devcontainer.json` file allows you to customize your Codespace container and VS Code environment using extensions. You can add more extensions to tailor the environment to your specific needs. Explore the VS Code extensions marketplace for additional tools that may enhance your workflow.

For more information about Braingeneers, visit our [website](https://braingeneers.ucsc.edu/).
