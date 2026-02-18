# Manual Range-Cache Integration Scripts

These scripts are integration diagnostics for `RangeCacheFile` and are intended to be run manually.
They are not collected by pytest because they are not named `test_*.py`.

## Scripts

- `tests/integration/manual/local_s3_streaming_emulation.py`
  - Synthetic latency benchmark comparing baseline tiny reads vs `RangeCacheFile`.
- `tests/integration/manual/h5_streaming_cache_validation.py`
  - Reads random all-channel HDF5 segments from a remote S3 file using:
    - `braingeneers.utils.smart_open_braingeneers.open`
    - `braingeneers.utils.range_cache_file.RangeCacheFile`
    - `h5py.File(...)`
  - Prints segment timings and `stats_markdown()` output.
  - Defaults: 20 random segments of ~25 MiB each.

## Run Commands

Run from repository root.

### Local emulation

```bash
PYTHONPATH=src python tests/integration/manual/local_s3_streaming_emulation.py
```

### Remote S3 Maxwell file (default URI)

```bash
PYTHONPATH=src python tests/integration/manual/h5_streaming_cache_validation.py
```

### Remote S3 with explicit dataset path

```bash
PYTHONPATH=src python tests/integration/manual/h5_streaming_cache_validation.py --dataset-path sig --segments 20 --target-mb 25.0
```

## Notes

- Remote script requires network and valid S3 credentials/environment access.
- If needed, set endpoint behavior with existing Braingeneers endpoint configuration conventions.
