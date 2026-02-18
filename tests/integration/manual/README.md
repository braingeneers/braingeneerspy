# Manual Range-Cache Integration Scripts

These scripts are integration diagnostics for `RangeCacheFile` and are intended to be run manually.
They are not collected by pytest because they are not named `test_*.py`.

## Scripts

- `tests/integration/manual/local_s3_streaming_emulation.py`
  - Synthetic latency benchmark comparing baseline tiny reads vs `RangeCacheFile`.
- `tests/integration/manual/remote_s3_h5_range_cache_segments.py`
  - Reads random all-channel HDF5 segments from a remote S3 file using:
    - `braingeneers.utils.smart_open_braingeneers.open`
    - `braingeneers.utils.range_cache_file.RangeCacheFile`
    - `h5py.File(...)`
  - Prints segment timings and `stats_markdown()` output.

## Run Commands

Run from repository root.

### Local emulation

```bash
PYTHONPATH=src python tests/integration/manual/local_s3_streaming_emulation.py
```

### Remote S3 Maxwell file (default URI)

```bash
PYTHONPATH=src python tests/integration/manual/remote_s3_h5_range_cache_segments.py --segments 6 --target-mb 5.0
```

### Remote S3 with explicit dataset path

```bash
PYTHONPATH=src python tests/integration/manual/remote_s3_h5_range_cache_segments.py --dataset-path sig --segments 6 --target-mb 5.0
```

## Notes

- Remote script requires network and valid S3 credentials/environment access.
- If needed, set endpoint behavior with existing Braingeneers endpoint configuration conventions.
