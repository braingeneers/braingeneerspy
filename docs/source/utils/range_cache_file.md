# Range Cache File Guide

## Overview

`braingeneers.utils.range_cache_file.RangeCacheFile` wraps a seekable binary file-like object
and reduces backend range-read churn by caching and adaptively resizing fetch windows.

This is designed for workloads such as HDF5/NWB over S3 where:

- The file layer issues many small reads.
- Reads often become mostly forward/contiguous after initial metadata/index lookups.

`RangeCacheFile` is zero-config for callers:

```python
import h5py
import braingeneers.utils.smart_open_braingeneers as smart_open
from braingeneers.utils.range_cache_file import RangeCacheFile

with smart_open.open("s3://bucket/path/file.raw.h5", "rb") as raw:
    wrapped = RangeCacheFile(raw)
    with h5py.File(wrapped, "r") as h5:
        dataset = h5["sig"]
        data = dataset[:, 0:1000]
```

## How It Works

`RangeCacheFile` combines:

- A mergeable segment cache with byte-budget eviction.
- Access-pattern tracking over recent reads.
- Adaptive fetch windows:
  - `probe` mode starts conservative.
  - `seq` mode expands read-ahead when streaming behavior is observed.
  - Demotion back to `probe` occurs when patterns become jumpy/random.

The goal is to tolerate early small/random file probes, then collapse long sequential read phases
into fewer backend range reads.

## Stats and Diagnostics

Two reporting interfaces are available:

- `stats()` returns machine-readable counters and derived metrics.
- `stats_markdown()` returns a human-readable table.

Common metrics:

- `underlying_read_calls`: number of actual backend reads.
- `bytes_served_to_caller` vs `underlying_bytes_fetched`: useful for amplification analysis.
- `hit_rate` and `byte_hit_rate`: cache effectiveness.
- `probe_fetch_current` and `seq_fetch_current`: currently learned fetch windows.

## Recommended Usage Pattern

Use it where a file-like object is accepted and random tiny reads over S3 are expensive:

- `h5py.File(...)` over `smart_open_braingeneers.open(...)`.
- Maxwell/HDF5 reads that move into contiguous frame slices.

For simple local disk reads or uniformly random access, benefits may be smaller.

## Manual Validation Workflows

Manual integration scripts live under:

- `tests/integration/manual/local_s3_streaming_emulation.py`
- `tests/integration/manual/h5_streaming_cache_validation.py`

These scripts are intended for profiling and operational validation, not pytest collection.
See `tests/integration/manual/README.md` for run commands.

## Limitations

- Thread safety for concurrent multi-threaded consumers is not a primary target.
- Wrapper behavior is file-layer generic and does not parse HDF5 schema semantics.
- Metrics are process-local and reset per wrapper instance.
