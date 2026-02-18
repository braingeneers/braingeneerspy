from __future__ import annotations

import argparse
import random
import time

import h5py

import braingeneers.utils.smart_open_braingeneers as smart_open
from braingeneers.utils.range_cache_file import RangeCacheFile


DEFAULT_URI = "s3://braingeneers/ephys/2022-03-17-e-tetanusucsb/original/data/experiment1_stim.raw.h5"
DEFAULT_ALT_DATASET = "/data_store/data0000/groups/routed/raw"


def resolve_dataset_path(h5: h5py.File, explicit_path: str | None) -> str:
    if explicit_path:
        if explicit_path not in h5:
            raise KeyError(f"Dataset path not found: {explicit_path}")
        return explicit_path
    if "sig" in h5:
        return "sig"
    if DEFAULT_ALT_DATASET in h5:
        return DEFAULT_ALT_DATASET
    raise KeyError("Could not auto-resolve a primary data dataset path. Pass --dataset-path explicitly.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read random all-channel segments from a remote H5 file with RangeCacheFile wrapping smart_open."
    )
    parser.add_argument("--uri", default=DEFAULT_URI, help="S3 URI to the H5/NWB file.")
    parser.add_argument("--dataset-path", default=None, help="Dataset path inside the H5 file (default: auto-detect).")
    parser.add_argument("--target-mb", type=float, default=25.0, help="Approx target segment size in MiB.")
    parser.add_argument("--segments", type=int, default=20, help="Number of random segments to read.")
    parser.add_argument("--seed", type=int, default=20260218, help="Random seed for reproducible segment selection.")
    args = parser.parse_args()

    print("Range-cache remote H5 segment reader")
    print(f"- uri: {args.uri}")
    print(f"- target_mb: {args.target_mb}")
    print(f"- segments: {args.segments}")
    print(f"- seed: {args.seed}")

    rng = random.Random(args.seed)
    with smart_open.open(args.uri, "rb") as remote_file:
        wrapped = RangeCacheFile(remote_file)
        with h5py.File(wrapped, "r") as h5:
            dataset_path = resolve_dataset_path(h5, args.dataset_path)
            dataset = h5[dataset_path]
            if len(dataset.shape) != 2:
                raise ValueError(f"Expected a 2D dataset, got shape={dataset.shape} at {dataset_path}")

            n_channels, n_frames = dataset.shape
            bytes_per_frame = n_channels * dataset.dtype.itemsize
            target_bytes = int(args.target_mb * 1024 * 1024)
            frames_per_segment = max(1, target_bytes // max(1, bytes_per_frame))
            if frames_per_segment >= n_frames:
                raise ValueError(
                    f"frames_per_segment={frames_per_segment} exceeds dataset length n_frames={n_frames}."
                )

            print(f"- dataset_path: {dataset_path}")
            print(f"- shape: {dataset.shape}")
            print(f"- dtype: {dataset.dtype}")
            print(f"- frames_per_segment: {frames_per_segment}")
            print(f"- estimated_segment_bytes: {frames_per_segment * bytes_per_frame}")

            total_bytes = 0
            total_time = 0.0
            for idx in range(1, args.segments + 1):
                start_frame = rng.randrange(0, n_frames - frames_per_segment)
                end_frame = start_frame + frames_per_segment

                t0 = time.perf_counter()
                block = dataset[:, start_frame:end_frame]
                dt = time.perf_counter() - t0

                total_time += dt
                total_bytes += block.nbytes
                checksum = int(block.sum() % 1_000_000_007)
                print(
                    f"- segment {idx}: frames=[{start_frame}, {end_frame}), "
                    f"bytes={block.nbytes}, sec={dt:.4f}, checksum_mod={checksum}"
                )

        print("")
        print(f"Read summary: bytes={total_bytes}, seconds={total_time:.4f}")
        print("")
        print(wrapped.stats_markdown())


if __name__ == "__main__":
    main()
