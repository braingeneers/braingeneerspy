from __future__ import annotations

import io
import os
import time

from braingeneers.utils.range_cache_file import RangeCacheConfig, RangeCacheFile


class LatencyCountingFile(io.RawIOBase):
    def __init__(self, data: bytes, latency_sec: float) -> None:
        self._data = data
        self._pos = 0
        self._latency_sec = latency_sec
        self.read_calls = 0
        self.bytes_read = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            new_pos = offset
        elif whence == io.SEEK_CUR:
            new_pos = self._pos + offset
        elif whence == io.SEEK_END:
            new_pos = len(self._data) + offset
        else:
            raise ValueError(f"Invalid whence={whence}")
        if new_pos < 0:
            raise ValueError("Negative seek")
        self._pos = min(new_pos, len(self._data))
        return self._pos

    def read(self, size: int = -1) -> bytes:
        self.read_calls += 1
        time.sleep(self._latency_sec)
        if size is None or size < 0:
            end = len(self._data)
        else:
            end = min(self._pos + size, len(self._data))
        chunk = self._data[self._pos:end]
        self._pos = end
        self.bytes_read += len(chunk)
        return chunk


def make_data(size: int, seed: int = 42) -> bytes:
    _ = seed
    return os.urandom(size)


def run_baseline(data: bytes, read_size: int, read_count: int, latency_sec: float) -> tuple[float, int, int]:
    backend = LatencyCountingFile(data, latency_sec=latency_sec)
    start = time.perf_counter()
    for _ in range(read_count):
        backend.read(read_size)
    elapsed = time.perf_counter() - start
    return elapsed, backend.read_calls, backend.bytes_read


def run_wrapped(data: bytes, read_size: int, read_count: int, latency_sec: float) -> tuple[float, dict[str, object]]:
    backend = LatencyCountingFile(data, latency_sec=latency_sec)
    wrapped = RangeCacheFile(
        backend,
        config=RangeCacheConfig(
            min_fetch_probe=256 * 1024,
            min_fetch_seq=4 * 1024 * 1024,
            max_fetch=16 * 1024 * 1024,
            alignment=64 * 1024,
            min_reads_for_seq=8,
            min_seq_bytes=256 * 1024,
        ),
    )
    start = time.perf_counter()
    for _ in range(read_count):
        wrapped.read(read_size)
    elapsed = time.perf_counter() - start
    return elapsed, wrapped.stats()


def main() -> None:
    data = make_data(64 * 1024 * 1024)
    read_size = 1024
    read_count = 3000
    latency_sec = 0.002

    baseline_time, baseline_calls, baseline_bytes = run_baseline(
        data=data,
        read_size=read_size,
        read_count=read_count,
        latency_sec=latency_sec,
    )
    wrapped_time, wrapped_stats = run_wrapped(
        data=data,
        read_size=read_size,
        read_count=read_count,
        latency_sec=latency_sec,
    )

    print("Local S3 streaming emulation results")
    print(f"- baseline_time_sec: {baseline_time:.4f}")
    print(f"- baseline_read_calls: {baseline_calls}")
    print(f"- baseline_bytes_read: {baseline_bytes}")
    print(f"- wrapped_time_sec: {wrapped_time:.4f}")
    print(f"- wrapped_underlying_read_calls: {wrapped_stats['underlying_read_calls']}")
    print(f"- wrapped_underlying_bytes_fetched: {wrapped_stats['underlying_bytes_fetched']}")
    print(f"- wrapped_mode: {wrapped_stats['mode']}")
    print(f"- wrapped_hit_rate: {wrapped_stats['hit_rate']:.4f}")
    print(f"- wrapped_byte_hit_rate: {wrapped_stats['byte_hit_rate']:.4f}")
    print(f"- wrapped_amplification: {wrapped_stats['amplification']:.4f}")


if __name__ == "__main__":
    main()
