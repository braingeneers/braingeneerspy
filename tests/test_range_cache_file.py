from __future__ import annotations

import io
import random
import time

from braingeneers.utils.range_cache_file import Mode, RangeCacheFile


class CountingFile(io.RawIOBase):
    def __init__(self, data: bytes, latency_sec: float = 0.0) -> None:
        self._data = data
        self._pos = 0
        self._closed = False
        self._latency_sec = latency_sec

        self.seek_calls = 0
        self.read_calls = 0
        self.bytes_read = 0

    def close(self) -> None:
        self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        self.seek_calls += 1
        if whence == io.SEEK_SET:
            new_pos = offset
        elif whence == io.SEEK_CUR:
            new_pos = self._pos + offset
        elif whence == io.SEEK_END:
            new_pos = len(self._data) + offset
        else:
            raise ValueError(f"Invalid whence: {whence}")
        if new_pos < 0:
            raise ValueError("Negative seek")
        self._pos = min(new_pos, len(self._data))
        return self._pos

    def read(self, size: int = -1) -> bytes:
        self.read_calls += 1
        if self._latency_sec > 0:
            time.sleep(self._latency_sec)
        if size is None or size < 0:
            end = len(self._data)
        else:
            end = min(self._pos + size, len(self._data))
        chunk = self._data[self._pos:end]
        self._pos = end
        self.bytes_read += len(chunk)
        return chunk


def _make_data(size: int, seed: int = 7) -> bytes:
    generator = random.Random(seed)
    return bytes(generator.randrange(0, 256) for _ in range(size))


def _speed_up_seq_promotion(wrapped: RangeCacheFile) -> None:
    wrapped._config.min_reads_for_seq = 8
    wrapped._config.min_seq_bytes = 64 * 1024
    wrapped._config.seq_fetch_min = 2 * 1024 * 1024
    wrapped._config.probe_fetch_min = 64 * 1024
    wrapped._config.probe_fetch_max = 2 * 1024 * 1024
    wrapped._config.alignment = 64 * 1024
    wrapped._probe_fetch_current = wrapped._config.probe_fetch_min
    wrapped._seq_fetch_current = wrapped._config.seq_fetch_min


def test_correctness_mixed_random_access_and_readinto() -> None:
    data = _make_data(1_500_000, seed=11)
    baseline = io.BytesIO(data)
    wrapped_backend = CountingFile(data)
    wrapped = RangeCacheFile(wrapped_backend)

    generator = random.Random(1234)
    for _ in range(250):
        seek_to = generator.randrange(0, len(data) - 4096)
        size = generator.randrange(0, 4096)

        baseline.seek(seek_to)
        expected = baseline.read(size)

        wrapped.seek(seek_to)
        got = wrapped.read(size)
        assert got == expected

    wrapped.seek(12_345)
    baseline.seek(12_345)
    target = bytearray(2048)
    n_read = wrapped.readinto(target)
    assert n_read == 2048
    assert bytes(target[:n_read]) == baseline.read(2048)

    wrapped.seek(1_400_000)
    baseline.seek(1_400_000)
    assert wrapped.read() == baseline.read()


def test_sequential_pattern_reduces_underlying_calls() -> None:
    data = _make_data(8 * 1024 * 1024, seed=99)
    read_size = 1024

    baseline_backend = CountingFile(data)
    baseline_out = bytearray()
    baseline_backend.seek(0)
    for _ in range(4096):
        baseline_out.extend(baseline_backend.read(read_size))

    wrapped_backend = CountingFile(data)
    wrapped = RangeCacheFile(wrapped_backend)
    _speed_up_seq_promotion(wrapped)

    wrapped_out = bytearray()
    wrapped.seek(0)
    for _ in range(4096):
        wrapped_out.extend(wrapped.read(read_size))

    assert bytes(wrapped_out) == bytes(baseline_out)
    assert wrapped_backend.read_calls < baseline_backend.read_calls / 20

    stats = wrapped.stats()
    assert stats["mode"] == Mode.SEQ.value
    assert stats["underlying_read_calls"] == wrapped_backend.read_calls


def test_random_workload_has_bounded_amplification() -> None:
    data = _make_data(6 * 1024 * 1024, seed=123)
    backend = CountingFile(data)
    wrapped = RangeCacheFile(backend)

    wrapped._config.probe_fetch_max = 2 * 1024 * 1024
    wrapped._config.seq_fetch_max = 4 * 1024 * 1024
    wrapped._config.alignment = 32 * 1024

    baseline = io.BytesIO(data)
    generator = random.Random(2026)
    for _ in range(220):
        offset = generator.randrange(0, len(data) - 8192)
        size = generator.randrange(256, 8192)

        baseline.seek(offset)
        expected = baseline.read(size)

        wrapped.seek(offset)
        got = wrapped.read(size)
        assert got == expected

    stats = wrapped.stats()
    assert stats["bytes_served_to_caller"] > 0
    assert stats["amplification"] <= 10.0


def test_mode_transitions_probe_to_seq_then_demote() -> None:
    data = _make_data(12 * 1024 * 1024, seed=314)
    backend = CountingFile(data)
    wrapped = RangeCacheFile(backend)

    _speed_up_seq_promotion(wrapped)
    wrapped._config.jump_threshold = 256 * 1024
    wrapped._config.demote_big_jumps_recent = 2
    wrapped._config.demote_recent_window = 6

    metadata_offsets = [1024, 333_000, 77_000, 1_100_000, 90_000]
    for offset in metadata_offsets:
        wrapped.seek(offset)
        wrapped.read(2048)

    wrapped.seek(2 * 1024 * 1024)
    for _ in range(640):
        wrapped.read(4096)

    assert wrapped.mode == Mode.SEQ

    for offset in [200_000, 7_000_000, 350_000, 9_000_000]:
        wrapped.seek(offset)
        wrapped.read(1024)

    assert wrapped.mode == Mode.PROBE


def test_cache_eviction_and_stats_markdown() -> None:
    data = _make_data(2 * 1024 * 1024, seed=808)
    backend = CountingFile(data)
    wrapped = RangeCacheFile(backend)

    wrapped._config.max_cache_bytes = 96 * 1024
    wrapped._cache._max_cache_bytes = 96 * 1024
    wrapped._config.probe_fetch_min = 64 * 1024
    wrapped._config.probe_fetch_max = 128 * 1024
    wrapped._probe_fetch_current = wrapped._config.probe_fetch_min

    offsets = [0, 200_000, 400_000, 700_000, 1_100_000, 1_500_000]
    baseline = io.BytesIO(data)
    for offset in offsets:
        baseline.seek(offset)
        expected = baseline.read(8192)
        wrapped.seek(offset)
        assert wrapped.read(8192) == expected

    stats = wrapped.stats()
    assert stats["cache_evictions"] > 0

    markdown = wrapped.stats_markdown()
    assert "| Metric | Value |" in markdown
    assert "Cache Evictions" in markdown
