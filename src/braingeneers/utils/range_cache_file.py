from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from enum import Enum
import io
import threading
import time
from typing import Any


def _align_down(value: int, alignment: int) -> int:
    if alignment <= 1:
        return value
    return value - (value % alignment)


def _align_up(value: int, alignment: int) -> int:
    if alignment <= 1:
        return value
    return ((value + alignment - 1) // alignment) * alignment


class Mode(str, Enum):
    PROBE = "probe"
    SEQ = "seq"
    RAND = "rand"


@dataclass
class _RangeCacheConfig:
    max_cache_bytes: int = 512 * 1024 * 1024
    merge_gap_bytes: int = 128 * 1024

    alignment: int = 256 * 1024

    probe_fetch_min: int = 256 * 1024
    probe_fetch_max: int = 8 * 1024 * 1024
    probe_growth_factor: float = 1.4
    probe_decay_factor: float = 0.85

    seq_fetch_min: int = 4 * 1024 * 1024
    seq_fetch_max: int = 64 * 1024 * 1024
    seq_read_ahead_multiplier: float = 4.0

    window_size: int = 32
    gap_threshold: int = 64 * 1024
    small_back_threshold: int = 64 * 1024
    jump_threshold: int = 4 * 1024 * 1024
    min_reads_for_seq: int = 16
    min_seq_bytes: int = 2 * 1024 * 1024
    promote_score_threshold: float = 0.75
    promote_big_jump_rate_max: float = 0.10
    demote_score_threshold: float = 0.40
    demote_big_jumps_recent: int = 2
    demote_recent_window: int = 8

    tune_every_reads: int = 32
    seq_fetch_growth_factor: float = 1.5
    seq_fetch_decay_factor: float = 0.7
    seq_target_byte_hit_rate: float = 0.98
    seq_max_amplification: float = 8.0


@dataclass
class _Segment:
    start: int
    data: bytes
    last_access: int

    @property
    def end(self) -> int:
        return self.start + len(self.data)

    @property
    def length(self) -> int:
        return len(self.data)


class _SegmentCache:
    def __init__(self, max_cache_bytes: int) -> None:
        self._max_cache_bytes = max_cache_bytes
        self._segments: list[_Segment] = []
        self._total_bytes = 0
        self._evictions = 0
        self._access_counter = 0

    def _next_access(self) -> int:
        self._access_counter += 1
        return self._access_counter

    def _touch(self, segment: _Segment) -> None:
        segment.last_access = self._next_access()

    @staticmethod
    def _merge_two(left: _Segment, right: _Segment, access: int) -> _Segment:
        union_start = min(left.start, right.start)
        union_end = max(left.end, right.end)
        merged = bytearray(union_end - union_start)

        left_offset = left.start - union_start
        merged[left_offset:left_offset + left.length] = left.data

        right_offset = right.start - union_start
        merged[right_offset:right_offset + right.length] = right.data

        return _Segment(
            start=union_start,
            data=bytes(merged),
            last_access=max(access, left.last_access, right.last_access),
        )

    def _recompute_bytes(self) -> None:
        self._total_bytes = sum(seg.length for seg in self._segments)

    def _coalesce(self) -> None:
        if not self._segments:
            return
        self._segments.sort(key=lambda seg: seg.start)
        merged: list[_Segment] = [self._segments[0]]

        for segment in self._segments[1:]:
            current = merged[-1]
            if segment.start < current.end:
                merged[-1] = self._merge_two(current, segment, self._next_access())
            elif segment.start == current.end:
                merged[-1] = _Segment(
                    start=current.start,
                    data=current.data + segment.data,
                    last_access=max(current.last_access, segment.last_access, self._next_access()),
                )
            else:
                merged.append(segment)

        self._segments = merged
        self._recompute_bytes()

    def _evict_if_needed(self) -> None:
        while self._total_bytes > self._max_cache_bytes and len(self._segments) > 1:
            lru_index = min(range(len(self._segments)), key=lambda idx: self._segments[idx].last_access)
            evicted = self._segments.pop(lru_index)
            self._total_bytes -= evicted.length
            self._evictions += 1

    def put(self, start: int, data: bytes) -> None:
        if not data:
            return
        self._segments.append(_Segment(start=start, data=data, last_access=self._next_access()))
        self._coalesce()
        self._evict_if_needed()

    def get_prefix(self, start: int, max_len: int) -> bytes:
        if max_len <= 0 or not self._segments:
            return b""

        out = bytearray()
        cursor = start
        remaining = max_len

        while remaining > 0:
            found = False
            for segment in self._segments:
                if segment.start <= cursor < segment.end:
                    segment_offset = cursor - segment.start
                    take = min(remaining, segment.end - cursor)
                    out.extend(segment.data[segment_offset:segment_offset + take])
                    cursor += take
                    remaining -= take
                    self._touch(segment)
                    found = True
                    break
            if not found:
                break

        return bytes(out)

    def previous_segment_end(self, pos: int) -> int | None:
        best: int | None = None
        for segment in self._segments:
            if segment.end <= pos and (best is None or segment.end > best):
                best = segment.end
        return best

    def next_segment_start(self, pos: int) -> int | None:
        best: int | None = None
        for segment in self._segments:
            if segment.start >= pos and (best is None or segment.start < best):
                best = segment.start
        return best

    def stats(self) -> dict[str, int]:
        return {
            "cache_bytes_current": self._total_bytes,
            "cache_segments": len(self._segments),
            "cache_evictions": self._evictions,
        }


@dataclass
class _ReadSample:
    delta: int | None
    contiguity: float
    big_jump: bool
    length: int


class _AccessPatternTracker:
    def __init__(self, config: _RangeCacheConfig) -> None:
        self._config = config
        self._samples: deque[_ReadSample] = deque()
        self._contiguity_sum = 0.0
        self._big_jump_count = 0
        self._bytes_total = 0
        self._previous_end: int | None = None

    def _score_delta(self, delta: int | None) -> float:
        if delta is None:
            return 0.0
        if delta == 0:
            return 1.0
        if 0 < delta <= self._config.gap_threshold:
            return 0.5
        if delta < 0 and abs(delta) <= self._config.small_back_threshold:
            return 0.25
        return 0.0

    def record_read(self, start: int, length: int) -> None:
        if length <= 0:
            return

        delta = None if self._previous_end is None else start - self._previous_end
        sample = _ReadSample(
            delta=delta,
            contiguity=self._score_delta(delta),
            big_jump=bool(delta is not None and abs(delta) > self._config.jump_threshold),
            length=length,
        )

        if len(self._samples) >= self._config.window_size:
            old = self._samples.popleft()
            self._contiguity_sum -= old.contiguity
            if old.big_jump:
                self._big_jump_count -= 1

        self._samples.append(sample)
        self._contiguity_sum += sample.contiguity
        self._bytes_total += sample.length
        if sample.big_jump:
            self._big_jump_count += 1

        self._previous_end = start + length

    def sample_count(self) -> int:
        return len(self._samples)

    def contiguity_score(self) -> float:
        if not self._samples:
            return 0.0
        return self._contiguity_sum / len(self._samples)

    def big_jump_rate(self) -> float:
        if not self._samples:
            return 0.0
        return self._big_jump_count / len(self._samples)

    def bytes_total(self) -> int:
        return self._bytes_total

    def recent_big_jumps(self, recent_window: int) -> int:
        if recent_window <= 0:
            return 0
        recent = list(self._samples)[-recent_window:]
        return sum(1 for sample in recent if sample.big_jump)

    def should_promote_to_seq(self) -> bool:
        return (
            self.sample_count() >= self._config.min_reads_for_seq
            and self.bytes_total() >= self._config.min_seq_bytes
            and self.contiguity_score() >= self._config.promote_score_threshold
            and self.big_jump_rate() <= self._config.promote_big_jump_rate_max
        )

    def should_demote_from_seq(self) -> bool:
        return (
            self.contiguity_score() < self._config.demote_score_threshold
            or self.recent_big_jumps(self._config.demote_recent_window) >= self._config.demote_big_jumps_recent
        )


class RangeCacheFile(io.RawIOBase):
    def __init__(self, file_obj: Any) -> None:
        super().__init__()
        self._f = file_obj
        self._config = _RangeCacheConfig()
        self._lock = threading.RLock()

        self._cache = _SegmentCache(max_cache_bytes=self._config.max_cache_bytes)
        self._tracker = _AccessPatternTracker(self._config)
        self._mode = Mode.PROBE
        self._mode_transitions: dict[Mode, int] = {mode: 0 for mode in Mode}

        self._probe_fetch_current = self._config.probe_fetch_min
        self._seq_fetch_current = self._config.seq_fetch_min
        self._probe_miss_count = 0

        self._known_size: int | None = None
        self._pos = 0

        self._read_calls = 0
        self._seek_calls = 0
        self._underlying_read_calls = 0
        self._underlying_bytes_fetched = 0
        self._bytes_served_to_caller = 0
        self._bytes_served_from_cache = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._time_underlying_read_sec = 0.0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._f, name)

    @property
    def mode(self) -> Mode:
        return self._mode

    @property
    def name(self) -> str | None:
        return getattr(self._f, "name", None)

    def _discover_size(self) -> None:
        if self._known_size is not None:
            return
        try:
            self._f.seek(0, io.SEEK_END)
            size = self._f.tell()
            if isinstance(size, int) and size >= 0:
                self._known_size = size
        except Exception:
            self._known_size = None

    def _set_mode(self, new_mode: Mode) -> None:
        if new_mode == self._mode:
            return
        self._mode = new_mode
        self._mode_transitions[new_mode] += 1
        if new_mode == Mode.SEQ:
            self._seq_fetch_current = max(self._seq_fetch_current, self._probe_fetch_current * 2, self._config.seq_fetch_min)

    def _update_mode(self) -> None:
        if self._mode == Mode.SEQ:
            if self._tracker.should_demote_from_seq():
                self._set_mode(Mode.PROBE)
        elif self._tracker.should_promote_to_seq():
            self._set_mode(Mode.SEQ)

    def _current_fetch_floor(self) -> int:
        if self._mode == Mode.SEQ:
            return min(max(self._seq_fetch_current, self._config.seq_fetch_min), self._config.seq_fetch_max)
        return min(max(self._probe_fetch_current, self._config.probe_fetch_min), self._config.probe_fetch_max)

    def _current_stream_chunk(self) -> int:
        return max(self._config.alignment, self._current_fetch_floor())

    def _compute_fetch_range(self, pos: int, requested_len: int) -> tuple[int, int]:
        req = max(1, requested_len)
        alignment = min(max(1, self._config.alignment), self._config.seq_fetch_max)

        if self._mode == Mode.SEQ:
            target = int(max(self._seq_fetch_current, req * self._config.seq_read_ahead_multiplier))
            target = min(max(target, self._config.seq_fetch_min), self._config.seq_fetch_max)
            cap = self._config.seq_fetch_max
        else:
            target = max(req, self._probe_fetch_current)
            target = min(max(target, self._config.probe_fetch_min), self._config.probe_fetch_max)
            cap = self._config.probe_fetch_max

        fetch_start = _align_down(pos, alignment)
        fetch_end = _align_up(fetch_start + target, alignment)

        left_end = self._cache.previous_segment_end(fetch_start)
        if left_end is not None and 0 < fetch_start - left_end <= self._config.merge_gap_bytes:
            fetch_start = left_end

        right_start = self._cache.next_segment_start(fetch_end)
        if right_start is not None and 0 < right_start - fetch_end <= self._config.merge_gap_bytes:
            fetch_end = right_start

        if fetch_end - fetch_start > cap:
            fetch_end = fetch_start + cap

        if self._known_size is not None:
            fetch_end = min(fetch_end, self._known_size)

        if pos >= fetch_end:
            fetch_start = pos
            fetch_end = pos + max(1, min(req, cap))
            if self._known_size is not None:
                fetch_end = min(fetch_end, self._known_size)

        if fetch_end <= fetch_start:
            fetch_end = fetch_start + max(1, min(req, cap))
            if self._known_size is not None:
                fetch_end = min(fetch_end, self._known_size)

        return fetch_start, fetch_end

    def _fetch_from_underlying(self, start: int, length: int) -> bytes:
        if length <= 0:
            return b""

        start_time = time.perf_counter()
        self._f.seek(start)
        raw = self._f.read(length)
        elapsed = time.perf_counter() - start_time

        self._time_underlying_read_sec += elapsed
        self._underlying_read_calls += 1

        if raw is None:
            data = b""
        elif isinstance(raw, str):
            raise TypeError("RangeCacheFile requires a binary file-like object.")
        else:
            data = bytes(raw)

        self._underlying_bytes_fetched += len(data)
        if len(data) < length:
            self._known_size = start + len(data)

        return data

    def _maybe_tune_probe_fetch(self) -> None:
        if self._mode != Mode.PROBE:
            return
        if self._probe_miss_count == 0 or self._probe_miss_count % 4 != 0:
            return

        score = self._tracker.contiguity_score()
        jump_rate = self._tracker.big_jump_rate()

        if score >= 0.35 and jump_rate <= 0.25:
            grown = int(self._probe_fetch_current * self._config.probe_growth_factor)
            self._probe_fetch_current = min(max(grown, self._config.probe_fetch_min), self._config.probe_fetch_max)
        elif score < 0.15:
            shrunk = int(self._probe_fetch_current * self._config.probe_decay_factor)
            self._probe_fetch_current = max(self._config.probe_fetch_min, shrunk)

    def _maybe_tune_seq_fetch(self) -> None:
        if self._mode != Mode.SEQ:
            return
        if self._read_calls % self._config.tune_every_reads != 0:
            return
        if self._bytes_served_to_caller == 0:
            return

        byte_hit_rate = self._bytes_served_from_cache / self._bytes_served_to_caller
        amplification = self._underlying_bytes_fetched / max(self._bytes_served_to_caller, 1)

        if byte_hit_rate >= self._config.seq_target_byte_hit_rate and amplification <= self._config.seq_max_amplification:
            grown = int(self._seq_fetch_current * self._config.seq_fetch_growth_factor)
            self._seq_fetch_current = min(max(grown, self._config.seq_fetch_min), self._config.seq_fetch_max)
        elif amplification > self._config.seq_max_amplification:
            shrunk = int(self._seq_fetch_current * self._config.seq_fetch_decay_factor)
            self._seq_fetch_current = max(self._config.seq_fetch_min, shrunk)

    def read(self, size: int = -1) -> bytes:
        with self._lock:
            self._checkClosed()
            self._read_calls += 1

            if size == 0:
                return b""

            read_start = self._pos
            remaining: int | None = None if size < 0 else size

            if remaining is None:
                self._discover_size()
                if self._known_size is not None:
                    remaining = max(0, self._known_size - self._pos)

            out = bytearray()
            while remaining is None or remaining > 0:
                if self._known_size is not None and self._pos >= self._known_size:
                    break

                request_len = remaining if remaining is not None else self._current_stream_chunk()
                if request_len <= 0:
                    break

                cached = self._cache.get_prefix(self._pos, request_len)
                if cached:
                    self._cache_hits += 1
                    self._bytes_served_from_cache += len(cached)
                    out.extend(cached)
                    self._pos += len(cached)
                    if remaining is not None:
                        remaining -= len(cached)
                    continue

                self._cache_misses += 1
                if self._mode == Mode.PROBE:
                    self._probe_miss_count += 1

                fetch_start, fetch_end = self._compute_fetch_range(self._pos, request_len)
                fetched = self._fetch_from_underlying(fetch_start, fetch_end - fetch_start)
                if not fetched:
                    if self._known_size is None:
                        self._known_size = self._pos
                    break

                self._cache.put(fetch_start, fetched)
                offset = self._pos - fetch_start
                if offset < 0 or offset >= len(fetched):
                    break

                take = len(fetched) - offset
                if remaining is not None:
                    take = min(take, remaining)
                if take <= 0:
                    break

                chunk = fetched[offset:offset + take]
                out.extend(chunk)
                self._pos += len(chunk)
                if remaining is not None:
                    remaining -= len(chunk)

            result = bytes(out)
            self._bytes_served_to_caller += len(result)
            if result:
                self._tracker.record_read(read_start, len(result))
                self._update_mode()
                self._maybe_tune_probe_fetch()
                self._maybe_tune_seq_fetch()
            return result

    def readinto(self, b: bytearray | memoryview) -> int:
        with self._lock:
            self._checkClosed()
            data = self.read(len(b))
            n_bytes = len(data)
            b[:n_bytes] = data
            return n_bytes

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        with self._lock:
            self._checkClosed()
            self._seek_calls += 1

            if whence == io.SEEK_SET:
                new_pos = offset
            elif whence == io.SEEK_CUR:
                new_pos = self._pos + offset
            elif whence == io.SEEK_END:
                self._discover_size()
                if self._known_size is None:
                    raise OSError("Unable to seek from end: wrapped stream size is unknown.")
                new_pos = self._known_size + offset
            else:
                raise ValueError(f"Invalid whence={whence}")

            if new_pos < 0:
                raise ValueError("Negative seek position.")

            self._pos = new_pos
            return self._pos

    def tell(self) -> int:
        with self._lock:
            self._checkClosed()
            return self._pos

    def close(self) -> None:
        with self._lock:
            if self.closed:
                return
            try:
                self._f.close()
            finally:
                super().close()

    def flush(self) -> None:
        return None

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def stats(self) -> dict[str, Any]:
        with self._lock:
            cache_stats = self._cache.stats()
            total_lookup_ops = self._cache_hits + self._cache_misses

            hit_rate = self._cache_hits / total_lookup_ops if total_lookup_ops else 0.0
            byte_hit_rate = (
                self._bytes_served_from_cache / self._bytes_served_to_caller
                if self._bytes_served_to_caller
                else 0.0
            )
            amplification = (
                self._underlying_bytes_fetched / self._bytes_served_to_caller
                if self._bytes_served_to_caller
                else 0.0
            )

            stats = {
                "mode": self._mode.value,
                "position": self._pos,
                "read_calls": self._read_calls,
                "seek_calls": self._seek_calls,
                "underlying_read_calls": self._underlying_read_calls,
                "underlying_bytes_fetched": self._underlying_bytes_fetched,
                "bytes_served_to_caller": self._bytes_served_to_caller,
                "bytes_served_from_cache": self._bytes_served_from_cache,
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "hit_rate": hit_rate,
                "byte_hit_rate": byte_hit_rate,
                "amplification": amplification,
                "avg_underlying_read_size": (
                    self._underlying_bytes_fetched / self._underlying_read_calls
                    if self._underlying_read_calls
                    else 0.0
                ),
                "avg_wrapper_read_size": (
                    self._bytes_served_to_caller / self._read_calls if self._read_calls else 0.0
                ),
                "time_underlying_read_sec": self._time_underlying_read_sec,
                "mode_transitions": {mode.value: count for mode, count in self._mode_transitions.items()},
                "probe_fetch_current": self._probe_fetch_current,
                "seq_fetch_current": self._seq_fetch_current,
                "known_size": self._known_size,
                "internal_tuning": asdict(self._config),
            }
            stats.update(cache_stats)
            return stats

    def stats_markdown(self) -> str:
        metrics = self.stats()
        rows: list[tuple[str, str]] = [
            ("Mode", str(metrics["mode"])),
            ("Read Calls", str(metrics["read_calls"])),
            ("Seek Calls", str(metrics["seek_calls"])),
            ("Underlying Read Calls", str(metrics["underlying_read_calls"])),
            ("Hit Rate", f"{metrics['hit_rate']:.4f}"),
            ("Byte Hit Rate", f"{metrics['byte_hit_rate']:.4f}"),
            ("Amplification", f"{metrics['amplification']:.4f}"),
            ("Bytes Served (MiB)", f"{metrics['bytes_served_to_caller'] / (1024 * 1024):.4f}"),
            ("Bytes Fetched (MiB)", f"{metrics['underlying_bytes_fetched'] / (1024 * 1024):.4f}"),
            ("Avg Underlying Read (KiB)", f"{metrics['avg_underlying_read_size'] / 1024:.2f}"),
            ("Avg Wrapper Read (KiB)", f"{metrics['avg_wrapper_read_size'] / 1024:.2f}"),
            ("Cache Bytes Current (MiB)", f"{metrics['cache_bytes_current'] / (1024 * 1024):.4f}"),
            ("Cache Segments", str(metrics["cache_segments"])),
            ("Cache Evictions", str(metrics["cache_evictions"])),
            ("Underlying Read Time (s)", f"{metrics['time_underlying_read_sec']:.6f}"),
            ("Probe Fetch Current (MiB)", f"{metrics['probe_fetch_current'] / (1024 * 1024):.4f}"),
            ("SEQ Fetch Current (MiB)", f"{metrics['seq_fetch_current'] / (1024 * 1024):.4f}"),
        ]

        table_lines = ["| Metric | Value |", "|---|---|"]
        table_lines.extend(f"| {metric} | {value} |" for metric, value in rows)
        return "\n".join(table_lines)
