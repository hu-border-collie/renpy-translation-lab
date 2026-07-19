"""Shared resource limits and validation helpers for local Ren'Py RPA archives."""

from dataclasses import dataclass
import zlib


@dataclass(frozen=True)
class RpaLimits:
    max_compressed_index_bytes: int = 64 * 1024 * 1024
    max_decompressed_index_bytes: int = 256 * 1024 * 1024
    max_entries: int = 200_000
    max_chunks_per_member: int = 65_536
    max_total_chunks: int = 1_000_000
    max_member_output_bytes: int = 512 * 1024 * 1024
    max_total_extraction_bytes: int = 2 * 1024 * 1024 * 1024
    copy_chunk_bytes: int = 1024 * 1024


DEFAULT_RPA_LIMITS = RpaLimits()


class RpaResourceLimitError(RuntimeError):
    pass


@dataclass
class RpaExtractionBudget:
    maximum: int
    used: int = 0

    def reserve(self, amount):
        next_used = self.used + amount
        if next_used > self.maximum:
            _raise_limit("total extraction output", next_used, self.maximum)
        self.used = next_used


def _raise_limit(label, actual, maximum):
    raise RpaResourceLimitError(
        f"RPA resource limit exceeded: {label} is {actual}; maximum is {maximum}."
    )


def read_bounded_compressed_index(infile, offset, archive_size, limits=DEFAULT_RPA_LIMITS):
    if not isinstance(offset, int) or offset < 0 or offset >= archive_size:
        raise RuntimeError(
            f"Invalid RPA index offset {offset!r} for archive size {archive_size}."
        )

    compressed_size = archive_size - offset
    if compressed_size > limits.max_compressed_index_bytes:
        _raise_limit(
            "compressed index",
            compressed_size,
            limits.max_compressed_index_bytes,
        )

    infile.seek(offset)
    compressed = infile.read(compressed_size)
    if len(compressed) != compressed_size:
        raise RuntimeError("Invalid RPA archive: compressed index is truncated.")

    decompressor = zlib.decompressobj()
    try:
        payload = decompressor.decompress(
            compressed,
            limits.max_decompressed_index_bytes + 1,
        )
        if (
            len(payload) > limits.max_decompressed_index_bytes
            or decompressor.unconsumed_tail
        ):
            _raise_limit(
                "decompressed index",
                limits.max_decompressed_index_bytes + 1,
                limits.max_decompressed_index_bytes,
            )
        tail = decompressor.flush(
            max(1, limits.max_decompressed_index_bytes - len(payload) + 1)
        )
    except zlib.error as exc:
        raise RuntimeError(f"Invalid RPA compressed index: {exc}") from exc

    if len(payload) + len(tail) > limits.max_decompressed_index_bytes:
        _raise_limit(
            "decompressed index",
            len(payload) + len(tail),
            limits.max_decompressed_index_bytes,
        )
    if not decompressor.eof:
        raise RuntimeError("Invalid RPA compressed index: incomplete zlib stream.")
    return payload + tail


def _coerce_start_bytes(start):
    if start is None:
        return b""
    if isinstance(start, bytes):
        return start
    return str(start).encode("latin-1", errors="ignore")


def decode_and_validate_index(
    raw_index,
    archive_size,
    *,
    key=None,
    stringify_names=False,
    limits=DEFAULT_RPA_LIMITS,
):
    if not isinstance(raw_index, dict):
        raise RuntimeError("Invalid RPA index: expected a dictionary.")
    if len(raw_index) > limits.max_entries:
        _raise_limit("index entries", len(raw_index), limits.max_entries)

    index = {}
    total_chunks = 0
    for raw_name, raw_chunks in raw_index.items():
        if not isinstance(raw_chunks, (list, tuple)):
            raise RuntimeError("Invalid RPA index: member chunks must be a list or tuple.")
        if len(raw_chunks) > limits.max_chunks_per_member:
            _raise_limit(
                "chunks for one member",
                len(raw_chunks),
                limits.max_chunks_per_member,
            )
        total_chunks += len(raw_chunks)
        if total_chunks > limits.max_total_chunks:
            _raise_limit("total chunks", total_chunks, limits.max_total_chunks)

        decoded_chunks = []
        member_size = 0
        for raw_chunk in raw_chunks:
            if not isinstance(raw_chunk, (list, tuple)) or len(raw_chunk) not in (2, 3):
                raise RuntimeError("Invalid RPA index: each chunk must have two or three fields.")

            chunk_offset = int(raw_chunk[0])
            chunk_len = int(raw_chunk[1])
            if key is not None:
                chunk_offset ^= key
                chunk_len ^= key
            start = _coerce_start_bytes(raw_chunk[2] if len(raw_chunk) == 3 else b"")

            if chunk_offset < 0 or chunk_len < 0:
                raise RuntimeError(
                    f"Invalid RPA chunk: negative offset or length for {raw_name!r}."
                )
            if chunk_offset > archive_size or chunk_len > archive_size - chunk_offset:
                raise RuntimeError(
                    f"Invalid RPA chunk: data for {raw_name!r} falls outside archive."
                )

            member_size += len(start) + chunk_len
            if member_size > limits.max_member_output_bytes:
                _raise_limit(
                    f"member output for {raw_name!r}",
                    member_size,
                    limits.max_member_output_bytes,
                )
            decoded_chunks.append((chunk_offset, chunk_len, start))

        name = str(raw_name) if stringify_names else raw_name
        index[name] = decoded_chunks

    return index


def member_output_size(chunks):
    return sum(len(start) + chunk_len for chunk_offset, chunk_len, start in chunks)


def read_member_bytes(source, chunks, limits=DEFAULT_RPA_LIMITS):
    expected_size = member_output_size(chunks)
    if expected_size > limits.max_member_output_bytes:
        _raise_limit(
            "member output",
            expected_size,
            limits.max_member_output_bytes,
        )

    data = bytearray()
    for chunk_offset, chunk_len, start in chunks:
        data.extend(start)
        source.seek(chunk_offset)
        remaining = chunk_len
        while remaining:
            block = source.read(min(remaining, limits.copy_chunk_bytes))
            if not block:
                raise RuntimeError("Invalid RPA archive: member data is truncated.")
            data.extend(block)
            remaining -= len(block)
    return bytes(data)


def copy_member(source, target, chunks, limits=DEFAULT_RPA_LIMITS):
    for chunk_offset, chunk_len, start in chunks:
        if start:
            target.write(start)
        source.seek(chunk_offset)
        remaining = chunk_len
        while remaining:
            block = source.read(min(remaining, limits.copy_chunk_bytes))
            if not block:
                raise RuntimeError("Invalid RPA archive: member data is truncated.")
            target.write(block)
            remaining -= len(block)
