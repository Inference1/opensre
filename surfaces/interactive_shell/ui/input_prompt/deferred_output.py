"""Disk-backed, ordered records for output held behind a full-screen form."""

from __future__ import annotations

import struct
import tempfile
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from typing import BinaryIO, cast

from config.constants.terminal_host import DEFERRED_OUTPUT_CHUNK_CHARS


@contextmanager
def _temporary_output_file() -> Iterator[BinaryIO]:
    with tempfile.TemporaryFile(mode="w+b", buffering=0) as output:
        yield cast(BinaryIO, output)


class DeferredOutput:
    """Keep captured output on disk and replay bounded text or control records."""

    def __init__(self) -> None:
        self._resources = ExitStack()
        self._file = self._resources.enter_context(_temporary_output_file())

    def write(self, text: str) -> None:
        """Append independent UTF-8 records without retaining the caller's text."""
        for start in range(0, len(text), DEFERRED_OUTPUT_CHUNK_CHARS):
            payload = text[start : start + DEFERRED_OUTPUT_CHUNK_CHARS].encode(
                "utf-8", errors="surrogatepass"
            )
            position = self._file.tell()
            try:
                self._write_all(struct.pack("!I", len(payload)))
                self._write_all(payload)
            except OSError:
                # A failed record was never accepted; retain the valid prefix.
                self._file.seek(position)
                self._file.truncate()
                raise

    def _write_all(self, payload: bytes) -> None:
        remaining = memoryview(payload)
        while remaining:
            written = self._file.write(remaining)
            if written is None or written <= 0:
                raise OSError("Deferred output write made no progress")
            remaining = remaining[written:]

    def __iter__(self) -> Iterator[str]:
        self._file.seek(0)
        while header := self._file.read(4):
            if len(header) != 4:
                raise OSError("Invalid deferred output record")
            size = struct.unpack("!I", header)[0]
            if size > DEFERRED_OUTPUT_CHUNK_CHARS * 4:
                raise OSError("Oversized deferred output record")
            payload = self._file.read(size)
            if len(payload) != size:
                raise OSError("Incomplete deferred output record")
            yield payload.decode("utf-8", errors="surrogatepass")

    def close(self) -> None:
        """Release the anonymous temporary file on every completion path."""
        self._resources.close()


__all__ = ["DeferredOutput"]
