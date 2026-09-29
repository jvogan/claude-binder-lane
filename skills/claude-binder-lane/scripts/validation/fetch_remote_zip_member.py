#!/usr/bin/env python3
"""Extract named members from a remote ZIP64 archive using HTTP range requests.

The archive this was written for is 74 GB, so downloading it whole is not an
option. A ZIP stores its index at the end of the file, and every member records
its own byte offset, so a client that can ask for byte ranges can pull one
member without touching the rest.

The read path is three steps.

1. Read the tail of the file. Find the end of central directory record, then the
   ZIP64 locator that sits in front of it, then the ZIP64 end of central
   directory record the locator points at. That record carries the offset and
   the size of the central directory.
2. Read the central directory in one range request and parse every entry. Each
   entry gives a member name, a compression method, a compressed size and the
   offset of that member's local file header. Cache the parsed index on disk so
   later runs skip this step.
3. For each requested member, read the local file header and the compressed
   bytes in one range request, then inflate.

`zipfile` cannot do this over HTTP because it needs a seekable file object with
the whole archive behind it. The range reads here are written by hand. The
structure layouts and the decompression come from the standard library.

Standard library only. No third-party packages.

Every byte read from a response body is counted. The counter is the point of
the script, so it is measured rather than estimated.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import struct
import sys
import threading
import time
import urllib.error
import urllib.request
import zlib

# Signatures, little endian, as they appear on the wire.
SIG_EOCD = b"PK\x05\x06"  # end of central directory record
SIG_EOCD64_LOCATOR = b"PK\x06\x07"  # ZIP64 end of central directory locator
SIG_EOCD64 = b"PK\x06\x06"  # ZIP64 end of central directory record
SIG_CENTRAL = 0x02014B50  # central directory file header
SIG_LOCAL = 0x04034B50  # local file header

EOCD_SIZE = 22
EOCD64_LOCATOR_SIZE = 20
EOCD64_MIN_SIZE = 56
CENTRAL_FIXED_SIZE = 46
LOCAL_FIXED_SIZE = 30

# A field set to all ones means the real value lives in the ZIP64 extra field.
U16_MAX = 0xFFFF
U32_MAX = 0xFFFFFFFF

CENTRAL_STRUCT = struct.Struct("<IHHHHHHIIIHHHHHII")
LOCAL_STRUCT = struct.Struct("<IHHHHHIIIHH")

# Extra bytes read past the local file header name so the header and the member
# data arrive in one request. A local extra field larger than this costs one
# more request, which the code handles.
LOCAL_EXTRA_SLACK = 512

USER_AGENT = "fetch_remote_zip_member/1.0 (stdlib urllib)"


class TransferCounter:
    """Counts what actually came back over the wire.

    Several worker threads report into one counter, so the update is locked.
    `seconds` adds up time spent inside requests, which runs ahead of wall clock
    once more than one worker is running.
    """

    def __init__(self) -> None:
        self.body_bytes = 0
        self.requests = 0
        self.seconds = 0.0
        self._lock = threading.Lock()

    def record(self, nbytes: int, seconds: float) -> None:
        with self._lock:
            self.body_bytes += nbytes
            self.requests += 1
            self.seconds += seconds

    def as_dict(self) -> dict:
        return {
            "body_bytes": self.body_bytes,
            "requests": self.requests,
            "seconds": round(self.seconds, 3),
        }


class RemoteFile:
    """A read-only view of a remote file addressed by byte range.

    The host redirects to a CDN, so the first request resolves the final URL and
    later requests go straight there. A signed CDN URL can expire, so an
    authorization failure sends the next request back through the original URL.
    """

    def __init__(self, url: str, counter: TransferCounter) -> None:
        self.url = url
        self.resolved_url = url
        self.counter = counter
        self._size: int | None = None
        self._lock = threading.Lock()

    def _open(self, url: str, start: int, end: int):
        request = urllib.request.Request(
            url,
            headers={"Range": f"bytes={start}-{end}", "User-Agent": USER_AGENT},
        )
        return urllib.request.urlopen(request, timeout=120)

    def read_range(self, start: int, length: int) -> bytes:
        """Reads `length` bytes beginning at `start`. Both are absolute."""
        if length <= 0:
            return b""
        end = start + length - 1
        began = time.monotonic()
        try:
            response = self._open(self.resolved_url, start, end)
        except urllib.error.HTTPError as error:
            if self.resolved_url == self.url or error.code not in (401, 403, 404):
                raise
            # The cached CDN URL went stale. Go back to the canonical one.
            self.resolved_url = self.url
            response = self._open(self.resolved_url, start, end)

        with response:
            status = response.status
            content_range = response.headers.get("Content-Range")
            data = response.read()
            final_url = response.geturl()

        with self._lock:
            self.resolved_url = final_url
            if content_range and self._size is None:
                total = content_range.rsplit("/", 1)[-1].strip()
                if total.isdigit():
                    self._size = int(total)
        self.counter.record(len(data), time.monotonic() - began)

        if status != 206:
            raise RuntimeError(
                f"server did not honour the range request, status {status}"
            )
        if len(data) != length:
            raise RuntimeError(
                f"asked for {length} bytes at {start}, received {len(data)}"
            )
        return data

    def read_tail(self, length: int) -> tuple[bytes, int]:
        """Reads the last `length` bytes. Returns the data and its start offset.

        The size of the file is unknown before the first request, so this uses an
        open-ended suffix range and reads the total out of the Content-Range
        header the server sends back.
        """
        began = time.monotonic()
        request = urllib.request.Request(
            self.resolved_url,
            headers={"Range": f"bytes=-{length}", "User-Agent": USER_AGENT},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            status = response.status
            content_range = response.headers.get("Content-Range")
            data = response.read()
            self.resolved_url = response.geturl()
        self.counter.record(len(data), time.monotonic() - began)

        if status != 206 or not content_range:
            raise RuntimeError(
                f"server did not honour the suffix range request, status {status}"
            )
        # Content-Range: bytes <first>-<last>/<total>
        span, _, total = content_range.partition("bytes ")[2].partition("/")
        first = int(span.split("-")[0])
        self._size = int(total)
        return data, first

    @property
    def size(self) -> int:
        if self._size is None:
            raise RuntimeError("file size is not known until after the first read")
        return self._size

    def seed_size(self, nbytes: int) -> None:
        """Sets the size from a cached index so a cache hit costs no request."""
        if self._size is None:
            self._size = nbytes


def parse_zip64_extra(extra: bytes, needs: list[str]) -> dict:
    """Pulls the fields the base header could not hold out of the extra field.

    ZIP64 writes only the fields that overflowed, in a fixed order, so the caller
    passes which ones to expect.
    """
    values: dict = {}
    position = 0
    while position + 4 <= len(extra):
        header_id, data_size = struct.unpack_from("<HH", extra, position)
        position += 4
        if header_id != 0x0001:
            position += data_size
            continue
        block = extra[position : position + data_size]
        cursor = 0
        for name in needs:
            width = 4 if name == "disk_start" else 8
            if cursor + width > len(block):
                break
            fmt = "<I" if width == 4 else "<Q"
            values[name] = struct.unpack_from(fmt, block, cursor)[0]
            cursor += width
        break
    return values


def read_end_of_central_directory(remote: RemoteFile, tail_bytes: int = 65536) -> dict:
    """Locates the central directory. Returns its offset, size and entry count."""
    tail, tail_start = remote.read_tail(tail_bytes)

    eocd_at = tail.rfind(SIG_EOCD)
    if eocd_at < 0:
        raise RuntimeError("no end of central directory record in the tail")
    if eocd_at + EOCD_SIZE > len(tail):
        raise RuntimeError("end of central directory record is truncated")

    (
        _sig,
        _disk,
        _cd_disk,
        entries_here,
        entries_total,
        cd_size,
        cd_offset,
        _comment_len,
    ) = struct.unpack_from("<4sHHHHIIH", tail, eocd_at)

    result = {
        "zip64": False,
        "entries": entries_total,
        "cd_size": cd_size,
        "cd_offset": cd_offset,
        "file_size": remote.size,
    }

    overflowed = U32_MAX in (cd_size, cd_offset) or U16_MAX in (
        entries_here,
        entries_total,
    )
    if not overflowed:
        return result

    locator_at = eocd_at - EOCD64_LOCATOR_SIZE
    if locator_at < 0 or tail[locator_at : locator_at + 4] != SIG_EOCD64_LOCATOR:
        raise RuntimeError("ZIP64 is required but the locator is missing")
    _sig, _disk, eocd64_offset, _disks = struct.unpack_from(
        "<4sIQI", tail, locator_at
    )

    # The record may sit inside the tail already. Read it remotely only if not.
    if eocd64_offset >= tail_start:
        head = tail[eocd64_offset - tail_start : eocd64_offset - tail_start + EOCD64_MIN_SIZE]
    else:
        head = remote.read_range(eocd64_offset, EOCD64_MIN_SIZE)
    if head[:4] != SIG_EOCD64:
        raise RuntimeError("the ZIP64 locator does not point at a ZIP64 record")

    (
        _sig,
        _record_size,
        _version_made,
        _version_needed,
        _disk,
        _cd_disk,
        _entries_here,
        entries_total,
        cd_size,
        cd_offset,
    ) = struct.unpack_from("<4sQHHIIQQQQ", head, 0)

    result.update(
        {
            "zip64": True,
            "entries": entries_total,
            "cd_size": cd_size,
            "cd_offset": cd_offset,
            "eocd64_offset": eocd64_offset,
        }
    )
    return result


def parse_central_directory(blob: bytes, expected_entries: int) -> dict[str, dict]:
    """Turns the raw central directory into a name to entry mapping."""
    index: dict[str, dict] = {}
    position = 0
    limit = len(blob)
    while position + CENTRAL_FIXED_SIZE <= limit:
        fields = CENTRAL_STRUCT.unpack_from(blob, position)
        if fields[0] != SIG_CENTRAL:
            break
        (
            _sig,
            _version_made,
            _version_needed,
            flags,
            method,
            _mtime,
            _mdate,
            crc,
            csize,
            usize,
            name_len,
            extra_len,
            comment_len,
            _disk_start,
            _internal,
            _external,
            local_offset,
        ) = fields
        position += CENTRAL_FIXED_SIZE
        name = blob[position : position + name_len].decode("utf-8", "surrogateescape")
        position += name_len
        extra = blob[position : position + extra_len]
        position += extra_len
        position += comment_len

        needs = []
        if usize == U32_MAX:
            needs.append("usize")
        if csize == U32_MAX:
            needs.append("csize")
        if local_offset == U32_MAX:
            needs.append("local_offset")
        if needs:
            wide = parse_zip64_extra(extra, needs + ["disk_start"])
            usize = wide.get("usize", usize)
            csize = wide.get("csize", csize)
            local_offset = wide.get("local_offset", local_offset)

        index[name] = {
            "method": method,
            "flags": flags,
            "crc": crc,
            "csize": csize,
            "usize": usize,
            "local_offset": local_offset,
            "name_len": name_len,
        }

    if expected_entries and len(index) != expected_entries:
        raise RuntimeError(
            f"parsed {len(index)} entries, the record said {expected_entries}"
        )
    return index


def load_index(remote: RemoteFile, cache_path: str | None, max_cd_bytes: int) -> dict:
    """Returns the parsed central directory, from cache when one exists."""
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as handle:
            cached = json.load(handle)
        if cached.get("url") == remote.url:
            remote.seed_size(cached["file_size"])
            return cached

    eocd = read_end_of_central_directory(remote)
    cd_size = eocd["cd_size"]
    if cd_size > max_cd_bytes:
        raise SystemExit(
            f"the central directory is {cd_size} bytes, over the {max_cd_bytes} byte "
            f"limit. Stopping without fetching it."
        )

    blob = remote.read_range(eocd["cd_offset"], cd_size)
    entries = parse_central_directory(blob, eocd["entries"])

    payload = {
        "url": remote.url,
        "file_size": eocd["file_size"],
        "zip64": eocd["zip64"],
        "cd_offset": eocd["cd_offset"],
        "cd_size": cd_size,
        "entry_count": len(entries),
        "entries": entries,
    }
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        temporary = cache_path + ".partial"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        os.replace(temporary, cache_path)
    return payload


def extract_member(remote: RemoteFile, entry: dict, name: str) -> bytes:
    """Reads one member and returns its uncompressed bytes."""
    method = entry["method"]
    csize = entry["csize"]
    local_offset = entry["local_offset"]

    # One request covers the local file header and the member data. The header is
    # 30 bytes plus the name plus an extra field of unknown length, so the read
    # is padded and the real data offset is found inside the buffer.
    span = LOCAL_FIXED_SIZE + entry["name_len"] + LOCAL_EXTRA_SLACK + csize
    span = min(span, remote.size - local_offset)
    buffer = remote.read_range(local_offset, span)

    header = LOCAL_STRUCT.unpack_from(buffer, 0)
    if header[0] != SIG_LOCAL:
        raise RuntimeError(f"no local file header at offset {local_offset} for {name}")
    local_name_len = header[9]
    local_extra_len = header[10]
    data_at = LOCAL_FIXED_SIZE + local_name_len + local_extra_len

    if data_at + csize > len(buffer):
        # The local extra field was larger than the padding. Read the data exactly.
        payload = remote.read_range(local_offset + data_at, csize)
    else:
        payload = buffer[data_at : data_at + csize]

    if method == 0:
        raw = payload
    elif method == 8:
        raw = zlib.decompressobj(-zlib.MAX_WBITS).decompress(payload)
    else:
        raise RuntimeError(f"compression method {method} is not supported for {name}")

    if len(raw) != entry["usize"]:
        raise RuntimeError(
            f"{name} inflated to {len(raw)} bytes, the directory said {entry['usize']}"
        )
    actual_crc = zlib.crc32(raw) & U32_MAX
    if entry["crc"] and actual_crc != entry["crc"]:
        raise RuntimeError(f"{name} failed its CRC check")
    return raw


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Extract named members from a remote ZIP64 archive by range request."
    )
    parser.add_argument("--url", required=True, help="URL of the remote archive")
    parser.add_argument(
        "--index-cache",
        help="path for the parsed central directory, reused on later runs",
    )
    parser.add_argument(
        "--out-dir", help="directory to write extracted members into"
    )
    parser.add_argument(
        "--member", action="append", default=[], help="member path to extract"
    )
    parser.add_argument(
        "--members-from", help="file holding one member path per line"
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="report the central directory offset and size, then stop",
    )
    parser.add_argument(
        "--list-prefix", help="print index entries whose name starts with this"
    )
    parser.add_argument(
        "--max-cd-bytes",
        type=int,
        default=500 * 1024 * 1024,
        help="refuse to fetch a central directory larger than this",
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="write output files by basename instead of by archive path",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="how many members to fetch at once",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip members whose output file already exists at the right size",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="price the request from the cached index without transferring members",
    )
    args = parser.parse_args(argv)

    counter = TransferCounter()
    remote = RemoteFile(args.url, counter)

    if args.probe:
        eocd = read_end_of_central_directory(remote)
        print(json.dumps({"eocd": eocd, "transfer": counter.as_dict()}, indent=2))
        return 0

    index_cost_before = counter.body_bytes
    index = load_index(remote, args.index_cache, args.max_cd_bytes)
    index_cost = counter.body_bytes - index_cost_before
    entries = index["entries"]

    if args.list_prefix is not None:
        matches = sorted(n for n in entries if n.startswith(args.list_prefix))
        for name in matches:
            entry = entries[name]
            print(f"{entry['csize']}\t{entry['usize']}\t{entry['local_offset']}\t{name}")
        print(
            f"# {len(matches)} matching entries of {index['entry_count']}",
            file=sys.stderr,
        )
        return 0

    wanted = list(args.member)
    if args.members_from:
        with open(args.members_from, "r", encoding="utf-8") as handle:
            wanted.extend(line.strip() for line in handle if line.strip())
    if not wanted:
        parser.error("give at least one --member, or --members-from, or --probe")

    missing = [name for name in wanted if name not in entries]
    if missing:
        for name in missing:
            print(f"not in the archive: {name}", file=sys.stderr)
        return 2

    def destination_for(name: str) -> str | None:
        if not args.out_dir:
            return None
        if args.flat:
            return os.path.join(args.out_dir, os.path.basename(name))
        return os.path.join(args.out_dir, name)

    if args.resume:
        kept = []
        for name in wanted:
            path = destination_for(name)
            expected = entries[name]["usize"]
            if path and os.path.exists(path) and os.path.getsize(path) == expected:
                continue
            kept.append(name)
        skipped = len(wanted) - len(kept)
        wanted = kept
        print(f"resume: {skipped} already on disk, {len(wanted)} to fetch", file=sys.stderr)

    planned_bytes = sum(entries[name]["csize"] for name in wanted)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "members": len(wanted),
                    "planned_transfer_bytes": planned_bytes,
                    "planned_transfer_mib": round(planned_bytes / 1048576, 2),
                    "index_bytes_transferred": index_cost,
                },
                indent=2,
            )
        )
        return 0

    member_cost_before = counter.body_bytes
    wall_start = time.monotonic()
    written: list[dict] = []
    failures: list[dict] = []
    done = 0
    progress_lock = threading.Lock()

    def fetch_one(name: str) -> dict:
        raw = extract_member(remote, entries[name], name)
        destination = destination_for(name)
        if destination:
            os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
            # Write beside the target then rename, so an interrupted run never
            # leaves a short file that a later --resume would accept.
            temporary = destination + ".partial"
            with open(temporary, "wb") as handle:
                handle.write(raw)
            os.replace(temporary, destination)
        return {"member": name, "path": destination, "bytes": len(raw)}

    workers = max(1, args.workers)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_one, name): name for name in wanted}
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                written.append(future.result())
            except Exception as error:  # noqa: BLE001
                failures.append({"member": name, "error": f"{type(error).__name__}: {error}"})
            with progress_lock:
                done += 1
                if len(wanted) > 50 and done % 250 == 0:
                    print(f"  {done}/{len(wanted)} members", file=sys.stderr)

    member_cost = counter.body_bytes - member_cost_before
    wall_seconds = time.monotonic() - wall_start

    report = {
        "archive_bytes": index["file_size"],
        "central_directory_bytes": index["cd_size"],
        "entry_count": index["entry_count"],
        "members_requested": len(wanted),
        "members_extracted": len(written),
        "members_failed": len(failures),
        "index_bytes_transferred": index_cost,
        "member_bytes_transferred": member_cost,
        "total_bytes_transferred": counter.body_bytes,
        "uncompressed_bytes_written": sum(x["bytes"] for x in written),
        "http_requests": counter.requests,
        "request_seconds": round(counter.seconds, 3),
        "wall_seconds": round(wall_seconds, 3),
        "workers": workers,
    }
    if failures:
        report["failures"] = failures[:20]
    if len(written) <= 20:
        report["extracted"] = written
    print(json.dumps(report, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
