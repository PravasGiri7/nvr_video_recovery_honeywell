#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import struct
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


HEADER_ABS_START   = 0x5000
HEADER_SIZE        = 0x4000
BLOCK_GROUP_OFFSET = 0x0040 

VIDEO_REGION_START = 0x80005000

PARTITION1_START_REL_DEFAULT = 0x5000
VIDEO_DATA_REL_START_DEFAULT = 0x80000000

DAT_MARKERS = (b"\x80\x01\x00", b"\x81\x01\x00")
MIN_TS_S  = 946_684_800
MAX_TS_S  = 4_102_444_800
MIN_TS_US = MIN_TS_S * 1_000_000
MAX_TS_US = MAX_TS_S * 1_000_000

ENTRY_SIZE = 16

H264_MARKER = b"\x80\x01\x00"
H265_MARKER = b"\x81\x01\x00"


@dataclass
class Config:
    input_path:    str = ""
    output_dir:    str = ""
    partition1_start: int = PARTITION1_START_REL_DEFAULT

    zero_run_threshold:    int   = 20
    min_chunk_bytes:       int   = 512
    read_chunk_size:       int   = 64 * 1024 * 1024
    oversize_threshold_kb: float = 2500.0
    split_gap_seconds:     float = 2.0

    unknown_dir:   bool = False
    dump_metadata: bool = False
    manifest:      bool = True


@dataclass
class BlockGroupEntry:
    index:      int
    abs_offset: int
    start_time: int    # unix seconds
    group_num:  int


@dataclass
class Classification:
    category:   str
    reason:     str
    ts_count:   int
    ts_in_new:  int
    ts_ratio:   float
    first_ts:   Optional[float]
    median_ts:  Optional[float]


@dataclass
class CarvedChunk:
    path:       Path
    category:   str
    abs_start:  int
    abs_end:    int
    median_ts:  Optional[float]
    first_ts:   Optional[float]
    ts_count:   int
    ts_in_new:  int
    ts_ratio:   float
    reason:     str
    codec:      str = "unknown"
    h264_count: int = 0
    h265_count: int = 0


def fmt_time(ts: Optional[float]) -> str:
    if ts is None:
        return "-"
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
    except Exception:
        return str(ts)


def is_plausible_unix_seconds(v: int) -> bool:
    return MIN_TS_S <= v <= MAX_TS_S


def rounded_u32_to_offset(v: int) -> int:
    return v << 12



def parse_header_block_groups(
    fin, file_size: int
) -> tuple[list[BlockGroupEntry], Optional[float]]:

    if file_size < HEADER_ABS_START + HEADER_SIZE:
        print(f"  [WARN] File too small for header at 0x{HEADER_ABS_START:X}")
        return [], None

    fin.seek(HEADER_ABS_START)
    header = fin.read(min(HEADER_SIZE, file_size - HEADER_ABS_START))

    if len(header) >= 0x20:
        vdr = struct.unpack_from("<I", header, 0x0000)[0]
        nvr = struct.unpack_from("<I", header, 0x0008)[0]
        avr = struct.unpack_from("<I", header, 0x0010)[0]
        tor = struct.unpack_from("<I", header, 0x0018)[0]
        print(f"  Video Data Offset        : 0x{vdr:08X} -> abs 0x{rounded_u32_to_offset(vdr):012X}")
        print(f"  Next Video Offset        : 0x{nvr:08X} -> abs 0x{rounded_u32_to_offset(nvr):012X}")
        print(f"  Available Memory         : 0x{avr:08X} -> abs 0x{rounded_u32_to_offset(avr):012X}")
        print(f"  Total Allocatable Memory : 0x{tor:08X} -> abs 0x{rounded_u32_to_offset(tor):012X}")

    entry_region = header[BLOCK_GROUP_OFFSET:]
    num_possible = len(entry_region) // ENTRY_SIZE
    entries: list[BlockGroupEntry] = []

    for i in range(num_possible):
        off = i * ENTRY_SIZE
        raw = entry_region[off:off + ENTRY_SIZE]
        if not any(raw):
            continue
        ts       = struct.unpack_from("<I", raw, 4)[0]
        group_no = struct.unpack_from("<H", raw, 12)[0]
        if not is_plausible_unix_seconds(ts):
            continue
        abs_off = HEADER_ABS_START + BLOCK_GROUP_OFFSET + off
        entries.append(BlockGroupEntry(
            index=i, abs_offset=abs_off,
            start_time=ts, group_num=group_no,
        ))

    if not entries:
        return [], None

    new_boundary = float(min(e.start_time for e in entries))
    return entries, new_boundary


# Codec detection
def detect_codec(data: bytes) -> tuple[str, int, int]:
    h264 = data.count(H264_MARKER)
    h265 = data.count(H265_MARKER)
    if h264 == 0 and h265 == 0:
        return "unknown", 0, 0
    return ("h264", h264, h265) if h264 >= h265 else ("h265", h264, h265)


# DAT timestamp extraction
def extract_timestamps(data: bytes) -> list[tuple[int, float]]:
    results: list[tuple[int, float]] = []
    for marker in DAT_MARKERS:
        pos = 0
        while True:
            p = data.find(marker, pos)
            if p == -1:
                break
            ts_off = p + 11
            if ts_off + 8 <= len(data):
                ts_us = struct.unpack_from("<Q", data, ts_off)[0]
                if MIN_TS_US <= ts_us <= MAX_TS_US:
                    results.append((p, ts_us / 1_000_000.0))
            pos = p + 1
    results.sort(key=lambda x: x[0])
    return results


def find_split_point(data: bytes, gap_threshold: float) -> Optional[int]:
    timestamps = extract_timestamps(data)
    if len(timestamps) < 4:
        return None
    best_gap = 0.0
    split_at: Optional[int] = None
    for i in range(1, len(timestamps)):
        gap = abs(timestamps[i][1] - timestamps[i - 1][1])
        if gap > best_gap:
            best_gap = gap
            split_at = timestamps[i][0]
    if split_at is not None and best_gap >= gap_threshold:
        print(f"    Largest gap: {best_gap:.3f}s -> split at +0x{split_at:X}")
        return split_at
    print(f"    Largest gap: {best_gap:.3f}s below "
          f"{gap_threshold:.3f}s -- keeping together")
    return None


def split_oversized_segments(data: bytes, abs_start: int,
                              cfg: Config) -> list[tuple[bytes, int]]:
    threshold = int(cfg.oversize_threshold_kb * 1024)
    out: list[tuple[bytes, int]] = []

    def _split(buf: bytes, start: int) -> None:
        if len(buf) <= threshold:
            out.append((buf, start))
            return
        print(f"  [Oversized] 0x{start:012X} {len(buf)/1024:.1f} KB "
              f"-- scanning for timestamp gap")
        sp = find_split_point(buf, cfg.split_gap_seconds)
        if sp is None or sp <= 0 or sp >= len(buf):
            out.append((bytes(buf), start))
            return
        _split(buf[:sp], start)
        _split(buf[sp:], start + sp)

    _split(data, abs_start)
    return out


# Chunk classification
def classify_chunk(
    data:         bytes,
    abs_start:    int,
    new_boundary: Optional[float],
    cfg:          Config,
) -> tuple[Classification, Optional[float], Optional[float]]:
    timestamps = extract_timestamps(data)
    values     = [ts for _, ts in timestamps]
    first      = values[0]                if values else None
    median     = values[len(values) // 2] if values else None

    # If the header contains no valid Block Group boundary, carve everything
    # into one category instead of attempting old/new classification.
    if new_boundary is None:
        c = Classification(
            "unclassified",
            "no valid header boundary; old/new separation disabled",
            len(values), 0, 0.0, first, median,
        )
        return c, median, first

    if values:
        in_new = sum(1 for ts in values if ts >= new_boundary)
        ratio  = in_new / len(values)

        if median is not None and median >= new_boundary:
            c = Classification(
                "new",
                f"median {fmt_time(median)} >= header boundary "
                f"{fmt_time(new_boundary)} ({in_new}/{len(values)} in new session)",
                len(values), in_new, ratio, first, median,
            )
        else:
            c = Classification(
                "old",
                f"median {fmt_time(median)} < header boundary "
                f"{fmt_time(new_boundary)}",
                len(values), in_new, ratio, first, median,
            )
        return c, median, first

    # No timestamps
    if cfg.unknown_dir:
        c = Classification(
            "unknown",
            "no timestamps",
            0, 0, 0.0, None, None,
        )
    else:
        c = Classification(
            "old",
            "no timestamps",
            0, 0, 0.0, None, None,
        )
    return c, None, None


# Phase 1 — single carving pass
def phase1_carve(
    fin,
    regions:          list[tuple[int, int]],
    old_dir:          Path,
    new_dir:          Path,
    unknown_dir:      Optional[Path],
    unclassified_dir: Path,
    new_boundary:     Optional[float],
    cfg:              Config,
) -> tuple[list[CarvedChunk], dict]:
    """
    Single-pass carve. For each chunk:
        1. Carve chunk boundary (zero-run separator)
        2. Extract timestamps and classify old/new when a boundary exists
        3. Otherwise, place every chunk in the unclassified category
        4. Detect codec -> h264 / h265 / mixed / unknown
        5. Write once to <category>/<codec>/ sub-folder
    """
    total_bytes    = sum(e - s for s, e in regions)
    scanned        = 0
    last_pct       = -1
    sep            = b"\x00" * cfg.zero_run_threshold
    sep_len        = len(sep)
    oversize_bytes = int(cfg.oversize_threshold_kb * 1024)

    counters: dict[str, int] = {
        "old": 0, "new": 0, "unknown": 0, "unclassified": 0,
    }
    written: dict[str, int] = {
        "old": 0, "new": 0, "unknown": 0, "unclassified": 0,
    }
    carved:   list[CarvedChunk] = []

    def codec_subdir(base: Path, codec: str) -> Path:
        d = base / codec
        d.mkdir(parents=True, exist_ok=True)
        return d

    def out_dir(category: str, codec: str) -> Path:
        if category == "unclassified":
            base = unclassified_dir
        elif category == "new":
            base = new_dir
        elif category == "unknown" and unknown_dir is not None:
            base = unknown_dir
        else:
            base = old_dir
        return codec_subdir(base, codec)

    def process_one(buf: bytes, abs_start: int) -> None:
        if len(buf) < cfg.min_chunk_bytes:
            return

        c, median, first = classify_chunk(buf, abs_start, new_boundary, cfg)
        codec, h264_cnt, h265_cnt = detect_codec(buf)

        category = c.category
        counters[category] += 1

        filename = f"{category}_{codec}_chunk_{counters[category]:04d}.dat"
        path     = out_dir(category, codec) / filename
        path.write_bytes(buf)
        written[category] += len(buf)

        carved.append(CarvedChunk(
            path       = path,
            category   = category,
            abs_start  = abs_start,
            abs_end    = abs_start + len(buf),
            median_ts  = median,
            first_ts   = first,
            ts_count   = c.ts_count,
            ts_in_new  = c.ts_in_new,
            ts_ratio   = c.ts_ratio,
            reason     = c.reason,
            codec      = codec,
            h264_count = h264_cnt,
            h265_count = h265_cnt,
        ))

        print(
            f"[Carve] {filename} {len(buf)/1024:.1f} KB "
            f"0x{abs_start:012X}-0x{abs_start+len(buf):012X} "
            f"-> {category.upper()}  ({c.reason}; "
            f"median={fmt_time(median)}, ts={c.ts_count}, "
            f"codec={codec} [h264={h264_cnt} h265={h265_cnt}])"
        )

    def seal(buf: bytes, abs_start: int) -> None:
        if len(buf) < cfg.min_chunk_bytes:
            return
        sys.stdout.write("\r" + " " * 140 + "\r")
        if len(buf) > oversize_bytes:
            for sub, sub_start in split_oversized_segments(buf, abs_start, cfg):
                process_one(sub, sub_start)
        else:
            process_one(buf, abs_start)

    def progress(done: bool = False) -> None:
        nonlocal last_pct
        pct = min(100, int(scanned * 100 / total_bytes)) if total_bytes else 100
        if done or pct != last_pct:
            if new_boundary is None:
                status = (
                    f"carved:{counters['unclassified']}  "
                    f"boundary:not found"
                )
            else:
                status = (
                    f"old:{counters['old']} new:{counters['new']} "
                    f"unk:{counters['unknown']}  "
                    f"boundary:{fmt_time(new_boundary)}"
                )
            line = (
                f"\r  [Phase 1] {pct:3d}%  "
                f"{scanned/(1024**2):8.1f}/{total_bytes/(1024**2):.1f} MB  "
                f"{status}"
            )
            sys.stdout.write(line + (" ✓\n" if done else "  "))
            sys.stdout.flush()
            last_pct = pct

    for region_start, region_end in regions:
        carry       = b""
        carry_start = region_start
        pos         = region_start
        fin.seek(region_start)

        while pos < region_end:
            raw_start = pos
            to_read   = min(cfg.read_chunk_size, region_end - pos)
            raw       = fin.read(to_read)
            if not raw:
                break
            pos     += len(raw)
            scanned += len(raw)

            if carry:
                block       = carry + raw
                block_start = carry_start
            else:
                block       = raw
                block_start = raw_start

            parts = block.split(sep)
            if len(parts) == 1:
                carry       = block
                carry_start = block_start
                progress()
                continue

            cursor = block_start
            for part in parts[:-1]:
                if part:
                    seal(part, cursor)
                cursor += len(part) + sep_len

            carry       = parts[-1]
            carry_start = cursor
            progress()

        if carry:
            seal(carry, carry_start)
        progress()

    progress(done=True)
    return carved, {"counters": counters, "written": written}



# Manifest writer
def write_manifest(
    carved:        list[CarvedChunk],
    manifest_path: Path,
    new_boundary:  Optional[float],
) -> None:
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "category", "filename",
            "abs_start_hex", "abs_end_hex", "size_bytes",
            "reason",
            "timestamp_count", "timestamps_in_new_session",
            "first_timestamp_utc", "median_timestamp_utc",
            "codec", "h264_marker_count", "h265_marker_count",
        ])
        for chunk in carved:
            size = chunk.abs_end - chunk.abs_start
            w.writerow([
                chunk.category, chunk.path.name,
                f"0x{chunk.abs_start:X}", f"0x{chunk.abs_end:X}", size,
                chunk.reason,
                chunk.ts_count, chunk.ts_in_new,
                fmt_time(chunk.first_ts), fmt_time(chunk.median_ts),
                chunk.codec, chunk.h264_count, chunk.h265_count,
            ])



# Main pipeline
def run(cfg: Config) -> None:
    start_time = time.monotonic()
    input_path = Path(cfg.input_path)
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        sys.exit(1)

    file_size = input_path.stat().st_size
    print(f"\n{'='*78}")
    print("  Honeywell DVR/NVR Timestamp-Primary Carver  --  Single-Pass Edition")
    print(f"{'='*78}")
    print(f"  Input              : {input_path} "
          f"({file_size:,} bytes, {file_size/(1024**3):.2f} GB)")
    print(f"  Header at          : 0x{HEADER_ABS_START:X}")

    print("\n[Phase 0] Parsing header")
    with open(input_path, "rb") as fin:
        block_group_entries, new_boundary = parse_header_block_groups(
            fin, file_size
        )

    print(f"\n  Block Group entries found : {len(block_group_entries)}")
    for e in block_group_entries:
        print(f"    Group {e.group_num:>3} | {fmt_time(e.start_time)} "
              f"| offset 0x{e.abs_offset:X}")

    if new_boundary is None:
        print("\n[WARN] No valid Block Group Start Times found in header.")
        print("       Continuing carving without old/new separation.")
    else:
        print(f"\n  Header boundary    : {fmt_time(new_boundary)}")
        print("  (earliest Block Group Start Time -- chunks before this = old)")

    base_out = (Path(cfg.output_dir) if cfg.output_dir
                else input_path.parent / input_path.stem)
    old_dir          = base_out / "old_video"
    new_dir          = base_out / "new_video"
    unk_dir          = base_out / "unknown_video" if cfg.unknown_dir else None
    unclassified_dir = base_out / "carved_video"

    if new_boundary is None:
        unclassified_dir.mkdir(parents=True, exist_ok=True)
    else:
        old_dir.mkdir(parents=True, exist_ok=True)
        new_dir.mkdir(parents=True, exist_ok=True)
        if unk_dir is not None:
            unk_dir.mkdir(parents=True, exist_ok=True)

    manifest_path   = base_out / "carving_manifest.csv" if cfg.manifest else None
    video_start_abs = min(VIDEO_REGION_START, file_size)
    video_end_abs   = file_size
    regions = ([(video_start_abs, video_end_abs)]
               if video_start_abs < video_end_abs else [])
    if not regions:
        print("[ERROR] No video region to carve.")
        sys.exit(1)

    if new_boundary is None:
        print(f"\n  Output carved  : {unclassified_dir}")
    else:
        print(f"\n  Output old     : {old_dir}")
        print(f"  Output new     : {new_dir}")
        if unk_dir:
            print(f"  Output unknown : {unk_dir}")
    if manifest_path:
        print(f"  Manifest       : {manifest_path}")
    print("  Codec sub-dirs : h264/ h265/ mixed/ unknown/ (created on demand)")

    if new_boundary is None:
        print("\n[Phase 1] Carving (single pass -- no old/new separation)")
    else:
        print("\n[Phase 1] Carving (single pass -- header boundary + codec detection)")
    with open(input_path, "rb") as fin:
        carved, stats = phase1_carve(
            fin, regions,
            old_dir, new_dir, unk_dir, unclassified_dir,
            new_boundary, cfg,
        )

    codec_counts: dict[str, int] = {}
    for chunk in carved:
        codec_counts[chunk.codec] = codec_counts.get(chunk.codec, 0) + 1
    print(f"\n  Codec breakdown : " +
          "  ".join(f"{k}:{v}" for k, v in sorted(codec_counts.items())))

    if manifest_path is not None:
        write_manifest(carved, manifest_path, new_boundary)
        print(f"  Manifest written : {manifest_path}")

    counters = stats["counters"]
    written  = stats["written"]
    print(f"\n{'='*78}")
    if new_boundary is None:
        print("  Header boundary  : not found (old/new separation disabled)")
        print(f"  CARVED chunks : {counters['unclassified']:,}  "
              f"({written['unclassified']/(1024**2):.1f} MB)  "
              f"-> {unclassified_dir}")
    else:
        print(f"  Header boundary  : {fmt_time(new_boundary)}")
        print(f"  OLD chunks : {counters['old']:,}  "
              f"({written['old']/(1024**2):.1f} MB)  -> {old_dir}")
        print(f"  NEW chunks : {counters['new']:,}  "
              f"({written['new']/(1024**2):.1f} MB)  -> {new_dir}")
        if unk_dir:
            print(f"  UNK chunks : {counters['unknown']:,}  "
                  f"({written['unknown']/(1024**2):.1f} MB)  -> {unk_dir}")
    print(f"  Codec breakdown : " +
          "  ".join(f"{k}:{v}" for k, v in sorted(codec_counts.items())))
    if manifest_path:
        print(f"  Manifest   : {manifest_path}")

    elapsed = time.monotonic() - start_time
    mins, secs = divmod(elapsed, 60)
    hours, mins = divmod(mins, 60)
    print(f"  Processing time  : {int(hours):02d}:{int(mins):02d}:{secs:05.2f} "
          f"({elapsed:.2f}s total)")
    print(f"{'='*78}\n")



# CLI
def parse_int_auto(s: str) -> int:
    return int(s, 0)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Honeywell DVR/NVR single-pass carver -- "
                    "header block group boundary + codec detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Classification:
  Header boundary = earliest Block Group Start Time at 0x5000+0x0040.
  new  ->  chunk median >= header boundary
  old  ->  chunk median <  header boundary
  If no valid boundary exists, all chunks are written under carved_video/.
  Codec sub-folders: h264/ h265/ mixed/ unknown/
        """,
    )
    p.add_argument("--input",  "-i", required=True, help="Raw disk image path")
    p.add_argument("--output-dir", "-o", default="", help="Base output directory")
    p.add_argument("--partition1-start",
                   default=hex(PARTITION1_START_REL_DEFAULT), type=parse_int_auto,
                   help="Partition 1 start offset. Default: 0x5000")
    p.add_argument("--threshold",    default=20,           type=int,
                   help="Zero-run separator length. Default: 20")
    p.add_argument("--min-size",     default=512,          type=int,
                   help="Minimum carved chunk size in bytes. Default: 512")
    p.add_argument("--buffer",       default=64*1024*1024, type=int,
                   help="Read buffer size. Default: 64 MB")
    p.add_argument("--oversize-kb",  default=2500.0,       type=float,
                   help="Split chunks larger than this KB. Default: 2500")
    p.add_argument("--split-gap",    default=2.0,          type=float,
                   help="Timestamp gap for oversized splitting. Default: 2s")
    p.add_argument("--unknown-dir",   action="store_true",
                   help="Write no-timestamp chunks to unknown_video/")
    p.add_argument("--dump-metadata", action="store_true",
                   help="Write header block group entries to CSV")
    p.add_argument("--no-manifest",   action="store_true",
                   help="Do not write carving_manifest.csv")

    args = p.parse_args()
    cfg  = Config(
        input_path            = args.input,
        output_dir            = args.output_dir,
        partition1_start      = args.partition1_start,
        zero_run_threshold    = args.threshold,
        min_chunk_bytes       = args.min_size,
        read_chunk_size       = args.buffer,
        oversize_threshold_kb = args.oversize_kb,
        split_gap_seconds     = args.split_gap,
        unknown_dir           = args.unknown_dir,
        dump_metadata         = args.dump_metadata,
        manifest              = not args.no_manifest,
    )
    run(cfg)


if __name__ == "__main__":
    main()