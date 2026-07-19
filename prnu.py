from __future__ import annotations
from sklearn.metrics import silhouette_score

import glob
import hashlib
import logging
import multiprocessing
import os
import re
import shutil
import struct
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import h5py
import numpy as np
import psutil
from scipy.signal import wiener
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans, SpectralClustering
from tqdm import tqdm


@dataclass
class Config:
    input_dir:       str = r"PATH_TO_INPUT_DIR"
    output_dir:      str = r"PATH_TO_OUTPUT_DIR"
    hdf5_cache_path: str = r""

    num_cameras:         int   = 0
    max_k_auto:          int   = 16
    max_frames_per_clip: int   = 30
    video_extensions:    tuple = (".dat",)
    extraction_workers:  int   = 12     

    prnu_resize: tuple = (540, 960)
    wiener_size: int   = 5

    # Overlay mask
    mask_top_px:    int = 40
    mask_bottom_px: int = 30
    mask_left_px:   int = 0
    mask_right_px:  int = 0

    # Frame quality gates
    min_frame_brightness: float = 8.0
    max_frame_brightness: float = 247.0
    min_blur_laplacian:   float = 150.0

    # Per-clip PRNU outlier rejection
    mad_outlier_threshold: float = 2.5
    min_valid_frames:      int   = 2

    # Confidence
    low_confidence_threshold: float = 0.35
    low_confidence_damping:   float = 0.5

    # NCC matrix block size
    ncc_block_size: int = 5000

    # Spectral clustering
    refinement_rounds: int = 5

    copy_chunks: bool = True

    def __post_init__(self):
        if not self.hdf5_cache_path:
            self.hdf5_cache_path = str(Path(self.output_dir) / "prnu_cache.h5")
        if self.extraction_workers == 0:
            self.extraction_workers = max(1, multiprocessing.cpu_count() // 2)

    @property
    def prnu_dim(self) -> int:
        return self.prnu_resize[0] * self.prnu_resize[1]

    def ram_report(self, n: int) -> str:
        prnu_disk_gb  = n * self.prnu_dim * 4 / 1e9
        ncc_ram_gb    = n * n * 4 / 1e9
        block_ram_gb  = 2 * self.ncc_block_size * self.prnu_dim * 4 / 1e9
        avail_gb      = psutil.virtual_memory().available / 1e9
        return (
            f"PRNU {self.prnu_resize[0]}×{self.prnu_resize[1]} | "
            f"Cache on disk: {prnu_disk_gb:.1f} GB | "
            f"NCC matrix in RAM: {ncc_ram_gb:.1f} GB | "
            f"Two blocks in RAM: {block_ram_gb:.1f} GB | "
            f"Available RAM: {avail_gb:.1f} GB"
        )


FFMPEG_PATH  = Path(r"PATH_TO_FFMPEG")
FFPROBE_PATH = Path(r"PATH_TO_FFPROBE")

cfg = Config()
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


@dataclass
class ClipMeta:
    path:                str
    prnu_quality_ok:     bool  = True
    prnu_confidence:     float = 1.0
    n_valid_frames:      int   = 0
    n_rejected_frames:   int   = 0
    cluster:             int   = -1
    timestamp:           float = 0.0   
    timestamp_end:       float = 0.0   
    failed_extraction:   bool  = False


DAT_MARKERS = [b'\x80\x01\x00', 
               b'\x81\x01\x00']


def _find_all_dat_markers(data: bytes) -> list:
    results = []
    for marker in DAT_MARKERS:
        start = 0
        while True:
            pos = data.find(marker, start)
            if pos == -1:
                break
            results.append((pos, marker))
            start = pos + 1
    results.sort(key=lambda x: x[0])
    return results


def _valid_ts_us(ts_us: int) -> bool:
    MIN_TS_US = 946_684_800  * 1_000_000
    MAX_TS_US = 4_102_444_800 * 1_000_000
    return MIN_TS_US <= ts_us <= MAX_TS_US


def extract_timestamps_from_dat(dat_path: str) -> tuple[Optional[float], Optional[float]]:
    try:
        with open(dat_path, 'rb') as f:
            data = f.read()

        markers = _find_all_dat_markers(data)
        if not markers:
            return None, None

        start_ts: Optional[float] = None
        end_ts:   Optional[float] = None

        # Scan forward for the first valid timestamp
        for offset, _ in markers:
            ts_offset = offset + 11
            if ts_offset + 8 > len(data):
                continue
            ts_us = struct.unpack_from('<Q', data, ts_offset)[0]
            if _valid_ts_us(ts_us):
                start_ts = ts_us / 1_000_000.0
                break

        # Scan backward for the last valid timestamp
        for offset, _ in reversed(markers):
            ts_offset = offset + 11
            if ts_offset + 8 > len(data):
                continue
            ts_us = struct.unpack_from('<Q', data, ts_offset)[0]
            if _valid_ts_us(ts_us):
                end_ts = ts_us / 1_000_000.0
                break

        return start_ts, end_ts

    except Exception:
        return None, None


def parse_timestamps(clip_path: str) -> tuple[float, float]:

    start_ts, end_ts = extract_timestamps_from_dat(clip_path)

    if start_ts is not None:
        return start_ts, (end_ts if end_ts is not None else 0.0)

    log.warning(f"  DAT timestamp failed for {Path(clip_path).name}, using fallback")

    name = Path(clip_path).stem

    m = re.search(r'(\d{10,13})', name)
    if m:
        raw = int(m.group(1))
        return (raw / 1000.0 if raw > 1e12 else float(raw)), 0.0

    m = re.search(r'(\d{4}[-_]\d{2}[-_]\d{2}[-_T]\d{2}[-_:]\d{2})', name)
    if m:
        for fmt in ("%Y-%m-%d_%H-%M", "%Y-%m-%dT%H:%M", "%Y_%m_%d_%H_%M"):
            try:
                return datetime.strptime(m.group(1), fmt).timestamp(), 0.0
            except ValueError:
                pass
    m = re.search(r'chunk[_\-]?(\d+)', name, re.IGNORECASE)
    if m:
        log.warning(f"  Using sequence number for {Path(clip_path).name}")
        return float(int(m.group(1))), 0.0

    nums = re.findall(r'(\d+)', name)
    if nums:
        log.warning(f"  Using sequence number for {Path(clip_path).name}")
        return float(int(nums[-1])), 0.0

    log.warning(f"  Using mtime for {Path(clip_path).name}")
    return os.path.getmtime(clip_path), 0.0


def parse_timestamp(clip_path: str) -> float:
    return parse_timestamps(clip_path)[0]



def chunk_number_from_path(path: str) -> int:

    stem = Path(path).stem
    m = re.search(r'chunk[_\-]?(\d+)', stem, re.IGNORECASE)
    if m:
        return int(m.group(1))

    # Fallback: use the last number in the filename
    nums = re.findall(r'(\d+)', stem)
    return int(nums[-1]) if nums else 0


def detect_codec(clip_path: str) -> str:
    """Returns 'hevc', 'h264', or 'unknown'."""
    result = subprocess.run(
        [str(FFPROBE_PATH), "-v", "error",
         "-select_streams", "v:0",
         "-show_entries", "stream=codec_name",
         "-of", "csv=p=0", clip_path],
        capture_output=True, text=True,
    )
    codec = result.stdout.strip().lower()
    if "hevc" in codec or "h265" in codec:
        return "hevc"
    if "h264" in codec or "avc" in codec:
        return "h264"
    return "unknown"

def _run_ffmpeg_frame(clip_path: str, tmpdir: str,
                      frame_idx: int, post_seek: float | None) -> np.ndarray | None:
    """Extract a single frame with an optional post-input seek."""
    out_path = os.path.join(tmpdir, f"frame_{frame_idx:04d}.jpg")
    seek = ["-ss", str(post_seek)] if post_seek is not None else []
    cmd = (
        [str(FFMPEG_PATH), "-y",
         "-fflags", "+discardcorrupt+genpts",
         "-err_detect", "ignore_err",
         "-i", clip_path]
        + seek
        + ["-vframes", "1", "-q:v", "2", out_path]
    )
    subprocess.run(cmd, capture_output=True)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 10_000:
        return cv2.imread(out_path)
    return None


def extract_frames(clip_path: str, max_frames: int) -> list:
    tmpdir = tempfile.mkdtemp(prefix="frames_")
    frames = []
    try:
        codec   = detect_codec(clip_path)
        is_hevc = codec == "hevc"

        probe = subprocess.run(
            [str(FFPROBE_PATH), "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=duration", "-of", "csv=p=0", clip_path],
            capture_output=True, text=True,
        )
        duration = None
        try:
            val = float(probe.stdout.strip().split()[0])
            if val > 0:
                duration = val
        except (ValueError, IndexError):
            pass

        if is_hevc:
            if duration:
                seek_points = list(np.linspace(2.0, duration * 0.95, max_frames))
            else:
                seek_points = [2.0, 3.0, 5.0, 8.0, 10.0, 12.0, 15.0,
                               18.0, 20.0, 25.0, 30.0][:max_frames]

            for i, seek in enumerate(seek_points):
                frame = _run_ffmpeg_frame(clip_path, tmpdir, i, seek)
                if frame is not None:
                    frames.append(frame)
                if len(frames) >= max_frames:
                    break

            if not frames:
                frame = _run_ffmpeg_frame(clip_path, tmpdir, 999, None)
                if frame is not None:
                    frames.append(frame)

        else:
            if duration:
                for i, ts in enumerate(np.linspace(0, duration * 0.95, max_frames)):
                    frame = _run_ffmpeg_frame(clip_path, tmpdir, i, ts if i > 0 else None)
                    if frame is not None:
                        frames.append(frame)
            else:
                out_pattern = os.path.join(tmpdir, "frame_%04d.jpg")
                subprocess.run(
                    [str(FFMPEG_PATH), "-y",
                     "-fflags", "+discardcorrupt+genpts",
                     "-err_detect", "ignore_err",
                     "-i", clip_path,
                     "-vf", "select=not(mod(n\\,30))",
                     "-vsync", "vfr", "-vframes", str(max_frames),
                     "-q:v", "2", out_pattern],
                    capture_output=True,
                )
                for img_path in sorted(
                        glob.glob(os.path.join(tmpdir, "*.jpg")))[:max_frames]:
                    f = cv2.imread(img_path)
                    if f is not None:
                        frames.append(f)

    except Exception:
        pass
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return frames


def get_frame_quality(frame: np.ndarray, cfg) -> tuple[bool, bool, float, float]:
    """Return brightness status, Laplacian status, brightness, and variance."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

    brightness = float(gray.mean())
    laplacian_value = float(
        cv2.Laplacian(gray.astype(np.uint8), cv2.CV_32F).var()
    )

    brightness_ok = (
        cfg.min_frame_brightness
        <= brightness
        <= cfg.max_frame_brightness
    )
    laplacian_ok = laplacian_value >= cfg.min_blur_laplacian

    return brightness_ok, laplacian_ok, brightness, laplacian_value


def mask_overlay(frame: np.ndarray, cfg) -> np.ndarray:
    h, w   = frame.shape[:2]
    masked = frame.copy().astype(np.float32)

    if cfg.mask_top_px    > 0: masked[:cfg.mask_top_px, :]        = 0
    if cfg.mask_bottom_px > 0: masked[h - cfg.mask_bottom_px:, :] = 0
    if cfg.mask_left_px   > 0: masked[:, :cfg.mask_left_px]       = 0
    if cfg.mask_right_px  > 0: masked[:, w - cfg.mask_right_px:]  = 0

    overlay_h = int(h * 0.18)
    overlay_w = int(w * 0.35)
    masked[:overlay_h, :overlay_w] = 0

    return masked

def extract_prnu_from_frame(frame: np.ndarray, wiener_size: int,
                             resize: tuple) -> np.ndarray:
    resized   = cv2.resize(frame, (resize[1], resize[0]))
    residuals = []
    for c in range(3):
        ch       = resized[:, :, c].astype(np.float32) / 255.0
        denoised = wiener(ch, mysize=wiener_size)
        residuals.append(ch - denoised)
    noise = np.mean(residuals, axis=0).flatten()
    noise -= noise.mean()
    return noise


def _process_clip_worker(args: tuple) -> tuple:
    (clip_path, max_frames, wiener_size, prnu_resize,
     mt, mb, ml, mr, min_b, max_b, min_blur,
     mad_thresh, min_valid, low_conf_thresh) = args

    class _Cfg:
        mask_top_px = mt; mask_bottom_px = mb
        mask_left_px = ml; mask_right_px = mr
        min_frame_brightness = min_b; max_frame_brightness = max_b
        min_blur_laplacian = min_blur

    local_cfg = _Cfg()

    ts_start, ts_end = parse_timestamps(clip_path)

    meta = dict(
        path=clip_path,
        prnu_quality_ok=True,
        prnu_confidence=0.5,
        n_extracted_frames=0,
        n_valid_frames=0,
        n_brightness_failed=0,
        n_laplacian_failed=0,
        n_rejected_frames=0,
        mean_laplacian=0.0,
        min_laplacian=0.0,
        max_laplacian=0.0,
        timestamp=ts_start,
        timestamp_end=ts_end,
        failed_extraction=False,
    )

    frames = extract_frames(clip_path, max_frames)
    if not frames:
        meta["failed_extraction"] = True
        return clip_path, None, meta

    meta["n_extracted_frames"] = len(frames)

    usable = []
    laplacian_values = []

    for frame in frames:
        brightness_ok, laplacian_ok, _, laplacian_value = (
            get_frame_quality(frame, local_cfg)
        )
        laplacian_values.append(laplacian_value)

        if not brightness_ok:
            meta["n_brightness_failed"] += 1
        elif not laplacian_ok:
            meta["n_laplacian_failed"] += 1
        else:
            usable.append(frame)

    if laplacian_values:
        meta["mean_laplacian"] = float(np.mean(laplacian_values))
        meta["min_laplacian"] = float(np.min(laplacian_values))
        meta["max_laplacian"] = float(np.max(laplacian_values))

    meta["n_valid_frames"] = len(usable)

    if not usable:
        meta["prnu_quality_ok"] = False
        del frames
        return clip_path, None, meta

    del frames

    residuals = []
    for frame in usable:
        masked    = mask_overlay(frame, local_cfg)
        masked_u8 = np.clip(masked, 0, 255).astype(np.uint8)
        residuals.append(extract_prnu_from_frame(masked_u8, wiener_size, prnu_resize))
    del usable

    meta["n_valid_frames"] = len(residuals)

    if len(residuals) >= min_valid + 1:
        norms  = np.array([np.linalg.norm(r) for r in residuals])
        median = np.median(norms)
        mad    = np.median(np.abs(norms - median))
        if mad > 1e-12:
            thresh   = mad_thresh * mad
            accepted = [r for r, n in zip(residuals, norms)
                        if abs(n - median) <= thresh]
            meta["n_rejected_frames"] = len(residuals) - len(accepted)
            if accepted:
                residuals = accepted

    avg  = np.mean(residuals, axis=0)
    avg -= avg.mean()
    norm = np.linalg.norm(avg)
    if norm > 1e-10:
        avg /= norm
    else:
        meta["prnu_quality_ok"] = False

    if len(residuals) >= 2:
        unit = []
        for r in residuals:
            n = np.linalg.norm(r)
            unit.append(r / n if n > 1e-10 else r)
        sims = [float(np.dot(unit[i], unit[j]))
                for i in range(len(unit))
                for j in range(i + 1, len(unit))]
        meta["prnu_confidence"] = (float(np.mean(sims)) + 1.0) / 2.0
    else:
        meta["prnu_confidence"] = 0.5

    del residuals
    return clip_path, avg.astype(np.float32), meta


def _h5key(path: str) -> str:
    return hashlib.sha1(os.path.abspath(path).encode("utf-8")).hexdigest()


def _load_block(clips: list[ClipMeta], start: int, end: int,
                cfg: Config) -> np.ndarray:
    keys  = [_h5key(c.path) for c in clips[start:end]]
    with h5py.File(cfg.hdf5_cache_path, "r") as h5:
        block = np.stack([h5[k][:] for k in keys], axis=0).astype(np.float32)
    norms = np.linalg.norm(block, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return block / norms

def phase1_extract(all_files: list, cfg: Config) -> list[ClipMeta]:
    os.makedirs(cfg.output_dir, exist_ok=True)

    already = set()
    if os.path.exists(cfg.hdf5_cache_path):
        with h5py.File(cfg.hdf5_cache_path, "r") as h5:
            already = set(h5.keys())
        log.info(f"  Resume: {len(already):,} clips already cached")

    to_process = [f for f in all_files if _h5key(f) not in already]
    log.info(f"  To process: {len(to_process):,}  |  Cached: {len(already):,}")

    worker_args = [
        (p, cfg.max_frames_per_clip, cfg.wiener_size,
         cfg.prnu_resize,
         cfg.mask_top_px, cfg.mask_bottom_px,
         cfg.mask_left_px, cfg.mask_right_px,
         cfg.min_frame_brightness, cfg.max_frame_brightness,
         cfg.min_blur_laplacian,
         cfg.mad_outlier_threshold, cfg.min_valid_frames,
         cfg.low_confidence_threshold)
        for p in to_process
    ]

    meta_map: dict = {}
    if already:
        with h5py.File(cfg.hdf5_cache_path, "r") as h5:
            for key in already:
                ds = h5[key]
                p  = str(ds.attrs["path"])
                meta_map[p] = dict(
                    path              = p,
                    prnu_quality_ok     = bool(ds.attrs.get("prnu_quality_ok",      True)),
                    prnu_confidence     = float(ds.attrs.get("prnu_confidence",       0.5)),
                    n_extracted_frames  = int(ds.attrs.get("n_extracted_frames",      0)),
                    n_valid_frames      = int(ds.attrs.get("n_valid_frames",          0)),
                    n_brightness_failed = int(ds.attrs.get("n_brightness_failed",     0)),
                    n_laplacian_failed  = int(ds.attrs.get("n_laplacian_failed",      0)),
                    n_rejected_frames   = int(ds.attrs.get("n_rejected_frames",       0)),
                    mean_laplacian      = float(ds.attrs.get("mean_laplacian",         0.0)),
                    min_laplacian       = float(ds.attrs.get("min_laplacian",          0.0)),
                    max_laplacian       = float(ds.attrs.get("max_laplacian",          0.0)),
                    timestamp           = float(ds.attrs.get("timestamp",             0.0)),
                    timestamp_end       = float(ds.attrs.get("timestamp_end",         0.0)),
                    failed_extraction   = bool(ds.attrs.get("failed_extraction",      False)),
                )

    failed = []
    with h5py.File(cfg.hdf5_cache_path, "a") as h5:
        with ProcessPoolExecutor(max_workers=cfg.extraction_workers) as exe:
            futures = {exe.submit(_process_clip_worker, a): a[0]
                       for a in worker_args}
            with tqdm(total=len(futures), desc="Extracting PRNU") as pbar:
                for future in as_completed(futures):
                    clip_path, vec, meta = future.result()
                    key = _h5key(clip_path)
                    if vec is None:
                        if meta.get("failed_extraction", False):
                            failed.append(clip_path)
                    else:
                        ds = h5.require_dataset(
                            key,
                            shape=(cfg.prnu_dim,),
                            dtype=np.float32
                        )
                        ds[:] = vec

                        for k2, v in meta.items():
                            ds.attrs[k2] = v
                    meta_map[clip_path] = meta
                    pbar.update(1)

    if failed:
        log.warning(
            f"  {len(failed)} clip(s) failed frame extraction — quarantining."
        )
        _quarantine_clips(
            failed,
            cfg.output_dir,
            reason="frame_extraction_failed",
            copy=cfg.copy_chunks,
        )
    MIN_FRAMES_FOR_CLUSTERING = max(1, round(cfg.max_frames_per_clip * 0.15))
    low_frame_paths = [
        m["path"] for m in meta_map.values()
        if not m.get("failed_extraction", False)
        and m.get("n_valid_frames", 0) < MIN_FRAMES_FOR_CLUSTERING
    ]
    if low_frame_paths:
        log.warning(
            f"  {len(low_frame_paths)} clip(s) had fewer than "
            f"{MIN_FRAMES_FOR_CLUSTERING} valid frames — quarantining."
        )
        _quarantine_clips(
            low_frame_paths,
            cfg.output_dir,
            reason="frame_extraction_failed",
            copy=cfg.copy_chunks,
        )
    low_frame_set = set(low_frame_paths)

    total_extracted_frames = sum(
        m.get("n_extracted_frames", 0)
        for m in meta_map.values()
    )
    total_brightness_failed = sum(
        m.get("n_brightness_failed", 0)
        for m in meta_map.values()
    )
    total_laplacian_failed = sum(
        m.get("n_laplacian_failed", 0)
        for m in meta_map.values()
    )
    total_quality_passed = sum(
        m.get("n_valid_frames", 0)
        for m in meta_map.values()
    )

    all_laplacian_means = [
        m.get("mean_laplacian", 0.0)
        for m in meta_map.values()
        if m.get("n_extracted_frames", 0) > 0
    ]

    log.info(
        f"  Frame-filter statistics at Laplacian threshold "
        f"{cfg.min_blur_laplacian:g}:"
    )
    log.info(f"    Extracted frames:          {total_extracted_frames:,}")
    log.info(f"    Brightness rejected:       {total_brightness_failed:,}")
    log.info(f"    Laplacian rejected:        {total_laplacian_failed:,}")
    log.info(f"    Passed quality filtering:  {total_quality_passed:,}")
    if all_laplacian_means:
        log.info(
            f"    Mean clip Laplacian:        "
            f"{float(np.mean(all_laplacian_means)):.2f}"
        )

    clips = [
        ClipMeta(
            path              = m["path"],
            prnu_quality_ok   = m["prnu_quality_ok"],
            prnu_confidence   = m["prnu_confidence"],
            n_valid_frames    = m["n_valid_frames"],
            n_rejected_frames = m["n_rejected_frames"],
            timestamp         = m["timestamp"],
            timestamp_end     = m["timestamp_end"],
        )
        for m in meta_map.values()
        if not m.get("failed_extraction", False)
        and m["path"] not in low_frame_set
    ]
    clips.sort(key=lambda c: c.path)
    log.info(
        f"  Phase 1 done: {len(clips):,} valid | "
        f"{len(failed):,} failed extraction | "
        f"{len(low_frame_paths):,} low-frame quarantined"
    )
    return clips


def phase2_ncc_matrix(clips: list[ClipMeta], cfg: Config) -> np.ndarray:
    n   = len(clips)
    bs  = cfg.ncc_block_size

    log.info(f"  Building {n}×{n} NCC matrix in blocks of {bs}...")
    log.info(f"  NCC matrix size: {n*n*4/1e9:.1f} GB")

    sim  = np.zeros((n, n), dtype=np.float32)
    conf = np.array([c.prnu_confidence for c in clips], dtype=np.float32)

    n_blocks = (n + bs - 1) // bs
    total    = n_blocks * (n_blocks + 1) // 2

    with tqdm(total=total, desc="NCC blocks") as pbar:
        for bi in range(0, n, bs):
            ei      = min(bi + bs, n)
            block_i = _load_block(clips, bi, ei, cfg)

            for bj in range(bi, n, bs):
                ej      = min(bj + bs, n)
                block_j = _load_block(clips, bj, ej, cfg)

                ncc = block_i @ block_j.T
                np.clip(ncc, -1.0, 1.0, out=ncc)
                val = (ncc + 1.0) / 2.0
                val = np.maximum(val, 0.0)

                thresh = cfg.low_confidence_threshold
                damp   = cfg.low_confidence_damping
                for local_i in range(ei - bi):
                    gi = bi + local_i
                    ci = conf[gi]
                    if ci < thresh:
                        alpha = ci / thresh
                        val[local_i, :] = (alpha * val[local_i, :] +
                                           (1 - alpha) * 0.5 * damp)
                for local_j in range(ej - bj):
                    gj = bj + local_j
                    cj = conf[gj]
                    if cj < thresh:
                        alpha = cj / thresh
                        val[:, local_j] = (alpha * val[:, local_j] +
                                           (1 - alpha) * 0.5 * damp)

                sim[bi:ei, bj:ej] = val
                if bi != bj:
                    sim[bj:ej, bi:ei] = val.T

                del block_j
                pbar.update(1)

            del block_i

    np.fill_diagonal(sim, 1.0)
    sim = (sim + sim.T) / 2.0
    np.fill_diagonal(sim, 1.0)
    log.info("  NCC matrix complete")
    return sim


def silhouette_k(sim_matrix: np.ndarray, max_k: int) -> int:
    n     = sim_matrix.shape[0]
    max_k = max(2, min(max_k, n - 1))
    # Convert similarity → distance for silhouette computation
    dist_matrix = 1.0 - sim_matrix
    np.clip(dist_matrix, 0.0, None, out=dist_matrix)  
    np.fill_diagonal(dist_matrix, 0.0)

    best_k     = 2
    best_score = -2.0
    scores     = {}

    log.info(f"  Silhouette k-search: testing k=2..{max_k}")

    for k in range(2, max_k + 1):
        # Run spectral clustering at this k (reuses existing function)
        labels = spectral_cluster(sim_matrix, k)

        # Silhouette requires at least 2 non-empty clusters
        unique = np.unique(labels)
        if len(unique) < 2:
            log.warning(f"  k={k}: degenerate clustering (only 1 cluster formed), skipping")
            scores[k] = -1.0
            continue

        score      = silhouette_score(dist_matrix, labels, metric="precomputed")
        scores[k]  = score
        log.info(f"  k={k:3d}  silhouette={score:.4f}")

        if score > best_score:
            best_score = score
            best_k     = k

    log.info("  Silhouette scores summary:")
    for k, s in sorted(scores.items()):
        marker = "  <-- best" if k == best_k else ""
        log.info(f"    k={k:3d}  score={s:.4f}{marker}")

    log.info(f"  Silhouette -> best k={best_k}  (score={best_score:.4f})")
    return best_k

def spectral_cluster(sim_matrix: np.ndarray, k: int) -> np.ndarray:
    log.info(f"  Spectral clustering k={k}...")
    best_labels, best_score = None, np.inf

    for seed in [42, 7, 13, 99, 2024]:
        try:
            sc = SpectralClustering(
                n_clusters    = k,
                affinity      = "precomputed",
                assign_labels = "kmeans",
                random_state  = seed,
                n_init        = 50,
                eigen_solver  = "arpack",
            )
            labels = sc.fit_predict(sim_matrix)
            score  = sum(
                1.0 - sim_matrix[np.ix_(np.where(labels == c)[0],
                                         np.where(labels == c)[0])].mean()
                for c in range(k) if (labels == c).sum() >= 2
            )
            if score < best_score:
                best_score, best_labels = score, labels.copy()
        except Exception as e:
            log.warning(f"  Spectral seed={seed} failed: {e}")

    if best_labels is None:
        log.error("All spectral attempts failed. Falling back to sequential.")
        best_labels = np.arange(len(sim_matrix)) % k

    return best_labels


def _compute_centroids_streaming(clips: list[ClipMeta], labels: np.ndarray,
                                  k: int, cfg: Config) -> np.ndarray:
    dim    = cfg.prnu_dim
    sums   = np.zeros((k, dim), dtype=np.float64)
    counts = np.zeros(k, dtype=np.int64)
    n      = len(clips)
    bs     = cfg.ncc_block_size

    for s in range(0, n, bs):
        e     = min(s + bs, n)
        block = _load_block(clips, s, e, cfg).astype(np.float64)
        labs  = labels[s:e]
        np.add.at(sums,   labs, block)
        np.add.at(counts, labs, 1)

    active = counts > 0
    sums[active] /= counts[active, np.newaxis]
    norms = np.linalg.norm(sums, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return (sums / norms).astype(np.float32)


def _assign_streaming(clips: list[ClipMeta], centroids: np.ndarray,
                       cfg: Config) -> np.ndarray:
    n      = len(clips)
    bs     = cfg.ncc_block_size
    labels = np.empty(n, dtype=np.int32)
    for s in range(0, n, bs):
        e           = min(s + bs, n)
        block       = _load_block(clips, s, e, cfg)
        labels[s:e] = (block @ centroids.T).argmax(axis=1)
    return labels


def _reassign_singletons(clips: list[ClipMeta], labels: np.ndarray,
                          centroids: np.ndarray, k: int,
                          cfg: Config) -> np.ndarray:
    new    = labels.copy()
    counts = np.bincount(labels, minlength=k)
    singletons = [int(np.where(labels == c)[0][0])
                  for c in range(k) if counts[c] == 1]
    if not singletons:
        return new

    si_idx   = np.array(singletons)
    keys     = [_h5key(clips[i].path) for i in si_idx]
    with h5py.File(cfg.hdf5_cache_path, "r") as h5:
        si_batch = np.stack([h5[k][:] for k in keys], axis=0).astype(np.float32)
    norms    = np.linalg.norm(si_batch, axis=1, keepdims=True)
    norms    = np.where(norms < 1e-10, 1.0, norms)
    si_batch = si_batch / norms
    sims     = si_batch @ centroids.T

    for pos, i in enumerate(singletons):
        c      = int(labels[i])
        row    = sims[pos].copy()
        row[c] = -2.0
        best   = int(np.argmax(row))
        log.info(f"  Singleton cluster {c+1} -> {best+1} (sim={row[best]:.4f})")
        new[i]        = best
        counts[c]    -= 1
        counts[best] += 1

    return new


def _confidence_aware_reassignment(clips: list[ClipMeta], labels: np.ndarray,
                                    centroids: np.ndarray, k: int,
                                    cfg: Config) -> np.ndarray:
    new      = labels.copy()
    low_conf = [i for i, c in enumerate(clips)
                if c.prnu_confidence < cfg.low_confidence_threshold]
    if not low_conf:
        return new

    log.info(f"  Confidence-aware reassignment: {len(low_conf)} clips...")

    dim    = cfg.prnu_dim
    c_sum  = np.zeros((k, dim), dtype=np.float64)
    c_cnt  = np.zeros(k, dtype=np.int64)
    n      = len(clips)
    bs     = cfg.ncc_block_size
    for s in range(0, n, bs):
        e     = min(s + bs, n)
        block = _load_block(clips, s, e, cfg).astype(np.float64)
        np.add.at(c_sum, labels[s:e], block)
        np.add.at(c_cnt, labels[s:e], 1)

    lc_keys  = [_h5key(clips[i].path) for i in low_conf]
    with h5py.File(cfg.hdf5_cache_path, "r") as h5:
        lc_batch = np.stack([h5[k][:] for k in lc_keys], axis=0).astype(np.float32)
    norms    = np.linalg.norm(lc_batch, axis=1, keepdims=True)
    norms    = np.where(norms < 1e-10, 1.0, norms)
    lc_batch = lc_batch / norms

    for pos, i in enumerate(low_conf):
        cur       = int(new[i])
        vec       = lc_batch[pos].astype(np.float64)
        loo_sum   = c_sum.copy()
        loo_cnt   = c_cnt.copy()
        loo_sum[cur] -= vec
        loo_cnt[cur] -= 1

        loo_cents = np.zeros((k, dim), dtype=np.float32)
        for c in range(k):
            if loo_cnt[c] > 0:
                cv = loo_sum[c] / loo_cnt[c]
                nn = np.linalg.norm(cv)
                loo_cents[c] = cv / nn if nn > 1e-10 else cv

        sims   = loo_cents @ vec.astype(np.float32)
        best_c = int(np.argmax(sims))

        if best_c != cur:
            margin = float(sims[best_c]) - float(sims[cur])
            log.info(f"  -> {Path(clips[i].path).name}: "
                     f"conf={clips[i].prnu_confidence:.3f} "
                     f"cluster {cur+1} -> {best_c+1} (margin={margin:+.4f})")
            new[i]        = best_c
            c_sum[cur]   -= vec
            c_cnt[cur]   -= 1
            c_sum[best_c] += vec
            c_cnt[best_c] += 1
        else:
            log.info(f"  -> {Path(clips[i].path).name}: "
                     f"conf={clips[i].prnu_confidence:.3f} stays {cur+1}")

    return new


def phase3_refine(clips: list[ClipMeta], labels: np.ndarray,
                  k: int, cfg: Config) -> np.ndarray:
    log.info(f"  Refinement ({cfg.refinement_rounds} rounds, streaming)...")
    cur = labels.copy()

    centroids = _compute_centroids_streaming(clips, cur, k, cfg)
    cur       = _reassign_singletons(clips, cur, centroids, k, cfg)

    for r in range(cfg.refinement_rounds):
        centroids  = _compute_centroids_streaming(clips, cur, k, cfg)
        new_labels = _assign_streaming(clips, centroids, cfg)
        changed    = int((new_labels != cur).sum())
        cur        = new_labels
        log.info(f"  Round {r+1}: {changed} clips reassigned")
        centroids  = _compute_centroids_streaming(clips, cur, k, cfg)
        cur        = _reassign_singletons(clips, cur, centroids, k, cfg)
        if changed == 0:
            log.info(f"  Converged at round {r+1}.")
            break

    centroids = _compute_centroids_streaming(clips, cur, k, cfg)
    cur       = _confidence_aware_reassignment(clips, cur, centroids, k, cfg)
    return cur


def _quarantine_clips(paths: list[str], output_dir: str,
                      reason: str = "frame_extraction_failed",
                      copy: bool = True) -> None:
    if not paths:
        return

    folder = os.path.join(output_dir, reason)
    os.makedirs(folder, exist_ok=True)

    sorted_paths = sorted(paths, key=chunk_number_from_path)

    action = shutil.copy2 if copy else shutil.move
    verb   = "Copying" if copy else "Moving"
    log.info(f"{verb} {len(sorted_paths)} clip(s) to {folder} (ordered by chunk number)")

    for path in sorted_paths:
        chunk_num = chunk_number_from_path(path)
        dst_name  = f"{chunk_num:08d}_{Path(path).name}"
        dst       = os.path.join(folder, dst_name)
        if os.path.exists(dst):
            dst = os.path.join(folder, f"{chunk_num:08d}_dup_{Path(path).name}")
        action(path, dst)


def write_confidence_report(clips: list[ClipMeta], labels: np.ndarray,
                             output_dir: str, thresh: float) -> None:
    path = os.path.join(output_dir, "prnu_confidence_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{'Clip':<50} {'Cluster':>7} {'Confidence':>10} "
                f"{'Valid Frames':>12} {'Rejected':>8} {'Quality OK':>10}\n")
        f.write("-" * 105 + "\n")
        for i, clip in enumerate(clips):
            flags = []
            if clip.prnu_confidence < thresh: flags.append("LOW_CONF")
            if not clip.prnu_quality_ok:      flags.append("ZERO_VAR")
            f.write(
                f"{Path(clip.path).name:<50} "
                f"{labels[i]+1:>7} "
                f"{clip.prnu_confidence:>10.4f} "
                f"{clip.n_valid_frames:>12} "
                f"{clip.n_rejected_frames:>8} "
                f"{'Yes' if clip.prnu_quality_ok else 'No':>10}"
                + (f"  {', '.join(flags)}" if flags else "") + "\n"
            )
    log.info(f"  Report -> {path}")


def organise_into_source_folders(groups: dict, output_dir: str,
                                  copy: bool) -> None:
    action = shutil.copy2 if copy else shutil.move
    verb   = "Copying" if copy else "Moving"

    for c, clip_list in groups.items():
        if not clip_list:
            continue
        folder = os.path.join(output_dir, f"source_{c+1}")
        os.makedirs(folder, exist_ok=True)
        log.info(f"{verb} {len(clip_list):,} chunks -> {folder}")
        for idx, clip in enumerate(clip_list, 1):
            dst = os.path.join(folder, f"{idx:06d}_{Path(clip.path).name}")
            action(clip.path, dst)

def run_pipeline(cfg: Config):
    os.makedirs(cfg.output_dir, exist_ok=True)

    all_files = sorted(set(
        f for ext in cfg.video_extensions
        for pattern in (f"*{ext}", f"**/*{ext}")
        for f in glob.glob(os.path.join(cfg.input_dir, pattern), recursive=True)
    ))

    if not all_files:
        log.error(f"No video files found in {cfg.input_dir}")
        return

    n_clips = len(all_files)
    log.info("=" * 68)
    log.info("PRNU CLUSTERING  —  Scaled Edition")
    log.info("=" * 68)
    log.info(f"  Clips:    {n_clips:,}")
    log.info(f"  Workers:  {cfg.extraction_workers}")
    log.info(f"  {cfg.ram_report(n_clips)}")

    # ── Phase 1: Extract and cache PRNU ──────────────────────────────────────
    log.info("\n" + "=" * 68)
    log.info("PHASE 1  —  PRNU Extraction (parallel, checkpointed)")
    log.info("=" * 68)
    clips = phase1_extract(all_files, cfg)
    if not clips:
        log.error("No valid clips. Exiting.")
        return

    low_q    = sum(1 for c in clips if not c.prnu_quality_ok)
    low_conf = sum(1 for c in clips if c.prnu_confidence < cfg.low_confidence_threshold)
    log.info(f"  Low-quality PRNU: {low_q}")
    log.info(f"  Low-confidence:   {low_conf}")

    # ── Phase 2: Block-wise NCC similarity matrix ─────────────────────────────
    log.info("\n" + "=" * 68)
    log.info("PHASE 2  —  Block-wise NCC Similarity Matrix")
    log.info("=" * 68)
    sim_matrix = phase2_ncc_matrix(clips, cfg)

    # ── k selection ───────────────────────────────────────────────────────────
    k = cfg.num_cameras
    if k == 0:
        log.info("Auto k-selection via silhouette")
        #k = eigengap_k(sim_matrix, cfg.max_k_auto)
        k = silhouette_k(sim_matrix, cfg.max_k_auto)
        log.info(f"  Auto-selected k={k}")
    elif k > len(clips):
        k = len(clips)

    # ── Phase 3: Spectral clustering ──────────────────────────────────────────
    log.info("\n" + "=" * 68)
    log.info("PHASE 3  —  Spectral Clustering")
    log.info("=" * 68)
    labels = spectral_cluster(sim_matrix, k)
    del sim_matrix

    # ── Phase 4: Iterative refinement ─────────────────────────────────────────
    log.info("\n" + "=" * 68)
    log.info("PHASE 4  —  Iterative Refinement (streaming from HDF5)")
    log.info("=" * 68)
    labels = phase3_refine(clips, labels, k, cfg)

    for i, clip in enumerate(clips):
        clip.cluster = int(labels[i])

    write_confidence_report(clips, labels, cfg.output_dir,
                            cfg.low_confidence_threshold)

    groups: dict[int, list[ClipMeta]] = {c: [] for c in range(k)}
    for clip in clips:
        groups[clip.cluster].append(clip)
    for c in groups:
        groups[c].sort(key=lambda x: x.timestamp)

    log.info("\nOrganising clips into source folders...")
    organise_into_source_folders(groups, cfg.output_dir, cfg.copy_chunks)

    log.info("\nCluster sizes (after low-frame quarantine):")
    for c in range(k):
        log.info(f"  Camera {c+1}: {len(groups[c]):,} clips")

    fd = os.path.join(cfg.output_dir, "frame_extraction_failed")
    n_fail = (
        len(glob.glob(os.path.join(fd, "*")))
        if os.path.exists(fd)
        else 0
    )

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  PRNU resolution : {cfg.prnu_resize[0]}×{cfg.prnu_resize[1]}")
    print(f"  Cameras found   : {k}")
    print()
    print(f"  {'Source':<9} {'Chunks':>7}  {'AvgConf':>8}  {'LowQ':>5}  Folder")
    print("  " + "-" * 50)
    for c in range(k):
        n_ch = len(groups[c])
        ac   = np.mean([cl.prnu_confidence for cl in groups[c]]) if groups[c] else 0.0
        lq   = sum(1 for cl in groups[c] if not cl.prnu_quality_ok)
        print(f"  {c+1:<9} {n_ch:>7}  {ac:>8.3f}  "
              f"{'W'+str(lq) if lq else '-':>5}  source_{c+1}/")
    print(
        f"\n  Quarantined : {n_fail} files"
        + (" -> frame_extraction_failed/" if n_fail else "")
    )
    print(f"  Report      : prnu_confidence_report.txt")
    print(f"  HDF5 cache  : {cfg.hdf5_cache_path}")
    print(f"\nOutput: {os.path.abspath(cfg.output_dir)}")

if __name__ == "__main__":
    run_pipeline(cfg)