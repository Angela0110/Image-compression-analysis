#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PNG lossless per-band wrapper.
- I/O: multiband GeoTIFF <-> per-band PNG (uint16 / uint8)
- Tiling: processes one full band at a time (keeps RAM bounded)
- Rate control: N/A (PNG is lossless). Flags like --cr/--bpp/--quality are accepted but ignored.
- Spectral preproc: disabled (PNG is used as a lossless baseline, per band)
- External libs: imageio (preferred) / Pillow / pypng fallback for read/write
- Peak RAM: sampled in-process via psutil if available
- Output: one JSON line with timing, RAM and bitstream stats (last line)
"""

import argparse, json, os, sys, time
from pathlib import Path
from typing import List

import numpy as np
import rasterio

# --- peak memory sampler (per phase) ---
try:
    import psutil
except Exception:
    psutil = None

def _log(msg: str):
    print(msg, file=sys.stderr, flush=True)

def _total_bytes(path: Path) -> int:
    if not path or not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())

class PeakSampler:
    """Lightweight sampler of process RSS; returns peak in MiB (None if psutil unavailable)."""
    def __init__(self, interval_s: float = 0.01):
        self.interval_s = interval_s
        self._peak = 0
        self._stop = False
        self._thr = None
        self._proc = psutil.Process(os.getpid()) if psutil else None

    def _loop(self):
        import time as _t
        while not self._stop:
            try:
                if self._proc:
                    rss = self._proc.memory_info().rss
                    if rss > self._peak:
                        self._peak = rss
            except Exception:
                pass
            _t.sleep(self.interval_s)

    def __enter__(self):
        if self._proc:
            import threading
            self._thr = threading.Thread(target=self._loop, daemon=True)
            self._thr.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop = True
        if self._thr:
            self._thr.join(timeout=0.2)

    @property
    def peak_mb(self):
        return (self._peak / (1024 * 1024)) if self._peak else None

# --- PNG I/O helpers ---
def _save_png_u16(arr: np.ndarray, path: Path, zlevel: int):
    """
    Write a single-channel PNG (uint16 or uint8). Prefer imageio; fallback to Pillow, then pypng.
    """
    arr = np.ascontiguousarray(arr)
    if arr.dtype not in (np.uint16, np.uint8):
        # Force to uint16 (typical case: int16/float -> clamp/cast).
        arr = arr.astype(np.uint16, copy=False)

    # Try imageio.v3 (Pillow plugin accepts 'compress_level').
    try:
        import imageio.v3 as iio
        iio.imwrite(path.as_posix(), arr, extension=".png", compress_level=int(zlevel))
        return
    except Exception:
        pass

    # Try Pillow
    try:
        from PIL import Image
        mode = "I;16" if arr.dtype == np.uint16 else "L"
        im = Image.fromarray(arr, mode=mode)
        im.save(path.as_posix(), format="PNG", compress_level=int(zlevel))
        return
    except Exception:
        pass

    # Fallback: pypng (expects 16-bit big-endian rows)
    try:
        import png  # pypng
        h, w = int(arr.shape[0]), int(arr.shape[1])
        bitdepth = 16 if arr.dtype == np.uint16 else 8
        with open(path, "wb") as f:
            writer = png.Writer(width=w, height=h, greyscale=True, bitdepth=bitdepth, compression=int(zlevel))
            if arr.dtype == np.uint16:
                writer.write(f, arr.byteswap().tolist())  # big-endian
            else:
                writer.write(f, arr.tolist())
        return
    except Exception as e:
        raise RuntimeError(f"Could not write PNG {path}: {e}")

def _read_png_to_array(path: Path) -> np.ndarray:
    """Read PNG into a numpy array (prefer imageio; fallback to Pillow/pypng)."""
    try:
        import imageio.v3 as iio
        return np.array(iio.imread(path.as_posix()), copy=False)
    except Exception:
        pass
    try:
        from PIL import Image
        im = Image.open(path.as_posix())
        return np.array(im, copy=False)
    except Exception:
        pass
    try:
        import png
        r = png.Reader(filename=path.as_posix())
        w, h, rows, info = r.asDirect()
        bitdepth = info.get("bitdepth", 8)
        planes = info.get("planes", 1)
        arr = np.vstack([np.frombuffer(bytearray(row), dtype=np.uint8) for row in rows])
        arr = arr.reshape(h, w * planes)
        if bitdepth == 16:
            arr = arr.view(">u2").reshape(h, w * planes)
            arr = (arr if planes == 1 else arr.reshape(h, w, planes)).astype(np.uint16).copy()
        else:
            arr = (arr if planes == 1 else arr.reshape(h, w, planes)).astype(np.uint8).copy()
        return arr
    except Exception as e:
        raise RuntimeError(f"Could not read PNG {path}: {e}")

# --- CLI ---
def _parse_args():
    ap = argparse.ArgumentParser(description="PNG lossless wrapper (per-band PNG).")
    ap.add_argument("--in",  dest="inp",  required=True, help="Input GeoTIFF")
    ap.add_argument("--out", dest="out", required=True, help="Output reconstructed GeoTIFF")
    ap.add_argument("--keep-bitstream", dest="bitdir", default=None,
                    help="Directory to store per-band PNG bitstreams")
    ap.add_argument("--zlevel", type=int, default=6, help="Deflate level (0-9)")
    # Accept typical rate flags but ignore them (PNG is lossless).
    ap.add_argument("--cr", type=float, default=None)
    ap.add_argument("--bpp", type=float, default=None)
    ap.add_argument("--quality", type=float, default=None)
    return ap.parse_args()

def main():
    args = _parse_args()
    inp = Path(args.inp)
    out = Path(args.out)
    bitdir = Path(args.bitdir) if args.bitdir else None
    if bitdir:
        bitdir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(inp) as ds:
        B, H, W = ds.count, ds.height, ds.width
        dtype = ds.dtypes[0]  # assume homogeneous
        profile = ds.profile.copy()
        # Keep output lightweight; use a mild GTiff compression to avoid huge temporary files.
        profile.update(driver="GTiff", count=B, compress="DEFLATE", predictor=2, zlevel=1)

        # ---- ENCODE (measure peak RAM) ----
        t0 = time.perf_counter()
        with PeakSampler(interval_s=0.01) as pm_enc:
            bs_paths: List[Path] = []
            for i in range(1, B + 1):
                band = ds.read(i)  # (H, W)
                fname = f"b{i:02d}.png" if B > 1 else "b01.png"
                out_png = (bitdir / fname) if bitdir else Path(out.parent / fname)
                _save_png_u16(band, out_png, args.zlevel)
                bs_paths.append(out_png)
        t_comp_s = time.perf_counter() - t0
        mem_comp_peak_mb = pm_enc.peak_mb if psutil else None

    # ---- DECODE (measure peak RAM) ----
    t0 = time.perf_counter()
    with PeakSampler(interval_s=0.01) as pm_dec:
        bands = []
        for i in range(1, B + 1):
            fname = f"b{i:02d}.png" if B > 1 else "b01.png"
            p = (bitdir / fname) if bitdir else Path(out.parent / fname)
            arr = _read_png_to_array(p)
            if arr.ndim == 3:  # if accidentally saved with >1 channel, take the first
                arr = arr[..., 0]
            # Restore original dtype
            if dtype == "uint16":
                arr = arr.astype(np.uint16, copy=False)
            elif dtype == "uint8":
                arr = arr.astype(np.uint8, copy=False)
            bands.append(arr)
        stack = np.stack(bands, axis=0)  # (B, H, W)

        out.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out, "w", **profile) as dst:
            for i in range(B):
                dst.write(stack[i], i + 1)
    t_dec_s = time.perf_counter() - t0
    mem_dec_peak_mb = pm_dec.peak_mb if psutil else None

    # Total bitstream size (either in bitdir or alongside output)
    bitstream_bytes = _total_bytes(bitdir) if bitdir else _total_bytes(out.parent)

    print(json.dumps({
        "codec": "png_lossless",
        "encoder": "imageio/Pillow/pypng (auto)",
        "zlevel": int(args.zlevel),
        "bitstream_bytes": int(bitstream_bytes),
        "t_comp_s": float(t_comp_s),
        "t_dec_s": float(t_dec_s),
        "mem_comp_peak_mb": float(mem_comp_peak_mb) if mem_comp_peak_mb is not None else None,
        "mem_dec_peak_mb": float(mem_dec_peak_mb) if mem_dec_peak_mb is not None else None
    }))

if __name__ == "__main__":
    main()
