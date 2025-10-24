#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RGB quicklook + 8‑bit error maps helper.

Functions:
  - valid mask from dataset (dataset_mask + nodata)
  - per‑channel percentile stretch params (ignoring invalid pixels)
  - write 8‑bit RGB (clean metadata; writes mask)
  - write 8‑bit max‑abs‑diff error map(s), optional zoomed scale

CLI:
  --baseline           Multiband reference image (GeoTIFF)
  --out                Output RGB 8‑bit (optional)
  --error-against      Image to compare against baseline (same shape)
  --err-out-base       Output prefix for error maps (no suffix)
  --err-max-global     Fixed scale 0..N DN for global error map (default 255)
  --err-max-zoom       Fixed scale 0..N DN for an extra zoom map (optional)
  --rgb-order          1‑based band order for RGB (default 3 2 1)
  --rgb-pct            Percentiles for RGB stretch (default 2 98)
"""

from pathlib import Path
import argparse
import numpy as np
import rasterio

RGB_ORDER = [3, 2, 1]  # 1‑based indices for rasterio.read(list)

# ------------------------------
# Valid mask utilities
# ------------------------------

def _valid_mask_from_ds(ds):
    """Return valid mask combining dataset_mask() and nodata (if present)."""
    m = ds.dataset_mask() > 0  # combines internal alpha/nodata
    nd = ds.nodata
    if nd is not None and np.isfinite(nd):
        # mark invalid any pixel equal to nodata (fast path: first band)
        try:
            m &= (ds.read(1) != nd)
        except Exception:
            pass
    return m

# ------------------------------
# RGB percentile stretch params
# ------------------------------

def stretch_params_from_baseline(path, rgb_order=RGB_ORDER, pct=(2, 98)):
    """Compute (lo, hi) per channel ignoring invalid pixels."""
    with rasterio.open(path) as ds:
        bands = ds.read(rgb_order).astype(np.float32)  # (3,H,W)
        mvalid = _valid_mask_from_ds(ds)
        params = []
        for i in range(3):
            vals = bands[i]
            sel = mvalid & np.isfinite(vals)
            v = vals[sel]
            if v.size == 0:
                lo, hi = 0.0, 1.0
            else:
                lo, hi = np.percentile(v, pct)
                if not np.isfinite(lo):
                    lo = 0.0
                if (not np.isfinite(hi)) or hi <= lo:
                    hi = lo + 1.0
            params.append((float(lo), float(hi)))
    return params

# ------------------------------
# Write RGB 8‑bit
# ------------------------------

def write_rgb_8bit(src_path, out_path, params, rgb_order=RGB_ORDER):
    """Write 8‑bit RGB without propagating source nodata; writes valid mask."""

    def stretch8(x, lo, hi):
        y = np.clip((x.astype(np.float32) - lo) / (hi - lo + 1e-9), 0, 1)
        return (y * 255.0).astype(np.uint8)

    with rasterio.open(src_path) as ds:
        assert ds.count >= 3, f"Need ≥3 bands for RGB in {src_path}"
        b = ds.read(rgb_order)
        rgb = np.stack([stretch8(b[i], *params[i]) for i in range(3)], 0)  # (3,H,W) uint8

        meta = ds.meta.copy()
        meta.update(
            driver="GTiff",
            dtype=rasterio.uint8,
            count=3,
            photometric="RGB",
            tiled=True,
            blockxsize=512,
            blockysize=512,
            compress="DEFLATE",
        )
        meta.pop("nodata", None)  # do NOT carry int16 nodata into uint8

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path.as_posix(), "w", **meta) as dst:
            dst.write(rgb)
            # write source valid mask if available
            try:
                dst.write_mask(ds.dataset_mask())
            except Exception:
                pass

# ------------------------------
# 8‑bit error maps from per‑pixel max |Δ| across bands
# ------------------------------

def write_error_max8(a_path, b_path, out_path_base, err_max_global=255, err_max_zoom=None, pct=(2, 98)):
    """
    Generate 8‑bit error map(s) from per‑pixel max abs diff across bands:
      - <base>_ERR8_0_<err_max_global>.tif
      - <base>_ERR8_0_<err_max_zoom>.tif (optional)
    Returns: (global_path, zoom_path or None)
    """
    with rasterio.open(a_path) as a, rasterio.open(b_path) as b:
        A = a.read().astype(np.int32)  # (C,H,W)
        B = b.read().astype(np.int32)
        assert A.shape == B.shape, "Dims/band count must match"

        # valid mask: dataset_mask() AND nodata for both inputs
        maskA = _valid_mask_from_ds(a)
        maskB = _valid_mask_from_ds(b)
        valid = maskA & maskB

        # per‑pixel max |Δ| across bands; zero out invalid
        err = np.max(np.abs(A - B), axis=0).astype(np.float32)
        err[~valid] = 0.0

        def to_err8(err_arr, cap=None):
            if cap is None:
                nz = err_arr[err_arr > 0]
                if nz.size:
                    lo, hi = np.percentile(nz, pct)
                    if not np.isfinite(lo):
                        lo = 0.0
                    if (not np.isfinite(hi)) or hi <= lo:
                        hi = lo + 1.0
                else:
                    lo, hi = 0.0, 1.0
            else:
                lo, hi = 0.0, float(cap)
            e8 = np.clip((err_arr - lo) / (hi - lo + 1e-9), 0, 1) * 255.0
            return e8.astype(np.uint8), int(round(hi))

        meta = a.meta.copy()
        meta.update(
            driver="GTiff",
            count=1,
            dtype=rasterio.uint8,
            photometric="MINISBLACK",
            tiled=True,
            blockxsize=512,
            blockysize=512,
            compress="DEFLATE",
        )
        meta.pop("nodata", None)  # avoid nodata conflicts in uint8

        out_base = Path(out_path_base)
        out_base.parent.mkdir(parents=True, exist_ok=True)

        # GLOBAL output
        err8_g, cap_g = to_err8(err, cap=err_max_global)
        out_g = out_base.with_name(out_base.stem + f"_ERR8_0_{cap_g}.tif")
        with rasterio.open(out_g.as_posix(), "w", **meta) as dst:
            dst.write(err8_g[None, ...])
            try:
                dst.write_mask(valid.astype(np.uint8) * 255)
            except Exception:
                pass
            # helpful stats tags for GIS tools
            dst.update_tags(
                STATISTICS_MINIMUM="0",
                STATISTICS_MAXIMUM="255",
                STATISTICS_MEAN=str(float(err8_g.mean())),
                STATISTICS_STDDEV=str(float(err8_g.std())),
                PIXEL_MINIMUM="0",
                PIXEL_MAXIMUM="255",
            )

        # ZOOM output (optional)
        out_z = None
        if err_max_zoom is not None:
            err8_z, cap_z = to_err8(err, cap=err_max_zoom)
            out_z = out_base.with_name(out_base.stem + f"_ERR8_0_{cap_z}.tif")
            with rasterio.open(out_z.as_posix(), "w", **meta) as dst:
                dst.write(err8_z[None, ...])
                try:
                    dst.write_mask(valid.astype(np.uint8) * 255)
                except Exception:
                    pass
                dst.update_tags(
                    STATISTICS_MINIMUM="0",
                    STATISTICS_MAXIMUM="255",
                    STATISTICS_MEAN=str(float(err8_z.mean())),
                    STATISTICS_STDDEV=str(float(err8_z.std())),
                    PIXEL_MINIMUM="0",
                    PIXEL_MAXIMUM="255",
                )

        return out_g, out_z

# ------------------------------
# CLI
# ------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="RGB quicklook and 8‑bit error maps")
    ap.add_argument("--baseline", required=True, help="Reference multiband image")
    ap.add_argument("--out", help="Output 8‑bit RGB from baseline (optional)")
    ap.add_argument("--error-against", help="Image to compare with baseline (same shape)")
    ap.add_argument("--err-out-base", help="Output prefix for error maps (no suffix). Will create <base>_ERR8_0_<cap>.tif")
    ap.add_argument("--err-max-global", type=int, default=255, help="Fixed scale 0..N DN for global error output")
    ap.add_argument("--err-max-zoom", type=int, default=None, help="Fixed scale 0..N DN for optional zoom output")
    ap.add_argument("--rgb-order", nargs=3, type=int, default=RGB_ORDER, help="1‑based band order for RGB")
    ap.add_argument("--rgb-pct", nargs=2, type=float, default=(2, 98), help="RGB stretch percentiles")
    args = ap.parse_args()

    p = Path(args.baseline)

    if args.out:
        params = stretch_params_from_baseline(p, rgb_order=args.rgb_order, pct=tuple(args.rgb_pct))
        write_rgb_8bit(p, Path(args.out), params, rgb_order=args.rgb_order)

    if args.error_against:
        out_base = Path(args.err_out_base) if args.err_out_base else Path(args.baseline).with_suffix("")
        write_error_max8(
            a_path=args.baseline,
            b_path=args.error_against,
            out_path_base=out_base.as_posix(),
            err_max_global=args.err_max_global,
            err_max_zoom=args.err_max_zoom,
        )
