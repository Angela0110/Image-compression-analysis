#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tile complexity metrics 

Implements:
  A) Gradient complexity (Redies-style):
     - Per-band finite-difference gradients,
     - Per-pixel max magnitude across bands,
     - Mean/std of the max map.

  B) Fourier-based metrics:
     - Composite power spectrum (sum across bands after mean removal),
     - HF ratio = power above radial cutoff / total power,
     - ps_median, ps_mean,
     - MDF (median frequency) and MNF (mean frequency) from radial profile,
     - alpha: slope of log10(Pr) vs log10(r) in [rmin, rmax] (1/f^alpha behavior).

  C) Delentropy (2D gradient entropy) on a grayscale proxy built as
     the per-pixel max across bands.

Notes:
- Frequencies are cycles/pixel ([-0.5, 0.5)) via np.fft.fftfreq; FFT is fftshifted.
- For FFT, nodata is filled with the band mean to avoid NaNs.
- For gradients, nodata is ignored using NaNs in statistics.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import rasterio


# ----------------------------- I/O helpers -----------------------------

@dataclass
class ImgMeta:
    width: int
    height: int
    bands: int
    dtype: str
    nodata: float | None

def read_stack(path: str) -> tuple[np.ndarray, ImgMeta]:
    """Read a multiband GeoTIFF as float32 array (B, H, W) plus minimal metadata."""
    with rasterio.open(path) as ds:
        arr = ds.read(out_dtype="float32")  # (B,H,W)
        meta = ImgMeta(
            width=ds.width, height=ds.height, bands=ds.count,
            dtype=ds.dtypes[0], nodata=ds.nodata
        )
    return arr, meta


# ----------------------------- Gradients -------------------------------

def finite_diff_grad(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Finite differences (centered inside, forward/backward on borders). img: (H,W) float32."""
    H, W = img.shape
    Gx = np.empty_like(img, dtype=np.float32)
    Gy = np.empty_like(img, dtype=np.float32)

    # Horizontal d/dx
    Gx[:, 1:-1] = (img[:, 2:] - img[:, :-2]) * 0.5
    Gx[:, 0]    = img[:, 1] - img[:, 0]
    Gx[:, -1]   = img[:, -1] - img[:, -2]

    # Vertical d/dy
    Gy[1:-1, :] = (img[2:, :] - img[:-2, :]) * 0.5
    Gy[0, :]    = img[1, :] - img[0, :]
    Gy[-1, :]   = img[-1, :] - img[-2, :]

    return Gx, Gy

def gradient_complexity(arr: np.ndarray, nodata: float | None) -> dict:
    """
    Redies-style gradient complexity for multi-band:
      - |grad| per band,
      - per-pixel max across bands,
      - return mean/std (ignoring nodata).
    """
    B, H, W = arr.shape
    mags = []
    for b in range(B):
        img = arr[b]
        if nodata is not None:
            msk = (img == nodata)
            img = np.where(msk, np.nan, img.astype(np.float32))
        Gx, Gy = finite_diff_grad(img.astype(np.float32, copy=False))
        mag = np.hypot(Gx, Gy)  # sqrt(Gx^2 + Gy^2), stable
        mags.append(mag)
    mags = np.stack(mags, axis=0)  # (B,H,W)
    max_mag = np.nanmax(mags, axis=0)
    return {
        "grad_mean": float(np.nanmean(max_mag)),
        "grad_std":  float(np.nanstd(max_mag)),
    }


# ----------------------------- Fourier ---------------------------------

def fft2_power(img: np.ndarray) -> np.ndarray:
    """2D FFT power (|F|^2), fftshifted. img: (H,W) float64."""
    F = np.fft.fft2(img)
    F = np.fft.fftshift(F)
    P = (F.real * F.real) + (F.imag * F.imag)
    return P

def composite_power(arr: np.ndarray, nodata: float | None) -> np.ndarray:
    """
    Sum of power spectra across bands after nodata fill and mean removal.
    arr: (B,H,W) float32 -> returns (H,W) float64
    """
    B, H, W = arr.shape
    power_sum = np.zeros((H, W), dtype=np.float64)
    for b in range(B):
        img = arr[b].astype(np.float64, copy=False)
        if nodata is not None:
            img = np.where(img == nodata, np.nan, img)
        m = np.nanmean(img)
        if not np.isfinite(m):
            m = 0.0
        img = np.where(np.isnan(img), m, img)
        img = img - np.mean(img)  # zero-mean per band
        power_sum += fft2_power(img)
    return power_sum

def freq_grids(H: int, W: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build fx, fy, and radial frequency r (cycles/pixel), all fftshifted."""
    fy = np.fft.fftfreq(H)  # [-0.5, 0.5)
    fx = np.fft.fftfreq(W)
    FX, FY = np.meshgrid(fx, fy)
    FX = np.fft.fftshift(FX)
    FY = np.fft.fftshift(FY)
    R = np.sqrt(FX * FX + FY * FY)
    return FX, FY, R

def radial_profile(power: np.ndarray, R: np.ndarray, nbins: int = 256
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Radial spectrum: average power per radius bin in [0, rmax]. Returns (r_centers, Pr)."""
    r = R.ravel()
    p = power.ravel()
    rmax = r.max()
    bins = np.linspace(0, rmax, nbins + 1)
    inds = np.digitize(r, bins) - 1
    Pr = np.zeros(nbins, dtype=np.float64)
    C  = np.zeros(nbins, dtype=np.int64)
    for i, val in zip(inds, p):
        if 0 <= i < nbins:
            Pr[i] += val
            C[i]  += 1
    C = np.maximum(C, 1)
    Pr = Pr / C
    r_centers = 0.5 * (bins[:-1] + bins[1:])
    return r_centers, Pr

def fourier_metrics(arr: np.ndarray, nodata: float | None,
                    hf_cut: float = 0.30,
                    nbins_radial: int = 256,
                    alpha_fit_min: float = 0.02,
                    alpha_fit_max: float = 0.45,
                    nbins_theta: int = 36) -> dict:
    """
    Composite FFT metrics:
    - HF ratio: sum of power where r >= hf_cut, normalized by total power.
    - MDF: median frequency (50% cumulative radial power).
    - MNF: mean frequency (sum r*Pr / sum Pr).
    - alpha: slope of log10(Pr) vs log10(r) fitted on [alpha_fit_min, alpha_fit_max].
    Note: 'nbins_theta' is reserved (not used) for potential directional analysis.
    """
    B, H, W = arr.shape
    P = composite_power(arr, nodata=nodata)  # (H,W)
    total_power = float(np.sum(P))
    if not np.isfinite(total_power) or total_power <= 0.0:
        return {
            "hf_ratio": 0.0, "ps_median": 0.0, "ps_mean": 0.0,
            "mdf": 0.0, "mnf": 0.0, "alpha": 0.0,
        }

    ps_med = float(np.median(P))
    ps_mean = float(np.mean(P))

    FX, FY, R = freq_grids(H, W)

    # HF ratio
    hf_power = float(np.sum(P[R >= hf_cut]))
    hf_ratio = hf_power / total_power

    # Radial profile -> MDF, MNF, alpha
    r_centers, Pr = radial_profile(P, R, nbins=nbins_radial)
    cumsum = np.cumsum(Pr)
    mdf = float(np.interp(0.5 * cumsum[-1], cumsum, r_centers))
    mnf = float(np.sum(r_centers * Pr) / np.sum(Pr))

    # alpha fit (avoid r<=0 and zero power)
    mask = (r_centers >= alpha_fit_min) & (r_centers <= alpha_fit_max) & (Pr > 0)
    if np.count_nonzero(mask) >= 5:
        x = np.log10(r_centers[mask])
        y = np.log10(Pr[mask])
        a, b = np.polyfit(x, y, 1)  # y ≈ a*x + b  => slope a ≈ -alpha
        alpha = float(-a)
    else:
        alpha = 0.0

    return {
        "hf_ratio": hf_ratio,
        "ps_median": ps_med,
        "ps_mean": ps_mean,
        "mdf": mdf,
        "mnf": mnf,
        "alpha": alpha,
    }


# ----------------------------- Delentropy ------------------------------

def delentropy_on_maxband(arr: np.ndarray, nodata: float | None,
                          nbins: int = 256, clip_pct: float = 99.0) -> dict:
    """
    Delentropy-like measure:
      - Build grayscale proxy as per-pixel max across bands,
      - Compute (Gx, Gy) via finite differences,
      - 2D histogram in a symmetric clipped range (percentile),
      - Shannon entropy H = -sum p log2 p (bits).
    """
    if nodata is not None:
        gray = np.nanmax(np.where(arr == nodata, np.nan, arr), axis=0)
    else:
        gray = np.max(arr, axis=0)
    m = np.nanmean(gray)
    if not np.isfinite(m):
        m = 0.0
    gray = np.where(np.isnan(gray), m, gray).astype(np.float32)

    Gx, Gy = finite_diff_grad(gray)

    g = np.stack([Gx.ravel(), Gy.ravel()], axis=0)  # (2, N)
    lim = np.percentile(np.abs(g), clip_pct)
    lim = float(lim if lim > 0 else 1.0)
    gx = np.clip(Gx, -lim, lim).ravel()
    gy = np.clip(Gy, -lim, lim).ravel()

    edges = np.linspace(-lim, lim, nbins + 1)
    H2, _, _ = np.histogram2d(gx, gy, bins=[edges, edges], density=False)
    total = H2.sum()
    if total <= 0:
        return {"delentropy_bits": 0.0}
    p = H2 / total
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(p > 0, np.log2(p), 0.0)
    H_bits = float(-np.sum(p * logp))
    return {"delentropy_bits": H_bits}


# ----------------------------- Orchestrator ----------------------------

def compute_all(path: str,
                hf_cut: float = 0.30,
                nbins_radial: int = 256,
                alpha_fit_min: float = 0.02,
                alpha_fit_max: float = 0.45,
                nbins_theta: int = 36,
                delent_bins: int = 256,
                delent_clip_pct: float = 99.0) -> dict:
    """Run all metrics on a single tile path and return a flat dict."""
    arr, meta = read_stack(path)
    grad = gradient_complexity(arr, nodata=meta.nodata)
    freq = fourier_metrics(arr, nodata=meta.nodata, hf_cut=hf_cut,
                           nbins_radial=nbins_radial,
                           alpha_fit_min=alpha_fit_min, alpha_fit_max=alpha_fit_max,
                           nbins_theta=nbins_theta)  # reserved
    de = delentropy_on_maxband(arr, nodata=meta.nodata,
                               nbins=delent_bins, clip_pct=delent_clip_pct)
    out = {
        "path": path,
        "width": meta.width,
        "height": meta.height,
        "bands": meta.bands,
    }
    out.update(grad)
    out.update(freq)
    out.update(de)
    return out


# ----------------------------- CLI ------------------------------------

def main():
    ap = argparse.ArgumentParser(description="High/Low-frequency and gradient complexity metrics for tiles.")
    ap.add_argument("paths", nargs="+", help="GeoTIFF tiles (e.g., 1024x1024, bands B02/B03/B04/B08)")
    ap.add_argument("--hf-cut", type=float, default=0.30, help="HF ratio cutoff in cycles/pixel (default 0.30)")
    ap.add_argument("--radial-bins", type=int, default=256, help="Radial spectrum bins (default 256)")
    ap.add_argument("--alpha-min", type=float, default=0.02, help="Min radius for alpha fit (default 0.02)")
    ap.add_argument("--alpha-max", type=float, default=0.45, help="Max radius for alpha fit (default 0.45)")
    ap.add_argument("--delent-bins", type=int, default=256, help="2D histogram bins for delentropy (default 256)")
    ap.add_argument("--delent-clip", type=float, default=99.0, help="Percentile for gradient clipping (default 99)")
    ap.add_argument("--json", action="store_true", help="Print JSON lines instead of compact text")
    args = ap.parse_args()

    for p in args.paths:
        m = compute_all(
            p,
            hf_cut=args.hf_cut,
            nbins_radial=args.radial_bins,
            alpha_fit_min=args.alpha_min,
            alpha_fit_max=args.alpha_max,
            delent_bins=args.delent_bins,
            delent_clip_pct=args.delent_clip
        )
        if args.json:
            import json
            print(json.dumps(m))
        else:
            print(
                f'{Path(m["path"]).name}: '
                f'grad_mean={m["grad_mean"]:.3f}, '
                f'hf_ratio={m["hf_ratio"]:.4f}, '
                f'MDF={m["mdf"]:.4f}, MNF={m["mnf"]:.4f}, alpha={m["alpha"]:.3f}, '
                f'ps_med={m["ps_median"]:.3e}, ps_mean={m["ps_mean"]:.3e}, '
                f'delentropy_bits={m["delentropy_bits"]:.3f}'
            )

if __name__ == "__main__":
    main()
