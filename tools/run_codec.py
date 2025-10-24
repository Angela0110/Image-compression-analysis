#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Codec runner: execute wrapper, collect bitstreams, compute metrics per tile, and write CSVs.

Short flow:
  1) Load indices.json (case, asset, items with tile path and optional mask).
  2) For each rate and repetition: run the codec wrapper, capture JSON meta, and store recon.
  3) Write quicklooks (RGB + 8-bit error) if quicklooks module is available.
  4) Compute lightweight metrics (PSNR/SSIM, per-band max |Δ|, etc.).
  5) Compute SAM/SID/LMSE **only for Case B** (skip for Case A).
  6) Write per-run CSV; if reps>1 also write an aggregated means CSV.


"""

from __future__ import annotations
import argparse
import csv
import json
import math
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import rasterio

# ------------------------------
# Logging / shell
# ------------------------------

def log(s: str):
    print(s, flush=True, file=sys.stderr)


def run(cmd: List[str] | str) -> Tuple[int, str, str]:
    """Run a command; return (rc, stdout, stderr) decoded as UTF‑8 (ignore errors)."""
    use_shell = isinstance(cmd, str)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=use_shell)
    out_b, err_b = p.communicate()
    out = (out_b or b"").decode("utf-8", errors="ignore")
    err = (err_b or b"").decode("utf-8", errors="ignore")
    return p.returncode, out, err

# ------------------------------
# Scalar metrics
# ------------------------------

def mse(a: np.ndarray, b: np.ndarray) -> float:
    d = a.astype(np.float64) - b.astype(np.float64)
    return float(np.mean(d * d))


def psnr(a: np.ndarray, b: np.ndarray, data_range: float) -> float:
    m = mse(a, b)
    if m == 0:
        return float("inf")
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(m)


def ssim_global(a: np.ndarray, b: np.ndarray, data_range: float) -> float:
    """Global SSIM (no window) for robustness and speed."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mu_x = float(np.mean(a)); mu_y = float(np.mean(b))
    sigma_x2 = float(np.var(a)); sigma_y2 = float(np.var(b))
    sigma_xy = float(np.mean((a - mu_x) * (b - mu_y)))
    L = data_range
    C1 = (0.01 * L) ** 2; C2 = (0.03 * L) ** 2
    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x**2 + mu_y**2 + C1) * (sigma_x2 + sigma_y2 + C2)
    if den == 0:
        return 1.0
    return max(0.0, min(1.0, num / den))

# ------------------------------
# Data range heuristics
# ------------------------------

def effective_data_range(ds: rasterio.DatasetReader) -> int:
    """Estimate peak (for PSNR/SSIM) based on dtype and bit‑packing.
       uint8 → 255; uint16 (12‑in‑16 multiples) → 4095; full uint16 → 65535;
       int16 (14‑in‑16 signed) → 8191; otherwise observed max(abs))."""
    dtype = ds.dtypes[0]
    if dtype == "uint8":
        return 255
    if dtype == "uint16":
        is_12in16 = True; mx = 0
        for i in range(1, ds.count + 1):
            arr = ds.read(i, out_dtype="uint16")
            mx = max(mx, int(arr.max()))
            if is_12in16 and np.any((arr & 0xF) != 0):
                is_12in16 = False
        if is_12in16 and mx <= 4095 * 16:
            return 4095
        return 65535
    if dtype == "int16":
        is_14in16 = True; mn, mx = 0, 0
        for i in range(1, ds.count + 1):
            arr = ds.read(i, out_dtype="int16")
            mn = min(mn, int(arr.min()))
            mx = max(mx, int(arr.max()))
            if is_14in16 and np.any((arr & 0x3) != 0):
                is_14in16 = False
        if is_14in16 and (mn >= -8192) and (mx <= 8191):
            return 8191
        return int(max(abs(mn), abs(mx)))
    try:
        return int(np.iinfo(np.dtype(dtype)).max)
    except Exception:
        return 65535

# ------------------------------
# Edge magnitude (for LMSE)
# ------------------------------

def sobel_mag(img: np.ndarray) -> np.ndarray:
    """3×3 Sobel gradient magnitude (NumPy only)."""
    img = img.astype(np.float64)
    kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float64)
    ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64)
    pad = 1
    pimg = np.pad(img, ((pad, pad), (pad, pad)), mode="edge")
    H, W = img.shape
    gx = np.zeros_like(img, dtype=np.float64)
    gy = np.zeros_like(img, dtype=np.float64)
    for i in range(3):
        for j in range(3):
            gx += kx[i, j] * pimg[i:i+H, j:j+W]
            gy += ky[i, j] * pimg[i:i+H, j:j+W]
    return np.sqrt(gx * gx + gy * gy)

# ------------------------------
# CSV helpers (decimal comma)
# ------------------------------

def _is_number(x):
    return isinstance(x, (int, float, np.number)) and not isinstance(x, bool)


def _fmt_decimal_comma(x):
    if x is None:
        return ""
    if isinstance(x, float):
        if math.isinf(x):
            return "inf"
        if math.isnan(x):
            return ""
        s = f"{x:.6f}".rstrip("0").rstrip(".")
        return s.replace(".", ",")
    if isinstance(x, (np.floating,)):
        return _fmt_decimal_comma(float(x))
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return str(x)


def format_row_decimal_comma(row: Dict[str, object]) -> Dict[str, str]:
    out = {}
    for k, v in row.items():
        if _is_number(v) or (isinstance(v, float) and (math.isfinite(v) or math.isinf(v))):
            out[k] = _fmt_decimal_comma(v)
        elif isinstance(v, np.generic):
            out[k] = _fmt_decimal_comma(v.item())
        else:
            out[k] = "" if v is None else str(v)
    return out

# ------------------------------
# IO utilities
# ------------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def import_quicklooks():
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        import quicklooks as ql  # type: ignore
        return ql
    except Exception:
        return None


def collect_bitstream_bytes(bit_dir: Path) -> int:
    total = 0
    if bit_dir and bit_dir.exists():
        for p in bit_dir.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
    return total

# ------------------------------
# Lambda from band descriptions
# ------------------------------

def lambdas_from_descriptions(ds: rasterio.DatasetReader) -> np.ndarray | None:
    descs = getattr(ds, "descriptions", None)
    if not descs:
        return None
    vals = []
    import re
    for d in descs:
        if not d:
            vals.append(np.nan); continue
        m = re.search(r"lambda_nm\s*=\s*([0-9.]+)", d)
        vals.append(float(m.group(1)) if m else np.nan)
    arr = np.array(vals, float)
    return arr if np.isfinite(arr).any() else None


def pick_rgb_bands_by_lambda(tif_path: Path, targets_nm=(665.0, 560.0, 490.0)) -> List[int]:
    try:
        with rasterio.open(tif_path) as ds:
            lams = lambdas_from_descriptions(ds)
            if lams is None or not np.isfinite(lams).any():
                return [3, 2, 1]
            def nb(t): return int(np.nanargmin(np.abs(lams - t))) + 1
            return [nb(targets_nm[0]), nb(targets_nm[1]), nb(targets_nm[2])]
    except Exception:
        return [3, 2, 1]


def guess_mask_path(src_path: Path) -> Path | None:
    cand = src_path.with_name(src_path.stem + "_mask").with_suffix(".tif")
    return cand if cand.exists() else None

# ------------------------------
# Metrics core
# ------------------------------

def compute_metrics(ref_path: Path, tst_path: Path, valid: np.ndarray | None = None) -> Dict[str, float]:
    """Compute PSNR/SSIM per band + global PSNR/SSIM and per‑band max|Δ| (fast path)."""
    with rasterio.open(ref_path) as ref, rasterio.open(tst_path) as tst:
        assert ref.count == tst.count and ref.width == tst.width and ref.height == tst.height, \
            "Reference and test must match in size and band count."
        rng = effective_data_range(ref)
        B = ref.count; H, W = ref.height, ref.width

        # Robust valid mask (bitmap)
        vm = (ref.dataset_mask() > 0) & (tst.dataset_mask() > 0)
        if ref.nodata is not None and np.isfinite(ref.nodata):
            nd = ref.nodata
            for i in range(1, B + 1):
                try: vm &= (ref.read(i) != nd)
                except Exception: pass
        if tst.nodata is not None and np.isfinite(tst.nodata):
            nd = tst.nodata
            for i in range(1, B + 1):
                try: vm &= (tst.read(i) != nd)
                except Exception: pass
        if valid is not None:
            if valid.shape != (H, W):
                raise ValueError(f"Mask shape {valid.shape} != {(H, W)}")
            vm &= valid.astype(bool)
        use_mask = np.any(vm)

        psnrs, ssims, maxerrs = [], [], []
        sse_total = 0.0; n_total = 0; rng_obs = 0.0
        for i in range(1, B + 1):
            A = ref.read(i)
            R = tst.read(i)
            if use_mask:
                sel = vm; a = A[sel]; r = R[sel]
            else:
                a = A; r = R
            diff_i32 = np.abs(a.astype(np.int32) - r.astype(np.int32))
            me = int(diff_i32.max()) if diff_i32.size else 0
            maxerrs.append(me)
            p = psnr(a, r, rng) if a.size else float("nan")
            s = ssim_global(a, r, rng) if a.size else float("nan")
            psnrs.append(p); ssims.append(s)
            d = (a.astype(np.float64) - r.astype(np.float64))
            sse_total += float(np.sum(d * d))
            n_total += int(a.size)
            if a.size:
                rng_obs = max(rng_obs, float(np.max(np.abs(a))), float(np.max(np.abs(r))))
        if n_total > 0:
            rng_use = float(max(rng, rng_obs)) if np.isfinite(rng) else float(rng_obs)
            psnr_total = float("inf") if sse_total == 0.0 else (
                20.0 * math.log10(rng_use) - 10.0 * math.log10(sse_total / n_total)
            )
        else:
            psnr_total = float("nan")
        ssim_total = float(np.nanmean(ssims)) if ssims else float("nan")
        out = {
            "psnr_band_avg": float(np.nanmean(psnrs)) if psnrs else float("nan"),
            "ssim_band_avg": float(np.nanmean(ssims)) if ssims else float("nan"),
            "psnr_global": psnr_total,
            "ssim_global": ssim_total,
            "max_abs_err": int(max(maxerrs)) if maxerrs else 0,
            "lossless": 1 if max(maxerrs) == 0 else 0,
        }
        for i, (p, s, me) in enumerate(zip(psnrs, ssims, maxerrs), start=1):
            out[f"psnr_b{i}"] = p; out[f"ssim_b{i}"] = s; out[f"maxerr_b{i}"] = me
        return out

# --- SAM/SID/LMSE (Case B only) ---

def compute_sam_sid_lmse_caseB(ref_path: Path, tst_path: Path, valid: np.ndarray | None = None) -> Dict[str, float]:
    """Compute SAM (deg), SID, and LMSE for Case B. Loads full cubes (C,H,W)."""
    with rasterio.open(ref_path) as ref, rasterio.open(tst_path) as tst:
        B = ref.count; H, W = ref.height, ref.width
        A = ref.read().astype(np.float64)  # (B,H,W)
        R = tst.read().astype(np.float64)
        if valid is not None:
            if valid.shape != (H, W):
                raise ValueError("Mask shape mismatch for Case B metrics")
            vm = valid.astype(bool)
        else:
            vm = (ref.dataset_mask() > 0) & (tst.dataset_mask() > 0)
        # flatten spatial dims
        vm_flat = vm.ravel()
        A2 = A.reshape(B, -1)[:, vm_flat]
        R2 = R.reshape(B, -1)[:, vm_flat]
        n = A2.shape[1]
        if n == 0:
            return {"sam_deg": float("nan"), "sid": float("nan"), "lmse": float("nan")}
        # SAM (deg)
        dot = np.sum(A2 * R2, axis=0)
        na = np.sqrt(np.sum(A2 * A2, axis=0)) + 1e-12
        nr = np.sqrt(np.sum(R2 * R2, axis=0)) + 1e-12
        cosang = np.clip(dot / (na * nr), -1.0, 1.0)
        sam_deg = float(np.degrees(np.mean(np.arccos(cosang))))
        # SID (make spectra positive, normalize per pixel)
        Amin = A2.min(axis=0); Rmin = R2.min(axis=0)
        Ap = A2 - Amin + 1e-12; Rp = R2 - Rmin + 1e-12
        Ap /= np.sum(Ap, axis=0, keepdims=True)
        Rp /= np.sum(Rp, axis=0, keepdims=True)
        sid = float(np.mean(np.sum(Ap * np.log((Ap + 1e-15) / (Rp + 1e-15)), axis=0) +
                             np.sum(Rp * np.log((Rp + 1e-15) / (Ap + 1e-15)), axis=0)))
        # LMSE (Sobel magnitude per band → MSE, then mean over bands)
        lmse_acc = 0.0
        for b in range(B):
            ea = sobel_mag(A[b])
            er = sobel_mag(R[b])
            lmse_acc += mse(ea, er)
        lmse = float(lmse_acc / B)
        return {"sam_deg": sam_deg, "sid": sid, "lmse": lmse}

# ------------------------------
# Indices JSON
# ------------------------------

def load_indices(path: Path) -> Tuple[str, str, List[Dict[str, Path]]]:
    js = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(js, dict) and "items" in js:
        case = js.get("case", "caseA"); asset = js.get("asset", "tile_1024"); items = js["items"]
    elif isinstance(js, list):
        case = "caseA"; asset = "tile_1024"; items = js
    else:
        raise ValueError("Unsupported indices.json format")
    norm = []
    for it in items:
        p = Path(it["path"]).resolve()
        rec = {"tile_id": it["tile_id"], "path": p}
        if "mask" in it and it["mask"]:
            rec["mask"] = Path(it["mask"]).resolve()
        norm.append(rec)
    return case, asset, norm

# ------------------------------
# Main
# ------------------------------

def main():
    ap = argparse.ArgumentParser(description="Codec runner: execute wrappers and collect metrics per tile")
    ap.add_argument("--indices", required=True, help="JSON with tiles (id, path[, mask])")
    ap.add_argument("--codec", required=True, help="Codec label (for CSV)")
    ap.add_argument("--rate-key", default="none", choices=["none", "cr", "bpp", "nearlossless_eps", "quality"],
                    help="Rate control key passed to wrapper; use 'none' for lossless anchor")
    ap.add_argument("--rates", nargs="+", default=None, help="Rate values; omit if --rate-key none")
    ap.add_argument("--outdir", required=True, help="Base output directory")
    ap.add_argument("--compressor-cmd", nargs="+", required=True, help="Wrapper command (e.g., python tools/codecs/png/png_wrap.py)")
    ap.add_argument("--keep-bitstream", action="store_true", help="Keep wrapper bitstreams on disk")
    ap.add_argument("--quicklooks", default=None, help="Path to quicklooks.py; if omitted, try local import")
    ap.add_argument("--case", default=None, help="Override 'case' for CSV")
    ap.add_argument("--asset", default=None, help="Override 'asset' for CSV")
    ap.add_argument("--single-csv", default=None, help="Path to single CSV (default <outdir>/metrics.csv)")
    ap.add_argument("--reps", type=int, default=1, help="Repetitions per rate point")
    # Link parameters (only affect link time estimates)
    ap.add_argument("--caseA-link-mbps", type=float, default=1.0)
    ap.add_argument("--caseA-eff", type=float, default=0.80)
    ap.add_argument("--caseB-link-mbps", type=float, default=None)
    ap.add_argument("--caseB-eff", type=float, default=None)
    # Quicklooks
    ap.add_argument("--ql-err-global", type=int, default=255, help="Fixed scale for global error 0..N DN")
    ap.add_argument("--ql-err-zoom", type=int, default=None, help="Fixed scale 0..N DN for zoom (optional)")
    ap.add_argument("--ql-rgb", action="store_true", help="Also write baseline/recon RGB8 quicklooks")
    args, extra = ap.parse_known_args()
    extra = [x for x in extra if x != "--"]

    outdir = Path(args.outdir).resolve(); ensure_dir(outdir)
    single_csv = Path(args.single_csv).resolve() if args.single_csv else (outdir / "metrics.csv")

    case_name, asset_name, items = load_indices(Path(args.indices))
    if args.case: case_name = args.case
    if args.asset: asset_name = args.asset
    case_key = str(case_name).lower()

    # Link setup
    if case_key in ("caseb", "b"):
        link_mbps = args.caseB_link_mbps if args.caseB_link_mbps is not None else 150.0
        link_eff  = args.caseB_eff       if args.caseB_eff       is not None else 0.80
    else:
        link_mbps = args.caseA_link_mbps
        link_eff  = args.caseA_eff
    Reff_bps = max(1e-9, link_mbps * 1e6 * link_eff)

    # Quicklooks module
    if args.quicklooks:
        ql_path = Path(args.quicklooks)
        if ql_path.exists():
            sys.path.insert(0, ql_path.parent.as_posix())
            try:
                import quicklooks as ql_mod  # type: ignore
            except Exception:
                ql_mod = import_quicklooks()
        else:
            ql_mod = import_quicklooks()
    else:
        ql_mod = import_quicklooks()

    rows: List[Dict[str, object]] = []

    # Normalize rates
    if args.rate_key == "none":
        rates: List[float | int | None] = [None]
    else:
        rates = []
        for r in (args.rates or []):
            try:
                if isinstance(r, str) and ("." in r or "e" in r.lower()):
                    rates.append(float(r))
                else:
                    rates.append(int(r))
            except Exception:
                rates.append(float(r))

    for item in items:
        tile_id = item["tile_id"]; src_path: Path = item["path"]
        assert Path(src_path).exists(), f"Missing {src_path}"
        with rasterio.open(src_path) as ds:
            W, H, B = ds.width, ds.height, ds.count
            dtype = ds.dtypes[0]
        mask_path = item.get("mask") if isinstance(item, dict) else None
        if not mask_path:
            mask_path = guess_mask_path(src_path)
        valid_mask = None
        if mask_path and Path(mask_path).exists():
            try:
                with rasterio.open(mask_path) as m:
                    mv = m.read(1) > 0
                if mv.shape == (H, W):
                    valid_mask = mv
                else:
                    warnings.warn(f"Mask {mask_path} shape mismatch; ignored.")
            except Exception:
                warnings.warn(f"Failed to read mask {mask_path}; ignored.")
        bytes_per_sample = 2 if dtype in ("uint16", "int16") else 1
        container_bytes = int(W * H * B * bytes_per_sample)
        raw16_bytes = int(W * H * B * 16 // 8)

        for r in rates:
            rk = None if args.rate_key == "none" else args.rate_key
            rate_slug = "norate" if rk is None else str(rk).replace(" ", "") + "_" + str(r).replace(".", "p")
            for rep in range(args.reps):
                run_dir = outdir / tile_id / rate_slug / f"rep_{rep+1:02d}"; ensure_dir(run_dir)
                recon_path = run_dir / "recon.tif"
                # Bitstreams dir
                temp_dir_obj = None
                if args.keep_bitstream:
                    bit_dir = run_dir / "bit"
                else:
                    temp_dir_obj = tempfile.TemporaryDirectory(); bit_dir = Path(temp_dir_obj.name)
                # Build wrapper command
                cmd = list(args.compressor_cmd) + ["--in", src_path.as_posix(), "--out", recon_path.as_posix(), "--keep-bitstream", bit_dir.as_posix()] + extra
                if rk is not None:
                    cmd += [f"--{rk}", str(r)]
                # Execute (skip if recon exists)
                if recon_path.exists():
                    log(f"[SKIP] Reusing reconstruction: {recon_path}")
                    rc, out_txt, err_txt = 0, "{}", ""; t_wrap = 0.0
                else:
                    t0 = time.perf_counter(); rc, out_txt, err_txt = run(cmd); t_wrap = time.perf_counter() - t0
                    if rc != 0:
                        raise RuntimeError(f"Wrapper failed ({rc}). Stderr:\n{err_txt}\nStdout:\n{out_txt}")
                # Parse wrapper JSON (last stdout line)
                out_txt = (out_txt or "").strip(); meta: Dict[str, object] = {}
                if out_txt:
                    try: meta = json.loads(out_txt.splitlines()[-1])
                    except Exception as e:
                        log(f"[WARN] Wrapper JSON parse failed. Tail: {out_txt[-500:]} ERROR:{e}")
                # Set NoData on recon to match source (best effort)
                try:
                    with rasterio.open(src_path) as src: nd = src.nodata
                    if nd is not None:
                        with rasterio.open(recon_path, "r+") as dst:
                            if dst.nodata != nd: dst.nodata = nd
                except Exception as e:
                    log(f"[WARN] NoData set failed for {recon_path}: {e}")
                # Quicklooks
                if ql_mod is not None:
                    try:
                        rgb_order = pick_rgb_bands_by_lambda(src_path) if case_key in ("caseb", "b") else [3, 2, 1]
                        if args.ql_rgb:
                            params = ql_mod.stretch_params_from_baseline(src_path.as_posix(), rgb_order=rgb_order)
                            ql_mod.write_rgb_8bit(src_path.as_posix(), (run_dir / "baseline_RGB8.tif").as_posix(), params, rgb_order=rgb_order)
                            ql_mod.write_rgb_8bit(recon_path.as_posix(), (run_dir / "recon_RGB8.tif").as_posix(), params, rgb_order=rgb_order)
                        ql_mod.write_error_max8(a_path=src_path.as_posix(), b_path=recon_path.as_posix(), out_path_base=(run_dir / "recon").as_posix(), err_max_global=int(args.ql_err_global), err_max_zoom=(int(args.ql_err_zoom) if args.ql_err_zoom is not None else None))
                    except Exception as e:
                        log(f"[WARN] Quicklooks failed in {run_dir}: {e}")
                # Metrics (fast)
                met_img = compute_metrics(src_path, recon_path, valid=valid_mask)
                # Case B extras: SAM/SID/LMSE
                if case_key in ("caseb", "b"):
                    try:
                        extra_metrics = compute_sam_sid_lmse_caseB(src_path, recon_path, valid=valid_mask)
                    except Exception as e:
                        log(f"[WARN] SAM/SID/LMSE failed: {e}"); extra_metrics = {"sam_deg": float("nan"), "sid": float("nan"), "lmse": float("nan")}
                    met_img.update(extra_metrics)
                else:
                    met_img.update({"sam_deg": float("nan"), "sid": float("nan"), "lmse": float("nan")})
                # Bitstream size
                bs_bytes = None
                if "bitstream_bytes" in meta and meta["bitstream_bytes"] is not None:
                    try: bs_bytes = int(meta["bitstream_bytes"]) 
                    except Exception: bs_bytes = None
                if bs_bytes is None:
                    bs_bytes = collect_bitstream_bytes(bit_dir)
                # CSV row
                row: Dict[str, object] = {
                    "case": case_name, "asset": asset_name, "codec": args.codec,
                    "rate_key": (rk or ""), "rate_value": ("" if rk is None else r), "tile_id": tile_id,
                    "width": W, "height": H, "bands": B, "in_bytes": container_bytes,
                    "link_mbps": link_mbps, "link_eff": link_eff, "t_wrap_s": t_wrap,
                }
                for k in ("bitstream_bytes", "cr", "bpp", "t_comp_s", "t_dec_s", "mem_comp_peak_mb", "mem_dec_peak_mb", "encoder", "nearlossless_eps", "near", "mem_comp_peak_bytes", "mem_dec_peak_bytes"):
                    if k in meta and meta[k] is not None:
                        row[k] = meta[k]
                if bs_bytes and (bs_bytes > 0):
                    row["bitstream_bytes"] = int(bs_bytes)
                    bpp_band = (bs_bytes * 8.0) / (W * H * B)
                    row["bpp"] = bpp_band
                    row["cr"]  = raw16_bytes / bs_bytes
                    t_link_tile = (8.0 * bs_bytes) / Reff_bps
                    row["t_link_tile_s"] = t_link_tile
                    t_enc = float(meta.get("t_comp_s")) if meta.get("t_comp_s") is not None else None
                    t_dec = float(meta.get("t_dec_s"))  if meta.get("t_dec_s")  is not None else None
                    row["t_e2e_tile_s"] = (t_enc + t_link_tile + t_dec) if (t_enc is not None and t_dec is not None) else (t_wrap + t_link_tile)
                row.update(met_img)
                rows.append(row)
                if temp_dir_obj is not None:
                    try: temp_dir_obj.cleanup()
                    except Exception: pass

    # ------------------------------
    # Write per‑run CSV
    # ------------------------------
    base_cols = [
        "case","asset","codec","encoder","nearlossless_eps",
        "rate_key","rate_value","tile_id",
        "width","height","bands","in_bytes","bitstream_bytes",
        "cr","bpp",
        "psnr_band_avg","ssim_band_avg","psnr_global","ssim_global",
        "max_abs_err","lossless",
        "sam_deg","sid","lmse",
        "t_comp_s","t_dec_s","t_wrap_s","mem_comp_peak_mb","mem_dec_peak_mb",
        "link_mbps","link_eff","t_link_tile_s","t_e2e_tile_s",
        "mem_comp_peak_bytes","mem_dec_peak_bytes",
    ]
    band_cols: List[str] = []
    for i in range(1, 64):
        for k in (f"psnr_b{i}", f"ssim_b{i}", f"maxerr_b{i}"):
            if any((k in r) for r in rows):
                band_cols.append(k)
    header = base_cols + band_cols

    single_csv.parent.mkdir(parents=True, exist_ok=True)
    with single_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore", delimiter=";")
        w.writeheader()
        for row in rows:
            w.writerow(format_row_decimal_comma(row))
    log(f"[OK] Wrote CSV: {single_csv.as_posix()} ({len(rows)} rows)")

    # ------------------------------
    # Aggregate reps (means; IQR for times/RAM only)
    # ------------------------------
    if args.reps and args.reps > 1 and len(rows) > 0:
        from collections import defaultdict
        def _flt(x):
            try:
                v = float(x); return v if math.isfinite(v) else None
            except Exception:
                return None
        def vec(grp, col):
            return [v for v in (_flt(r.get(col)) for r in grp) if v is not None]
        def mean_of(vs):
            return (sum(vs) / len(vs)) if vs else None
        def iqr_only(vs):
            if not vs: return None
            a = np.asarray(vs, dtype=float)
            return float(np.percentile(a, 75) - np.percentile(a, 25))
        band_keys = sorted({k for r in rows for k in r.keys() if k.startswith("psnr_b") or k.startswith("ssim_b") or k.startswith("maxerr_b")})
        def gkey(r):
            return (r.get("case"), r.get("asset"), r.get("codec"), r.get("encoder"), r.get("nearlossless_eps"), r.get("rate_key"), r.get("rate_value"), r.get("tile_id"), r.get("width"), r.get("height"), r.get("bands"), r.get("link_mbps"), r.get("link_eff"))
        groups = defaultdict(list)
        for r in rows: groups[gkey(r)].append(r)
        mean_cols = [
            "case","asset","codec","encoder","nearlossless_eps",
            "rate_key","rate_value","tile_id",
            "width","height","bands","in_bytes",
            "bitstream_bytes_mean","bpp_mean","cr_mean",
            "psnr_band_avg_rep","ssim_band_avg_rep","max_abs_err_mean",
            "psnr_global_rep","ssim_global_rep",
            "sam_deg_rep","sid_rep","lmse_rep",
            "lossless_all",
            "t_comp_s_mean","t_comp_s_iqr",
            "t_dec_s_mean","t_dec_s_iqr",
            "t_e2e_tile_s_mean","t_e2e_tile_s_iqr",
            "t_link_tile_s_mean",
            "mem_comp_peak_mb_mean","mem_comp_peak_mb_iqr",
            "mem_dec_peak_mb_mean","mem_dec_peak_mb_iqr",
            "link_mbps","link_eff","n_reps"
        ] + [k + "_rep" for k in band_keys]
        mean_csv = single_csv.with_name("metrics_mean.csv")
        with mean_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=mean_cols, extrasaction="ignore", delimiter=";")
            w.writeheader()
            for key, grp in groups.items():
                r0 = grp[0]
                W = int(r0["width"]); H = int(r0["height"]); B = int(r0["bands"])
                bs_v = vec(grp, "bitstream_bytes"); bpp_v = vec(grp, "bpp"); cr_v = vec(grp, "cr")
                psnr_v = vec(grp, "psnr_band_avg"); ssim_v = vec(grp, "ssim_band_avg"); mae_v = vec(grp, "max_abs_err")
                psnrT_v = vec(grp, "psnr_global"); ssimT_v = vec(grp, "ssim_global")
                sam_v = vec(grp, "sam_deg"); sid_v = vec(grp, "sid"); lmse_v = vec(grp, "lmse")
                tc_v = vec(grp, "t_comp_s"); td_v = vec(grp, "t_dec_s"); tl_v = vec(grp, "t_link_tile_s"); te_v = vec(grp, "t_e2e_tile_s")
                mcomp_v = vec(grp, "mem_comp_peak_mb"); mdec_v = vec(grp, "mem_dec_peak_mb")
                rowm = {
                    "case": r0.get("case"), "asset": r0.get("asset"), "codec": r0.get("codec"), "encoder": r0.get("encoder"),
                    "nearlossless_eps": r0.get("nearlossless_eps"), "rate_key": r0.get("rate_key"), "rate_value": r0.get("rate_value"),
                    "tile_id": r0.get("tile_id"), "width": W, "height": H, "bands": B, "in_bytes": int(r0.get("in_bytes")),
                    "bitstream_bytes_mean": mean_of(bs_v), "bpp_mean": mean_of(bpp_v), "cr_mean": mean_of(cr_v),
                    "psnr_band_avg_rep": mean_of(psnr_v), "ssim_band_avg_rep": mean_of(ssim_v), "max_abs_err_mean": mean_of(mae_v),
                    "psnr_global_rep": mean_of(psnrT_v), "ssim_global_rep": mean_of(ssimT_v),
                    "sam_deg_rep": mean_of(sam_v), "sid_rep": mean_of(sid_v), "lmse_rep": mean_of(lmse_v),
                    "lossless_all": 1 if all(int(r.get("lossless", 0)) == 1 for r in grp) else 0,
                    "t_comp_s_mean": mean_of(tc_v), "t_comp_s_iqr": iqr_only(tc_v),
                    "t_dec_s_mean": mean_of(td_v),  "t_dec_s_iqr": iqr_only(td_v),
                    "t_e2e_tile_s_mean": mean_of(te_v), "t_e2e_tile_s_iqr": iqr_only(te_v),
                    "t_link_tile_s_mean": mean_of(tl_v),
                    "mem_comp_peak_mb_mean": mean_of(mcomp_v), "mem_comp_peak_mb_iqr": iqr_only(mcomp_v),
                    "mem_dec_peak_mb_mean":  mean_of(mdec_v),  "mem_dec_peak_mb_iqr":  iqr_only(mdec_v),
                    "link_mbps": r0.get("link_mbps"), "link_eff": r0.get("link_eff"), "n_reps": len(grp),
                }
                if rowm["lossless_all"] == 1:
                    rowm.update({"psnr_band_avg_rep": float("inf"), "ssim_band_avg_rep": 1.0, "max_abs_err_mean": 0, "psnr_global_rep": float("inf"), "ssim_global_rep": 1.0})
                for bk in band_keys:
                    vals = vec(grp, bk); rowm[bk + "_rep"] = mean_of(vals)
                w.writerow(format_row_decimal_comma(rowm))
        log(f"[OK] Wrote aggregated CSV (means; IQR for times/RAM): {mean_csv.as_posix()}")

if __name__ == "__main__":
    main()
