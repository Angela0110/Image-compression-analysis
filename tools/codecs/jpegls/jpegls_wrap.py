#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
JPEG-LS per-band wrapper with optional spectral first-difference (diff1).
- I/O: multiband GeoTIFF (uint16 / int16 / uint8)
- Tiling: whole-image, per-band subprocess (encode_one_band.py / decode_one_band.py)
- Rate control: lossless (NEAR=0) or near-lossless (NEAR>0) via imagecodecs.jpegls
- Spectral preproc: reversible band-to-band diff1.
                    Enabled only in Case B and only in strictly lossless runs
                    to avoid cross-band error propagation (disabled otherwise).
- int16 mapping: int16 -> uint16 by adding +32768 before encode; inverse on decode
- Timing: reports codec-only times and end-to-end (pre/codec/post) breakdown
- Output: one JSON line with timing, RAM and bitstream stats (last line)
"""

import argparse, json, sys, tempfile, os, time
from pathlib import Path
import numpy as np
import rasterio

# --- repo-local helpers ---
root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root))
from tools.common.proc_metrics import run_and_measure, bytes_to_mib

def _clamp_near(level: int) -> int:
    return int(max(0, min(255, int(level))))

def derive_near(args, ds) -> int:
    """
    Choose NEAR (JPEG-LS near-lossless tolerance).
    If bpp/CR are requested, probe band 1 to approximate the target rate.
    Probe is done in the same domain used by the real encode.
    """
    import imagecodecs
    if args.lossless:
        return 0
    if args.nearlossless_eps is not None:
        return _clamp_near(args.nearlossless_eps)
    if args.cr is None and args.bpp is None:
        return 1  # mild default

    H, W = ds.height, ds.width
    band = ds.read(1)

    # Map to codec domain: int16 -> uint16 via +32768; keep others native.
    if ds.dtypes[0] == "int16":
        band_u16 = (band.astype(np.int32) + 32768).astype(np.uint16)
    elif ds.dtypes[0] == "uint16":
        band_u16 = band.astype(np.uint16, copy=False)
    else:
        band_u16 = band.astype(np.uint8, copy=False)
    band_u16 = np.ascontiguousarray(band_u16)

    # Target bpp (per band)
    if args.bpp is not None:
        bpp_target = float(args.bpp)
    else:
        baseline_bpp = 16.0 if ds.dtypes[0] in ("uint16","int16") else 8.0
        bpp_target = baseline_bpp / float(args.cr)

    cand = [0,1,2,3,4,5,6,8,10,12,16,20,24,32,40,48,64,80,96,128,160,192,224,255]
    def size_bpp(n):
        j = imagecodecs.jpegls_encode(band_u16, level=_clamp_near(n))
        return (8.0*len(j))/(H*W)

    best_n, best_err = 0, abs(size_bpp(0) - bpp_target)
    prev_n, prev_bpp = 0, size_bpp(0)
    pick = 0
    for n in cand[1:]:
        cur = size_bpp(n)
        err = abs(cur - bpp_target)
        if err < best_err:
            best_n, best_err, pick = n, err, n
        crossed = (prev_bpp >= bpp_target and cur <= bpp_target) or (prev_bpp <= bpp_target and cur >= bpp_target)
        if crossed:
            lo, hi = prev_n, n
            for _ in range(6):
                mid = (lo + hi) // 2
                curm = size_bpp(mid)
                if abs(curm - bpp_target) < best_err:
                    best_n, best_err, pick = mid, abs(curm - bpp_target), mid
                if curm > bpp_target: lo = mid + 1
                else: hi = mid - 1
            break
        prev_n, prev_bpp = n, cur
    pick = _clamp_near(pick)
    return 1 if (pick == 0 and bpp_target < prev_bpp) else (pick or 1)

# --- reversible spectral diff1 (lossless only) ---
def _diff1_forward(cur: np.ndarray, prev: np.ndarray | None, dtype_str: str) -> np.ndarray:
    """R = X[b] - X[b-1] (mod 2^N). Use only in strictly lossless to avoid error propagation."""
    if prev is None:
        return cur
    if dtype_str == "uint16":
        R = (cur.astype(np.uint32) - prev.astype(np.uint32)) & 0xFFFF
        return R.astype(np.uint16)
    if dtype_str == "int16":
        R = (cur.astype(np.int32) - prev.astype(np.int32))
        return np.clip(R, -32768, 32767).astype(np.int16)
    if dtype_str == "uint8":
        R = (cur.astype(np.uint16) - prev.astype(np.uint16)) & 0xFF
        return R.astype(np.uint8)
    return cur

def _diff1_inverse(R: np.ndarray, prev_recon: np.ndarray | None, dtype_str: str) -> np.ndarray:
    """X[b] = R[b] + X[b-1] (mod 2^N)."""
    if prev_recon is None:
        return R
    if dtype_str == "uint16":
        X = (R.astype(np.uint32) + prev_recon.astype(np.uint32)) & 0xFFFF
        return X.astype(np.uint16)
    if dtype_str == "int16":
        X = (R.astype(np.int32) + prev_recon.astype(np.int32))
        return np.clip(X, -32768, 32767).astype(np.int16)
    if dtype_str == "uint8":
        X = (R.astype(np.uint16) + prev_recon.astype(np.uint16)) & 0xFF
        return X.astype(np.uint8)
    return R

def main():
    ap = argparse.ArgumentParser(description="JPEG-LS wrapper (per-band subprocess) with optional spectral diff1.")
    ap.add_argument("--in",  dest="inp",  required=True, help="Input multiband GeoTIFF")
    ap.add_argument("--out", dest="out", required=True, help="Output reconstructed GeoTIFF")

    # Mode / rate control
    g = ap.add_mutually_exclusive_group(required=False)
    g.add_argument("--nearlossless_eps", type=int)
    g.add_argument("--lossless", action="store_true")
    ap.add_argument("--cr", type=float)
    ap.add_argument("--bpp", type=float)
    ap.add_argument("--quality", type=float)  # compatibility no-op
    ap.add_argument("--keep-bitstream", default=None, help="Folder to keep per-band .jls")

    # Case policy: A=no spectral preproc; B=diff1 only in lossless
    ap.add_argument("--preproc", choices=["none","diff1"], default="none",
                    help="Spectral preprocessor: 'none' (Case A) or 'diff1' (Case B, lossless only)")

    ap.add_argument("--tmp-base", default=None, help="Base folder for temporaries")

    args = ap.parse_args()
    inp = Path(args.inp); out = Path(args.out)
    bit_dir = Path(args.keep_bitstream) if args.keep_bitstream else None
    if bit_dir:
        bit_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(inp) as ds:
        dtype = ds.dtypes[0]; H, W, B = ds.height, ds.width, ds.count
        if dtype not in ("uint16","int16","uint8"):
            raise ValueError(f"Unsupported dtype: {dtype}")

        near = derive_near(args, ds)

        # Policy: disable diff1 for near-lossless to prevent cross-band error propagation.
        if near > 0 and args.preproc == "diff1":
            print("[WARN] Disabling spectral diff1 for near-lossless (NEAR>0) to prevent inter-band error propagation.", file=sys.stderr)
            args.preproc = "none"

        dtype_sub = "uint16" if dtype in ("uint16","int16") else "uint8"

        meta = ds.meta.copy()
        meta.update(driver="GTiff", dtype=dtype, count=B, tiled=True)
        meta.pop("compress", None); meta.pop("predictor", None)

        peak_enc = peak_dec = 0
        t_enc_total = t_dec_total = 0.0          # codec-only
        t_comp_pre = t_dec_post = 0.0            # pre/post (E2E breakdown)
        sum_bytes = 0

        py = sys.executable
        tmp_kwargs = {}
        if args.tmp_base:
            Path(args.tmp_base).mkdir(parents=True, exist_ok=True)
            tmp_kwargs["dir"] = args.tmp_base

        with tempfile.TemporaryDirectory(prefix="jls_", **tmp_kwargs) as tmpd:
            tmpd = Path(tmpd)
            with rasterio.open(out, "w", **meta) as dst:
                prev_band_for_diff = None
                prev_recon_for_inv = None

                for i in range(1, B+1):
                    band = ds.read(i)  # (H,W)

                    # --- PRE (E2E): spectral diff1 + RAW dump ---
                    t0_pre = time.perf_counter()
                    if args.preproc == "diff1":
                        band_proc = _diff1_forward(band, prev_band_for_diff, dtype)
                        prev_band_for_diff = band.copy()
                    else:
                        band_proc = band

                    raw_in = tmpd / f"b{i:02d}.raw"
                    if dtype == "uint16":
                        band_u16 = band_proc.astype(np.uint16, copy=False)
                        band_u16.astype("<u2", copy=False).tofile(raw_in)
                    elif dtype == "int16":
                        band_u16 = (band_proc.astype(np.int32) + 32768).astype(np.uint16)
                        band_u16.astype("<u2", copy=False).tofile(raw_in)
                    else:
                        band_u8 = band_proc.astype(np.uint8, copy=False)
                        band_u8.astype("u1", copy=False).tofile(raw_in)
                    del band_proc
                    t_comp_pre += (time.perf_counter() - t0_pre)

                    jls_path = (bit_dir / f"band_{i:02d}.jls") if bit_dir else (tmpd / f"b{i:02d}.jls")
                    raw_out  = tmpd / f"b{i:02d}_dec.raw"

                    # --- CODEC-ONLY: ENCODE ---
                    enc_cmd = [py, str(Path(__file__).with_name("encode_one_band.py")),
                               "--in-raw", str(raw_in),
                               "--out-jls", str(jls_path),
                               "--near", str(int(near)),
                               "--dtype", dtype_sub,
                               "--width", str(W),
                               "--height", str(H)]
                    t_e, m_e, _, err, rc = run_and_measure(enc_cmd, use_uss=True)
                    if rc != 0:
                        raise RuntimeError(f"Encode failed on band {i}: {err}")
                    t_enc_total += t_e
                    if m_e: peak_enc = max(peak_enc, m_e)
                    try:
                        sum_bytes += jls_path.stat().st_size
                    except Exception:
                        pass

                    # --- CODEC-ONLY: DECODE ---
                    dec_cmd = [py, str(Path(__file__).with_name("decode_one_band.py")),
                               "--in-jls", str(jls_path),
                               "--out-raw", str(raw_out),
                               "--dtype", dtype_sub,
                               "--width", str(W),
                               "--height", str(H)]
                    t_d, m_d, _, err, rc = run_and_measure(dec_cmd, use_uss=True)
                    if rc != 0:
                        raise RuntimeError(f"Decode failed on band {i}: {err}")
                    t_dec_total += t_d
                    if m_d: peak_dec = max(peak_dec, m_d)

                    # --- POST (E2E): read RAW -> original dtype, inverse diff1, write band ---
                    t0_post = time.perf_counter()
                    if dtype == "uint16":
                        rec_u = np.fromfile(raw_out, dtype="<u2").reshape(H, W)
                        rec_X = rec_u.astype(np.uint16, copy=False)
                    elif dtype == "int16":
                        rec_u = np.fromfile(raw_out, dtype="<u2").reshape(H, W).astype(np.uint16, copy=False)
                        rec_s32 = rec_u.astype(np.int32) - 32768
                        rec_X = np.clip(rec_s32, -32768, 32767).astype(np.int16)
                    else:
                        rec_u8 = np.fromfile(raw_out, dtype="u1").reshape(H, W)
                        rec_X = rec_u8.astype(np.uint8, copy=False)

                    if args.preproc == "diff1":
                        rec_X = _diff1_inverse(rec_X, prev_recon_for_inv, dtype)
                        prev_recon_for_inv = rec_X.copy()

                    dst.write(rec_X, i)
                    del rec_X, band
                    t_dec_post += (time.perf_counter() - t0_post)

    # Aggregate timings
    t_comp_e2e = t_comp_pre + t_enc_total          # pre + codec
    t_dec_e2e  = t_dec_total + t_dec_post          # codec + post

    print(json.dumps({
        "codec": "jpegls_subproc",
        "encoder": "imagecodecs.jpegls (per-band subprocess)",
        "preproc": args.preproc,
        "nearlossless_eps": int(near),
        "bitstream_bytes": int(sum_bytes),

        # Codec-only timings (as before)
        "t_comp_s": float(t_enc_total),
        "t_dec_s":  float(t_dec_total),

        # End-to-end breakdown (new)
        "t_comp_pre_s": float(t_comp_pre),
        "t_comp_end2end_s": float(t_comp_e2e),
        "t_dec_post_s": float(t_dec_post),
        "t_dec_end2end_s": float(t_dec_e2e),

        # Peak RAM from subprocesses
        "mem_comp_peak_mb": bytes_to_mib(peak_enc),
        "mem_dec_peak_mb":  bytes_to_mib(peak_dec),
        "mem_comp_peak_bytes": int(peak_enc) if peak_enc else None,
        "mem_dec_peak_bytes":  int(peak_dec) if peak_dec else None
    }))

if __name__ == "__main__":
    main()
