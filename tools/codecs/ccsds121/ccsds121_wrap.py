#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CCSDS-121 tile-wise wrapper with optional spectral first-difference (diff1).
- I/O: multiband GeoTIFF (uint16 / int16 / uint8)
- Tiling: default 512x512
- Rate control: strictly lossless (no near-lossless parameter in CCSDS-121)
- Spectral preproc: reversible band-to-band diff1.
                    Enabled only in Case B and only in strictly lossless runs
                    to avoid cross-band error propagation (disabled otherwise).
- Interleave for external codec: bip / bil / bsq
- External codec: 'aec' (libaec, CCSDS-121.0-B-3), via WSL if requested
- int16 mapping: int16 -> uint16 by adding +32768 before encode; inverse on decode
- Output: one JSON line with timing, RAM and bitstream stats (last line)
"""

import argparse, json, tempfile, sys, shlex, subprocess, time, re
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window

def _win_to_wsl_path(p: str) -> str:
    if not p: return p
    if p.startswith(('/mnt/', '/', '~')): return p
    if len(p) >= 2 and p[1] == ':':
        drive = p[0].lower(); rest = p[2:].replace('\\', '/').lstrip('/')
        return f"/mnt/{drive}/{rest}"
    return p.replace('\\', '/')

def _template_to_list(cmd_tpl):
    if cmd_tpl is None: return None
    if isinstance(cmd_tpl, (list, tuple)): return list(cmd_tpl)
    if isinstance(cmd_tpl, str): return shlex.split(cmd_tpl)
    raise TypeError("enc-cmd/dec-cmd must be str or list.")

def _dtype_to_numpy(dtype_str):
    if dtype_str == "uint16": return np.dtype("<u2")
    if dtype_str == "int16":  return np.dtype("<i2")
    if dtype_str == "uint8":  return np.dtype("u1")
    raise ValueError(f"Unsupported dtype: {dtype_str}")

def _write_raw_interleaved(tile_bsq, interleave, out_path, np_dtype):
    B, Ht, Wt = tile_bsq.shape
    if interleave == "bsq":
        with open(out_path, "wb") as f: tile_bsq.astype(np_dtype, copy=False).tofile(f)
    elif interleave == "bil":
        with open(out_path, "wb") as f:
            for r in range(Ht):
                f.write(tile_bsq[:, r, :].astype(np_dtype, copy=False).tobytes(order="C"))
    elif interleave == "bip":
        arr = np.moveaxis(tile_bsq, 0, -1)
        with open(out_path, "wb") as f: arr.astype(np_dtype, copy=False).tofile(f)
    else:
        raise ValueError("interleave must be one of: bsq, bil, bip")

def _read_raw_interleaved(in_path, interleave, np_dtype, B, Ht, Wt):
    n_expected = B * Ht * Wt
    arr = np.fromfile(in_path, dtype=np_dtype)
    if arr.size != n_expected: raise RuntimeError("Unexpected RAW size")
    if interleave == "bsq": return arr.reshape(B, Ht, Wt)
    if interleave == "bil": return np.moveaxis(arr.reshape(Ht, B, Wt), 1, 0)
    if interleave == "bip": return np.moveaxis(arr.reshape(Ht, Wt, B), -1, 0)
    raise ValueError("interleave must be one of: bsq, bil, bip")

def _diff1_bsq_signed(tile_bsq: np.ndarray) -> np.ndarray:
    X = tile_bsq.view(np.uint16).astype(np.uint32, copy=False)
    R = X.copy(); R[1:, :, :] = (X[1:, :, :] - X[:-1, :, :]) & 0xFFFF
    return R.astype(np.uint16, copy=False).view(np.int16)

def _int1_bsq_signed(R: np.ndarray) -> np.ndarray:
    X = R.view(np.uint16).astype(np.uint32, copy=False)
    for b in range(1, X.shape[0]): X[b, :, :] = (X[b, :, :] + X[b-1, :, :]) & 0xFFFF
    return X.astype(np.uint16, copy=False).view(np.int16)

def _diff1_bsq_unsigned(tile_bsq: np.ndarray) -> np.ndarray:
    R = tile_bsq.astype(np.uint32, copy=False).copy()
    R[1:, :, :] = (R[1:, :, :] - R[:-1, :, :]) & 0xFFFF
    return R.astype(np.uint16, copy=False)

def _int1_bsq_unsigned(R: np.ndarray) -> np.ndarray:
    X = R.astype(np.uint32, copy=False).copy()
    for b in range(1, X.shape[0]): X[b, :, :] = (X[b, :, :] + X[b-1, :, :]) & 0xFFFF
    return X.astype(np.uint16, copy=False)

_TIME_MAXRSS_RE = re.compile(r"Maximum resident set size.*?:\s*([0-9]+)", re.I)

def _run_with_time_v(cmd_list, run_in_wsl=False):
    time_cmd = ["/usr/bin/time","-v"] if Path("/usr/bin/time").exists() else ["time","-v"]
    if run_in_wsl:
        if cmd_list and str(cmd_list[0]).lower() == "wsl":
            cmd = cmd_list[:1] + time_cmd + cmd_list[1:]
        else:
            cmd = ["wsl"] + time_cmd + cmd_list
    else:
        cmd = time_cmd + cmd_list
    t0 = time.perf_counter()
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    elapsed = time.perf_counter() - t0
    peak_bytes = None
    m = _TIME_MAXRSS_RE.search(p.stderr or "")
    if m: peak_bytes = int(m.group(1)) * 1024
    return elapsed, peak_bytes, (p.stdout or ""), (p.stderr or ""), p.returncode

def _bytes_to_mib(nbytes):
    return None if not nbytes else round(nbytes / (1024*1024), 2)

def main():
    ap = argparse.ArgumentParser(description="CCSDS-121 tile-wise wrapper with optional spectral diff1 and WSL support.")
    ap.add_argument("--in",  dest="inp",  required=True, help="Input multiband GeoTIFF")
    ap.add_argument("--out", dest="out", required=True, help="Output reconstructed GeoTIFF")
    ap.add_argument("--tile", type=int, default=512, help="Square tile size (px)")
    ap.add_argument("--interleave", choices=["bip","bil","bsq"], default="bip", help="RAW interleave for external codec")
    ap.add_argument("--preproc", choices=["none","diff1"], default="diff1", help="Spectral preprocessor (lossless-only)")
    ap.add_argument("--nbit", type=int, default=16, help="Bits per sample for 'aec' (-n)")
    ap.add_argument("--enc-cmd", default=None, help='Encoder cmd template (default: "aec -n {nbit} {in} {out}")')
    ap.add_argument("--dec-cmd", default=None, help='Decoder cmd template (default: "aec -d -n {nbit} {in} {out}")')
    ap.add_argument("--keep-bitstream", default=None, help="Folder to keep per-tile .aec")
    ap.add_argument("--tmp-base", default="C:/ccsds121_tmp", help="Base folder for temporaries")
    ap.add_argument("--run-in-wsl", action="store_true", help="Run under WSL and convert paths to /mnt/*")
    ap.add_argument("--validate-14bit", action="store_true", help="Warn if DN exceed 14-bit effective range")
    args = ap.parse_args()

    in_path, out_path = Path(args.inp), Path(args.out)
    bit_dir = Path(args.keep_bitstream) if args.keep_bitstream else None
    if bit_dir: bit_dir.mkdir(parents=True, exist_ok=True)

    if args.enc_cmd is None:
        enc_cmd_tpl = ("wsl aec -n {nbit} {in} {out}" if args.run_in_wsl else "aec -n {nbit} {in} {out}")
    else:
        enc_cmd_tpl = args.enc_cmd
    if args.dec_cmd is None:
        dec_cmd_tpl = ("wsl aec -d -n {nbit} {in} {out}" if args.run_in_wsl else "aec -d -n {nbit} {in} {out}")
    else:
        dec_cmd_tpl = args.dec_cmd
    enc_cmd_tpl = _template_to_list(enc_cmd_tpl); dec_cmd_tpl = _template_to_list(dec_cmd_tpl)

    interleave = args.interleave.lower()
    use_diff1  = (args.preproc == "diff1")

    with rasterio.open(in_path) as ds:
        B, H, W = ds.count, ds.height, ds.width
        dtype_str = ds.dtypes[0]
        if dtype_str not in ("uint16","int16","uint8"): raise ValueError(f"Unsupported dtype: {dtype_str}")
        meta = ds.meta.copy()
        meta.update(driver="GTiff", dtype=dtype_str, count=B, tiled=True, blockxsize=args.tile, blockysize=args.tile)
        meta.pop("compress", None); meta.pop("predictor", None)
        np_dtype = {"uint16": np.dtype("<u2"), "int16": np.dtype("<i2"), "uint8": np.dtype("u1")}[dtype_str]

        if args.validate_14bit:
            sample = ds.read(window=Window(0,0,min(W,1024),min(H,1024)))
            if np.issubdtype(sample.dtype, np.signedinteger):
                if not ((sample >= -8192).all() and (sample <= 8191).all()):
                    print("[WARN] Values exceed signed 14-bit range", file=sys.stderr)
            else:
                if not ((sample >= 0).all() and (sample <= 16383).all()):
                    print("[WARN] Values exceed unsigned 14-bit range", file=sys.stderr)

        peak_enc_bytes = peak_dec_bytes = 0
        t_enc_total = t_dec_total = 0.0
        sum_bytes = 0

        base_tmp = Path(args.tmp_base); base_tmp.mkdir(parents=True, exist_ok=True)
        tile = args.tile

        with tempfile.TemporaryDirectory(prefix="ccsds121_", dir=str(base_tmp)) as tmpd:
            tmpd = Path(tmpd)
            with rasterio.open(out_path, "w", **meta) as dst:
                for y0 in range(0, H, tile):
                    for x0 in range(0, W, tile):
                        tw, th = min(tile, W - x0), min(tile, H - y0)
                        win = Window(x0, y0, tw, th)
                        tile_bsq = ds.read(window=win)

                        if use_diff1:
                            if dtype_str == "uint16": pre_bsq = _diff1_bsq_unsigned(tile_bsq)
                            elif dtype_str == "int16": pre_bsq = _diff1_bsq_signed(tile_bsq)
                            else:
                                pre_bsq = tile_bsq.astype(np.uint16, copy=False)
                                pre_bsq = _diff1_bsq_unsigned(pre_bsq).astype(np.uint8, copy=False)
                        else:
                            pre_bsq = tile_bsq

                        raw_in  = tmpd / f"t_x{x0:05d}_y{y0:05d}.raw"
                        raw_out = tmpd / f"t_x{x0:05d}_y{y0:05d}_dec.raw"
                        bitfile = (bit_dir / f"t_x{x0:05d}_y{y0:05d}.aec") if bit_dir else (tmpd / f"t_x{x0:05d}_y{y0:05d}.aec")
                        _write_raw_interleaved(pre_bsq, interleave, raw_in, np_dtype)

                        mp = {"in": str(raw_in), "out": str(bitfile), "nbit": args.nbit,
                              "w": tw, "h": th, "bands": B, "mode": interleave}
                        if args.run_in_wsl:
                            mp["in"]  = _win_to_wsl_path(mp["in"]); mp["out"] = _win_to_wsl_path(mp["out"])
                        enc_cmd = [tok.format(**mp) for tok in enc_cmd_tpl]
                        dt_e, pk_e, _, se_e, rc_e = _run_with_time_v(enc_cmd, run_in_wsl=args.run_in_wsl)
                        if rc_e != 0: raise RuntimeError(f"Encoder failed on tile ({x0},{y0}): {se_e}")
                        t_enc_total += dt_e; 
                        if pk_e: peak_enc_bytes = max(peak_enc_bytes, int(pk_e))
                        try: sum_bytes += Path(bitfile).stat().st_size
                        except Exception: pass

                        mpd = mp.copy(); mpd["in"], mpd["out"] = str(bitfile), str(raw_out)
                        if args.run_in_wsl:
                            mpd["in"] = _win_to_wsl_path(mpd["in"]); mpd["out"] = _win_to_wsl_path(mpd["out"])
                        dec_cmd = [tok.format(**mpd) for tok in dec_cmd_tpl]
                        dt_d, pk_d, _, se_d, rc_d = _run_with_time_v(dec_cmd, run_in_wsl=args.run_in_wsl)
                        if rc_d != 0: raise RuntimeError(f"Decoder failed on tile ({x0},{y0}): {se_d}")
                        t_dec_total += dt_d; 
                        if pk_d: peak_dec_bytes = max(peak_dec_bytes, int(pk_d))

                        rec_bsq = _read_raw_interleaved(raw_out, interleave, np_dtype, B, th, tw)
                        if use_diff1:
                            if dtype_str == "uint16": rec_bsq = _int1_bsq_unsigned(rec_bsq)
                            elif dtype_str == "int16": rec_bsq = _int1_bsq_signed(rec_bsq)
                            else:
                                rec_bsq = rec_bsq.astype(np.uint16, copy=False)
                                rec_bsq = _int1_bsq_unsigned(rec_bsq).astype(np.uint8, copy=False)

                        dst.write(rec_bsq, window=win)

    total_pixels = W * H
    bpp_effective_total = (sum_bytes * 8.0) / max(total_pixels, 1)
    print(json.dumps({
        "codec": "ccsds121_ext",
        "preproc": "diff1" if use_diff1 else "none",
        "encoder": "aec (libaec CLI, CCSDS-121.0-B-3)",
        "bands": int(B), "dtype": dtype_str, "tile": int(tile),
        "bitstream_bytes": int(sum_bytes),
        "bpp_effective_total": float(bpp_effective_total),
        "bpp_effective_per_band": float(bpp_effective_total / max(B, 1)),
        "t_comp_s": float(t_enc_total), "t_dec_s": float(t_dec_total),
        "interleave": interleave,
        "mem_comp_peak_bytes": int(peak_enc_bytes) if peak_enc_bytes else None,
        "mem_dec_peak_bytes": int(peak_dec_bytes)  if peak_dec_bytes else None,
        "mem_comp_peak_mb": _bytes_to_mib(peak_enc_bytes),
        "mem_dec_peak_mb":  _bytes_to_mib(peak_dec_bytes),
    }))

if __name__ == "__main__":
    main()
