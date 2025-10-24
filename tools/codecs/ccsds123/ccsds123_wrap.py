#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CCSDS-123 tile-wise wrapper (hyperspectral), lossless only.
- I/O: multiband GeoTIFF (uint16 / int16 / uint8)
- Tiling: default 512x512 (codec-only RAM measured per tile)
- Rate control: strictly lossless (quantizer_fidelity_control_method = 0)
- Spectral preproc: disabled (codec already exploits spectral redundancy internally)
- Interleave for external codec: bip / bil / bsq (default: bsq)
- External codec: reference CLI (e.g., CNES enc123/dec123), via WSL if requested
- int16 mapping: none (pass-through); encode domain must match external codec dtype
- Output: one JSON line with timing, RAM and bitstream stats (last line)

Add-ons (optional, no-op unless enabled):
- --crop-nodata    : skip tiles that are 100% NoData (based on dataset mask and nodata value)
- --sparse-output  : SPARSE_OK=TRUE on output GeoTIFF so all-NaN/NoData tiles are not written on disk
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
        drive = p[0].lower(); rest = p[2:].replace('\\','/').lstrip('/')
        return f"/mnt/{drive}/{rest}"
    return p.replace('\\','/')

def _template_to_list(cmd_tpl):
    if cmd_tpl is None: return None
    if isinstance(cmd_tpl, (list, tuple)): return list(cmd_tpl)
    if isinstance(cmd_tpl, str): return shlex.split(cmd_tpl)
    raise TypeError("enc-cmd/dec-cmd must be str or list.")

def _np_dtype_from_token(tok: str):
    return {"u16le": np.dtype("<u2"), "s16le": np.dtype("<i2"), "u1": np.dtype("u1"), "u8le": np.dtype("u1")}[tok]

def _write_raw_interleaved(tile_bsq, interleave, out_path, np_dtype_out):
    B, Ht, Wt = tile_bsq.shape
    if interleave == "bsq":
        with open(out_path, "wb") as f: tile_bsq.astype(np_dtype_out, copy=False).tofile(f); return
    if interleave == "bil":
        with open(out_path, "wb") as f:
            for r in range(Ht):
                f.write(tile_bsq[:, r, :].astype(np_dtype_out, copy=False).tobytes(order="C")); return
    if interleave == "bip":
        arr = np.moveaxis(tile_bsq, 0, -1)
        with open(out_path, "wb") as f: arr.astype(np_dtype_out, copy=False).tofile(f); return
    raise ValueError("interleave must be one of: bsq, bil, bip")

def _read_raw_interleaved(in_path, interleave, np_dtype_in, B, Ht, Wt):
    n_expected = B * Ht * Wt
    arr = np.fromfile(in_path, dtype=np_dtype_in)
    if arr.size != n_expected: raise RuntimeError("Unexpected RAW size")
    if interleave == "bsq": return arr.reshape(B, Ht, Wt)
    if interleave == "bil": return np.moveaxis(arr.reshape(Ht, B, Wt), 1, 0)
    if interleave == "bip": return np.moveaxis(arr.reshape(Ht, Wt, B), -1, 0)
    raise ValueError("interleave must be one of: bsq, bil, bip")

_TIME_MAXRSS_RE = re.compile(r"Maximum resident set size.*?:\s*([0-9]+)", re.I)

def _run_with_time_v(cmd_list, run_in_wsl=False):
    # En WSL necesitamos forzar GNU time (soporta -v); el builtin de bash no vale.
    if run_in_wsl:
        time_cmd = ["/usr/bin/time", "-v"]
    else:
        # En host (no WSL): intenta GNU time si existe; si no, cae a 'time -v'
        time_cmd = ["/usr/bin/time", "-v"] if Path("/usr/bin/time").exists() else ["time", "-v"]

    if run_in_wsl:
        # Si ya viene 'wsl' en la plantilla, lo respetamos y metemos time después.
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
    if m:
        peak_bytes = int(m.group(1)) * 1024

    return elapsed, peak_bytes, (p.stdout or ""), (p.stderr or ""), p.returncode


def _bytes_to_mib(nbytes): return None if not nbytes else round(nbytes/(1024*1024), 2)

def main():
    ap = argparse.ArgumentParser(description="CCSDS-123 wrapper (tile-wise, lossless only) with WSL support.")
    ap.add_argument("--in",  dest="inp",  required=True, help="Input multiband GeoTIFF")
    ap.add_argument("--out", dest="out", required=True, help="Output reconstructed GeoTIFF")
    ap.add_argument("--tile", type=int, default=512, help="Square tile size (px)")
    ap.add_argument("--interleave", choices=["bip","bil","bsq"], default="bsq", help="RAW interleave for encoder/decoder")

    # External codec command templates
    ap.add_argument("--enc-cmd", default=None,
                    help=r'Encoder template, e.g. "wsl /path/enc123 -i {in} -o {out} -w {w} -h {h} -b {bands} --mode {mode} --dtype {dtype}"')
    ap.add_argument("--dec-cmd", default=None,
                    help=r'Decoder template, e.g. "wsl /path/dec123 -i {in} -o {out} -w {w} -h {h} -b {bands} --mode {mode} --dtype {dtype}"')
    ap.add_argument("--run-in-wsl", action="store_true", help="Run external binaries under WSL and convert paths")
    ap.add_argument("--wsl-enc", default="/home/user/ccsds123/build/enc123", help="WSL path to encoder if no --enc-cmd")
    ap.add_argument("--wsl-dec", default="/home/user/ccsds123/build/dec123", help="WSL path to decoder if no --dec-cmd")

    # New optional accelerators (no-op unless provided)
    ap.add_argument("--crop-nodata", action="store_true",
                    help="Skip tiles that are 100% NoData (based on dataset mask and nodata DN). Faster.")
    ap.add_argument("--sparse-output", action="store_true",
                    help="Create output GeoTIFF with SPARSE_OK=TRUE (all-NaN/NoData tiles not written on disk).")

    ap.add_argument("--keep-bitstream", default=None, help="Folder to keep per-tile bitstreams")
    ap.add_argument("--tmp-base", default="C:/ccsds123_tmp", help="Base folder for temporaries")
    args = ap.parse_args()

    in_path, out_path = Path(args.inp), Path(args.out)
    bit_dir = Path(args.keep_bitstream) if args.keep_bitstream else None
    if bit_dir: bit_dir.mkdir(parents=True, exist_ok=True)

    # Default command templates (lossless only; no near-lossless params)
    if args.enc_cmd is None:
        enc_cmd_tpl = [
            "wsl", "{wsl_enc}", "-i", "{in}", "-o", "{out}",
            "-w", "{w}", "-h", "{h}", "-b", "{bands}",
            "--mode", "{mode}", "--dtype", "{dtype}"
        ] if args.run_in_wsl else [
            "{wsl_enc}", "-i", "{in}", "-o", "{out}",
            "-w", "{w}", "-h", "{h}", "-b", "{bands}",
            "--mode", "{mode}", "--dtype", "{dtype}"
        ]
    else:
        enc_cmd_tpl = _template_to_list(args.enc_cmd)

    if args.dec_cmd is None:
        dec_cmd_tpl = [
            "wsl", "{wsl_dec}", "-i", "{in}", "-o", "{out}",
            "-w", "{w}", "-h", "{h}", "-b", "{bands}",
            "--mode", "{mode}", "--dtype", "{dtype}"
        ] if args.run_in_wsl else [
            "{wsl_dec}", "-i", "{in}", "-o", "{out}",
            "-w", "{w}", "-h", "{h}", "-b", "{bands}",
            "--mode", "{mode}", "--dtype", "{dtype}"
        ]
    else:
        dec_cmd_tpl = _template_to_list(args.dec_cmd)

    interleave = args.interleave.lower()

    with rasterio.open(in_path) as ds:
        B, H, W = ds.count, ds.height, ds.width
        dtype_str_in = ds.dtypes[0]
        if dtype_str_in not in ("uint16","int16","uint8"):
            raise ValueError(f"Unsupported dtype: {dtype_str_in}")

        meta = ds.meta.copy()
        # ensure tiled output and optionally sparse
        meta.update(
            driver="GTiff",
            dtype=dtype_str_in,
            count=B,
            tiled=True,
            blockxsize=args.tile,
            blockysize=args.tile,
            nodata=ds.nodata
        )
        meta.pop("compress", None); meta.pop("predictor", None)
        if args.sparse_output:
            # Rasterio exposes GDAL's SPARSE_OK via the "sparse_ok" creation option
            meta.update(sparse_ok=True)

        # Choose external dtype token
        if dtype_str_in == "uint16": dtype_token = "u16le"
        elif dtype_str_in == "int16": dtype_token = "s16le"
        else: dtype_token = "u8le"
        np_dtype_raw = _np_dtype_from_token(dtype_token)

        peak_enc = peak_dec = 0
        t_enc_total = t_dec_total = 0.0
        sum_bytes = 0
        tile = args.tile
        nd_value = ds.nodata

        def _tile_is_all_nodata(win: Window) -> bool:
            """True si el tile es 100% NoData según dataset_mask (preferente) o por valor nodata."""
            try:
                m = ds.dataset_mask(window=win)
                if (m == 0).all():
                    return True
            except Exception:
                pass
            if nd_value is not None:
                # comprobación estricta: todas las bandas igual a nd
                for i in range(1, B + 1):
                    arr = ds.read(i, window=win)
                    if not (arr == nd_value).all():
                        return False
                return True
            return False

        base_tmp = Path(args.tmp_base); base_tmp.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="ccsds123_", dir=str(base_tmp)) as tmpd:
            tmpd = Path(tmpd)
            with rasterio.open(out_path, "w", **meta) as dst:
                for y0 in range(0, H, tile):
                    for x0 in range(0, W, tile):
                        tw, th = min(tile, W-x0), min(tile, H-y0)
                        win = Window(x0, y0, tw, th)

                        # --- Fast path: skip 100% NoData tiles ---
                        if args.crop_nodata and _tile_is_all_nodata(win):
                            # Escribimos directamente un bloque nodata (rápido y compatible)
                            if nd_value is None:
                                # Si no hay nodata definida, usamos cero por seguridad (no afecta máscaras)
                                fill_val = 0
                            else:
                                fill_val = nd_value
                            # Creamos un bloque BSQ lleno de nodata
                            blk = (np.zeros((B, th, tw), dtype=dtype_str_in) + np.array(fill_val, dtype=dtype_str_in))
                            dst.write(blk, window=win)
                            # Nada que comprimir/decodificar ni bitstream que contar
                            continue

                        # --- Ruta normal: comprimir/decodificar el tile ---
                        tile_bsq = ds.read(window=win)  # [B, th, tw]

                        raw_in  = tmpd / f"t_x{x0:05d}_y{y0:05d}.raw"
                        raw_out = tmpd / f"t_x{x0:05d}_y{y0:05d}_dec.raw"
                        bitfile = (bit_dir / f"t_x{x0:05d}_y{y0:05d}.bit") if bit_dir else (tmpd / f"t_x{x0:05d}_y{y0:05d}.bit")

                        _write_raw_interleaved(tile_bsq.astype(np_dtype_raw, copy=False), interleave, raw_in, np_dtype_raw)

                        mp = {
                            "in": str(raw_in), "out": str(bitfile),
                            "w": tw, "h": th, "bands": B,
                            "mode": interleave, "dtype": dtype_token,
                            "wsl_enc": args.wsl_enc, "wsl_dec": args.wsl_dec
                        }
                        if args.run_in_wsl:
                            mp["in"]  = _win_to_wsl_path(mp["in"])
                            mp["out"] = _win_to_wsl_path(mp["out"])
                        enc_cmd = [tok.format(**mp) for tok in enc_cmd_tpl]
                        dt_e, pk_e, _, se_e, rc_e = _run_with_time_v(enc_cmd, run_in_wsl=args.run_in_wsl)
                        if rc_e != 0: raise RuntimeError(f"Encoder failed on tile ({x0},{y0}): {se_e}")
                        t_enc_total += dt_e
                        if pk_e: peak_enc = max(peak_enc, int(pk_e))
                        try: sum_bytes += Path(bitfile).stat().st_size
                        except Exception: pass

                        mpd = mp.copy(); mpd["in"], mpd["out"] = str(bitfile), str(raw_out)
                        if args.run_in_wsl:
                            mpd["in"]  = _win_to_wsl_path(mpd["in"])
                            mpd["out"] = _win_to_wsl_path(mpd["out"])
                        dec_cmd = [tok.format(**mpd) for tok in dec_cmd_tpl]
                        dt_d, pk_d, _, se_d, rc_d = _run_with_time_v(dec_cmd, run_in_wsl=args.run_in_wsl)
                        if rc_d != 0: raise RuntimeError(f"Decoder failed on tile ({x0},{y0}): {se_d}")
                        t_dec_total += dt_d
                        if pk_d: peak_dec = max(peak_dec, int(pk_d))

                        rec_bsq = _read_raw_interleaved(raw_out, interleave, np_dtype_raw, B, th, tw)
                        dst.write(rec_bsq, window=win)
                        # Limpieza de temporales del tile
                        try:
                            raw_in.unlink(missing_ok=True)
                            raw_out.unlink(missing_ok=True)
                            if not bit_dir:  # si no guardas bitstreams, borra el .bit también
                                Path(bitfile).unlink(missing_ok=True)
                        except Exception:
                            pass


                # preserva máscara de validez del origen si existe
                try:
                    dst.write_mask(ds.dataset_mask())
                except Exception:
                    pass

    total_pixels = H * W
    bpp_effective_total = (sum_bytes * 8.0) / max(total_pixels, 1)
    print(json.dumps({
        "codec": "ccsds123_ext",
        "mode": "lossless_only",
        "encoder": "external CLI (tile-wise, hyperspectral)",
        "bands": int(B), "dtype": dtype_str_in, "tile": int(tile),
        "bitstream_bytes": int(sum_bytes),
        "bpp_effective_total": float(bpp_effective_total),
        "bpp_effective_per_band": float(bpp_effective_total / max(B, 1)),
        "t_comp_s": float(t_enc_total), "t_dec_s": float(t_dec_total),
        "interleave": interleave,
        "mem_comp_peak_bytes": int(peak_enc) if peak_enc else None,
        "mem_dec_peak_bytes":  int(peak_dec) if peak_dec else None,
        "mem_comp_peak_mb": _bytes_to_mib(peak_enc),
        "mem_dec_peak_mb":  _bytes_to_mib(peak_dec),
    }))
    # Last line: JSON for downstream pipeline

if __name__ == "__main__":
    main()
