#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CCSDS-122 band-by-band wrapper (external CLI).
- I/O: multiband GeoTIFF (uint16 / uint8)
- Tiling: processes one full band at a time (keeps RAM bounded)
- Rate control: per-band bpp; if CR is provided it is converted to per-band bpp
- Spectral preproc: disabled (CCSDS-122 is 2D; comparison kept per-band without spectral transforms)
- External codec: generic BPE-style CLI (rate via '-r {bpp}' when applicable)
- Endianness: raw dumps use explicit little-endian ('<u2' for uint16)
- Output: one JSON line with timing, RAM and bitstream stats (last line)
"""

import argparse, json, tempfile, sys, shlex
from pathlib import Path
import numpy as np
import rasterio

# --- shared helpers (subprocess peak RAM measurement) ---
root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root))
from tools.common.proc_metrics import run_and_measure, bytes_to_mib


def _template_to_list(cmd_tpl):
    """Accept str or list/tuple and always return a token list (may contain placeholders)."""
    if isinstance(cmd_tpl, (list, tuple)):
        return list(cmd_tpl)
    if isinstance(cmd_tpl, str):
        return shlex.split(cmd_tpl)
    raise TypeError("enc-cmd/dec-cmd must be str or list.")


def _drop_rate_flag(tokens):
    """Remove '-r {bpp}' from the template if present (for effective lossless runs)."""
    out = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t == "-r" and (i + 1) < len(tokens):
            i += 2  # skip '-r' and its value token (usually '{bpp}')
            continue
        out.append(t)
        i += 1
    return out


def main():
    ap = argparse.ArgumentParser(description="CCSDS-122 generic CLI wrapper (band-by-band; codec-only RAM).")
    ap.add_argument("--in",  dest="inp",  required=True, help="Input multiband GeoTIFF (uint16/uint8)")
    ap.add_argument("--out", dest="out", required=True, help="Output reconstructed GeoTIFF")

    # Per-band rate control. If --cr is passed, it is converted to per-band bpp.
    ap.add_argument("--bpp", type=float, help="Requested bits-per-pixel PER BAND for the encoder")
    ap.add_argument("--cr",  type=float, help="Target compression ratio (converted to per-band bpp)")

    # CLI templates (placeholders: {in},{out},{w},{h},{bpp})
    ap.add_argument("--enc-cmd", default=None,
                    help=r'Example: C:\tools\bpe\bpe.exe -e {in} -o {out} -r {bpp} -w {w} -h {h} -b 16 -f 0')
    ap.add_argument("--dec-cmd", default=None,
                    help=r'Example: C:\tools\bpe\bpe.exe -d {in} -o {out}')

    ap.add_argument("--keep-bitstream", default=None, help="Folder to store per-band bitstreams (bXX.bit)")
    ap.add_argument("--tmp-base", default="C:/ccsds122_tmp",
                    help="Base folder for temporary files (ASCII-safe paths)")

    args = ap.parse_args()

    in_path  = Path(args.inp)
    out_path = Path(args.out)

    bit_dir = Path(args.keep_bitstream) if args.keep_bitstream else None
    if bit_dir:
        bit_dir.mkdir(parents=True, exist_ok=True)

    # ---- read metadata and prepare incremental output ----
    with rasterio.open(in_path) as ds:
        B, H, W = ds.count, ds.height, ds.width
        dtype_str = ds.dtypes[0]
        if dtype_str not in ("uint16", "uint8"):
            raise ValueError(f"Unsupported dtype: {dtype_str}. Expected uint16/uint8.")

        meta = ds.meta.copy()
        meta.update(
            driver="GTiff",
            dtype=dtype_str,
            count=B,
            tiled=True,
            blockxsize=512,
            blockysize=512,
        )
        meta.pop("compress", None)
        meta.pop("predictor", None)

        # Per-band target bpp
        bits_per_sample = 16.0 if dtype_str == "uint16" else 8.0
        if args.bpp is not None:
            target_bpp_band = float(args.bpp)
        elif args.cr is not None:
            # bpp_total_in = bits_per_sample * B; target_total = /CR; per-band = /B
            target_bpp_band = (bits_per_sample * B / max(args.cr, 1e-6)) / B
        else:
            target_bpp_band = bits_per_sample  # effectively lossless

        # Consider "lossless requested" if target bpp >= native bits per sample
        lossless_req = target_bpp_band >= (bits_per_sample - 1e-9)

        # Default templates (token lists with placeholders)
        sample_bits_flag = "16" if dtype_str == "uint16" else "8"

        if not args.enc_cmd:
            enc_cmd_tpl = [
                r"C:\tools\bpe\bpe.exe",
                "-e", "{in}", "-o", "{out}",
                "-r", "{bpp}",              # removed if lossless_req
                "-w", "{w}", "-h", "{h}",
                "-b", sample_bits_flag,
            ]
            if lossless_req:
                enc_cmd_tpl = _drop_rate_flag(enc_cmd_tpl)  # do not pass -r in effective lossless
        else:
            enc_cmd_tpl = _template_to_list(args.enc_cmd)

        if not args.dec_cmd:
            dec_cmd_tpl = [
                r"C:\tools\bpe\bpe.exe",
                "-d", "{in}", "-o", "{out}",
                "-w", "{w}", "-h", "{h}",
                "-b", sample_bits_flag,
            ]
        else:
            dec_cmd_tpl = _template_to_list(args.dec_cmd)

        peak_enc = 0
        peak_dec = 0
        t_enc_total = 0.0
        t_dec_total = 0.0
        sum_bytes = 0

        base_tmp = Path(args.tmp_base)
        base_tmp.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="ccsds122_", dir=str(base_tmp)) as tmpd:
            tmpd = Path(tmpd)

            # Open output and write band-by-band (avoids large full-image buffers)
            with rasterio.open(out_path, "w", **meta) as dst:
                for i in range(1, B + 1):
                    # ---- load one band into RAM ----
                    band = ds.read(i)
                    raw_in  = tmpd / f"b{i:02d}.raw"
                    raw_out = tmpd / f"b{i:02d}_dec.raw"
                    bitfile = (bit_dir / f"b{i:02d}.bit") if bit_dir else (tmpd / f"b{i:02d}.bit")

                    # Unambiguous endianness
                    if dtype_str == "uint16":
                        band.astype("<u2", copy=False).tofile(raw_in)
                    else:
                        band.astype("u1",  copy=False).tofile(raw_in)
                    del band  # free ASAP

                    # ----- ENCODE -----
                    mp = {"in": str(raw_in), "out": str(bitfile), "w": W, "h": H, "bpp": float(target_bpp_band)}
                    enc_cmd = [tok.format(**mp) for tok in enc_cmd_tpl]
                    t_e, m_e, _, err, rc = run_and_measure(enc_cmd, poll_interval=0.01, use_uss=True)
                    if rc != 0:
                        raise RuntimeError(f"Encoder failed on band {i}: {err}")
                    t_enc_total += t_e
                    if m_e: peak_enc = max(peak_enc, m_e)

                    try:
                        sum_bytes += Path(bitfile).stat().st_size
                    except Exception:
                        pass

                    # ----- DECODE -----
                    mpd = {"in": str(bitfile), "out": str(raw_out), "w": W, "h": H, "bpp": float(target_bpp_band)}
                    dec_cmd = [tok.format(**mpd) for tok in dec_cmd_tpl]
                    t_d, m_d, _, err, rc = run_and_measure(dec_cmd, poll_interval=0.01, use_uss=True)
                    if rc != 0:
                        raise RuntimeError(f"Decoder failed on band {i}: {err}")
                    t_dec_total += t_d
                    if m_d: peak_dec = max(peak_dec, m_d)

                    # ----- Read decoded RAW and write band -----
                    if dtype_str == "uint16":
                        data = np.fromfile(raw_out, dtype="<u2").reshape(H, W)
                    else:
                        data = np.fromfile(raw_out, dtype="u1").reshape(H, W)
                    dst.write(data, i)
                    del data

    # ---- JSON (last line) ----
    print(json.dumps({
        "codec": "ccsds122_ext",
        "encoder": "external CLI (band-by-band)",
        "bands": int(B),
        "bpp_target_band": float(target_bpp_band),
        "bitstream_bytes": int(sum_bytes),
        "t_comp_s": float(t_enc_total),
        "t_dec_s": float(t_dec_total),
        "mem_comp_peak_mb": bytes_to_mib(peak_enc),
        "mem_dec_peak_mb":  bytes_to_mib(peak_dec),
        "mem_comp_peak_bytes": int(peak_enc) if peak_enc else None,
        "mem_dec_peak_bytes":  int(peak_dec) if peak_dec else None
    }))


if __name__ == "__main__":
    main()
