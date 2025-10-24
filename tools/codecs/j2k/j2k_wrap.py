#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
JPEG 2000 (JP2OpenJPEG via gdal_translate) wrapper.
- I/O: multiband GeoTIFF <-> JP2 (OpenJPEG through GDAL)
- Tiling: optional JP2 tiling via GDAL creation options (TILEXSIZE/TILEYSIZE)
- Rate control: lossy via QUALITY (1..100); lossless via REVERSIBLE=YES
- Spectral preproc: disabled (transform is 2D; bands are encoded as provided by GDAL)
- External codec: GDAL 'gdal_translate' (or osgeo_utils.gdal_translate fallback)
- Output: one JSON line with timing, RAM and bitstream stats (last line)
"""

import argparse, json, shutil, sys
from pathlib import Path

# --- shared helpers (subprocess peak RAM measurement) ---
root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root))
from tools.common.proc_metrics import run_and_measure, bytes_to_mib


def find_gdal_translate():
    """Return ['gdal_translate'] if found in PATH, otherwise Python-module fallback."""
    exe = shutil.which("gdal_translate")
    if exe:
        return [exe]
    # Common fallback (conda/OSGeo)
    return [sys.executable, "-m", "osgeo_utils.gdal_translate"]


def quality_from_cr(cr: float) -> int:
    """Heuristic mapping CR -> QUALITY (≈ 100/CR, clamped)."""
    q = int(round(100.0 / max(cr, 1e-6)))
    return max(5, min(95, q))


def quality_from_bpp(bpp_band: float) -> int:
    """Heuristic mapping per-band bpp -> QUALITY."""
    if bpp_band >= 4.0:  return 80
    if bpp_band >= 3.0:  return 70
    if bpp_band >= 2.0:  return 60
    if bpp_band >= 1.5:  return 55
    if bpp_band >= 1.0:  return 45
    if bpp_band >= 0.75: return 38
    if bpp_band >= 0.5:  return 32
    return 28


def main():
    ap = argparse.ArgumentParser(description="JPEG 2000 (JP2OpenJPEG) wrapper with peak RAM and bitstream size.")
    ap.add_argument("--in",  dest="inp",  required=True, help="Input GeoTIFF")
    ap.add_argument("--out", dest="out", required=True, help="Output GeoTIFF")

    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--cr",      type=float, help="Target CR (heuristically mapped to QUALITY)")
    g.add_argument("--bpp",     type=float, help="Target per-band bpp (heuristically mapped to QUALITY)")
    g.add_argument("--quality", type=int,   help="QUALITY 1..100 (JP2OpenJPEG)")

    ap.add_argument("--lossless", action="store_true", help="Use REVERSIBLE=YES (ignores QUALITY)")
    ap.add_argument("--keep-bitstream", default=None,  help="Directory to store stream.jp2")
    ap.add_argument("--tilex", type=int, default=None, help="JP2 tile width")
    ap.add_argument("--tiley", type=int, default=None, help="JP2 tile height")
    args = ap.parse_args()

    gdalt = find_gdal_translate()

    bit_dir = Path(args.keep_bitstream) if args.keep_bitstream else None
    if bit_dir:
        bit_dir.mkdir(parents=True, exist_ok=True)
        jp2 = bit_dir / "stream.jp2"
        keep = True
    else:
        jp2 = Path(Path(args.out).parent / "tmp_stream.jp2")
        jp2.parent.mkdir(parents=True, exist_ok=True)
        keep = False

    # ---- ENCODE: GTiff -> JP2 ----
    enc_cmd = gdalt + ["-q", "-of", "JP2OpenJPEG", str(args.inp), str(jp2)]
    if args.tilex and args.tiley:
        enc_cmd += ["-co", f"TILEXSIZE={int(args.tilex)}", "-co", f"TILEYSIZE={int(args.tiley)}"]

    q_used = None
    if args.lossless:
        enc_cmd += ["-co", "REVERSIBLE=YES"]
    else:
        if args.quality is not None:
            q_used = int(args.quality)
        elif args.cr is not None:
            q_used = quality_from_cr(args.cr)
        elif args.bpp is not None:
            q_used = quality_from_bpp(args.bpp)
        else:
            q_used = 35
        enc_cmd += ["-co", f"QUALITY={q_used}"]

    t_enc, mem_enc, _, err, rc = run_and_measure(enc_cmd, use_uss=True)
    if rc != 0:
        raise RuntimeError(err)

    # ---- DECODE: JP2 -> GTiff ----
    dec_cmd = gdalt + ["-q", str(jp2), str(args.out), "-of", "GTiff", "-co", "TILED=YES"]
    t_dec, mem_dec, _, err, rc = run_and_measure(dec_cmd, use_uss=True)
    if rc != 0:
        raise RuntimeError(err)

    # Bitstream size
    try:
        bitstream_bytes = jp2.stat().st_size
    except Exception:
        bitstream_bytes = 0

    if not keep:
        try:
            jp2.unlink(missing_ok=True)
        except Exception:
            pass

    print(json.dumps({
        "codec": "j2k_gdal",
        "encoder": "gdal_translate JP2OpenJPEG",
        "quality_used": int(q_used) if q_used is not None else None,
        "bitstream_bytes": int(bitstream_bytes),
        "t_comp_s": float(t_enc),
        "t_dec_s": float(t_dec),
        "mem_comp_peak_mb": bytes_to_mib(mem_enc),
        "mem_dec_peak_mb":  bytes_to_mib(mem_dec),
        "mem_comp_peak_bytes": int(mem_enc) if mem_enc is not None else None,
        "mem_dec_peak_bytes":  int(mem_dec) if mem_dec is not None else None
    }))
    # Last line: JSON for downstream pipeline

if __name__ == "__main__":
    main()
