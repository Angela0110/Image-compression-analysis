#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
JPEG-LS single-band encoder (imagecodecs backend).
- I/O: RAW plane (uint16 little-endian '<u2' or uint8) -> .jls bitstream
- Geometry: expects a contiguous HxW array (row-major)
- Rate: 'near' in [0..255] (0 = lossless)
- Validation: checks RAW file size matches width/height/dtype
"""

import argparse
from pathlib import Path
import numpy as np
import imagecodecs


def main():
    ap = argparse.ArgumentParser(description="Encode one band (JPEG-LS via imagecodecs).")
    ap.add_argument("--in-raw",  required=True, help="Input RAW (row-major, '<u2' for uint16 or 'u1' for uint8)")
    ap.add_argument("--out-jls", required=True, help="Output .jls bitstream path")
    ap.add_argument("--near",    type=int, required=True, help="JPEG-LS NEAR tolerance (0..255; 0 = lossless)")
    ap.add_argument("--dtype",   choices=["uint16","uint8"], required=True, help="Input sample dtype")
    ap.add_argument("--width",   type=int, required=True, help="Width (pixels)")
    ap.add_argument("--height",  type=int, required=True, help="Height (pixels)")
    args = ap.parse_args()

    W, H = int(args.width), int(args.height)
    near = int(args.near)
    if near < 0 or near > 255:
        raise ValueError(f"--near out of range: {near} (expected 0..255)")

    in_path = Path(args.in_raw)
    if not in_path.exists():
        raise FileNotFoundError(f"RAW not found: {in_path}")

    # Validate file size against geometry and dtype
    bpp = 2 if args.dtype == "uint16" else 1
    expected_bytes = H * W * bpp
    actual_bytes = in_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise RuntimeError(
            f"RAW size mismatch: got {actual_bytes} bytes, expected {expected_bytes} "
            f"(H={H}, W={W}, dtype={args.dtype})"
        )

    # Load RAW with explicit endianness/packing
    if args.dtype == "uint16":
        arr = np.fromfile(in_path, dtype="<u2").reshape(H, W)
    else:
        arr = np.fromfile(in_path, dtype="u1").reshape(H, W)

    arr = np.ascontiguousarray(arr)

    # Encode
    jls_bytes = imagecodecs.jpegls_encode(arr, level=near)

    # Write bitstream
    Path(args.out_jls).write_bytes(jls_bytes)


if __name__ == "__main__":
    main()
