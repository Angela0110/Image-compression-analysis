#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
JPEG-LS single-band decoder (imagecodecs backend).
- I/O: .jls bitstream -> RAW plane (uint16/uint8, HxW)
- Channels: if the decoded array has multiple channels, keep the first
- Endianness: RAW output is explicit little-endian ('<u2' for uint16)
- Validation: checks decoded shape against (--height, --width)
"""

import argparse
from pathlib import Path
import numpy as np
import imagecodecs


def main():
    ap = argparse.ArgumentParser(description="Decode one band (JPEG-LS via imagecodecs).")
    ap.add_argument("--in-jls",  required=True, help="Input .jls bitstream")
    ap.add_argument("--out-raw", required=True, help="Output RAW path")
    ap.add_argument("--dtype",   choices=["uint16","uint8"], required=True, help="Output sample dtype")
    ap.add_argument("--width",   type=int, required=True, help="Output width (pixels)")
    ap.add_argument("--height",  type=int, required=True, help="Output height (pixels)")
    args = ap.parse_args()

    W, H = int(args.width), int(args.height)

    # Decode
    dec = imagecodecs.jpegls_decode(Path(args.in_jls).read_bytes())
    dec = np.asarray(dec)

    # If decoded has channels, take the first one
    if dec.ndim == 3:
        dec = dec[..., 0]
    elif dec.ndim != 2:
        raise RuntimeError(f"Unexpected decoded ndim={dec.ndim}; expected 2D or 3D.")

    # Validate spatial dims (some decoders may return WxH vs HxW; imagecodecs returns HxW)
    if dec.shape != (H, W):
        raise RuntimeError(f"Shape mismatch: got {dec.shape}, expected {(H, W)}.")

    # Cast and write RAW with explicit endianness/packing
    if args.dtype == "uint16":
        dec = np.ascontiguousarray(dec.astype(np.uint16, copy=False))
        dec.astype("<u2", copy=False).tofile(args.out_raw)
    else:  # uint8
        dec = np.ascontiguousarray(dec.astype(np.uint8, copy=False))
        dec.astype("u1", copy=False).tofile(args.out_raw)


if __name__ == "__main__":
    main()
