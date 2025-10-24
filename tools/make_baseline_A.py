#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

# --- INPUTS (Sentinel-2 10 m JP2) ---
b2 = Path("data/raw/T29TNH_20250809T113319_B02_10m.jp2")  # Blue
b3 = Path("data/raw/T29TNH_20250809T113319_B03_10m.jp2")  # Green
b4 = Path("data/raw/T29TNH_20250809T113319_B04_10m.jp2")  # Red
b8 = Path("data/raw/T29TNH_20250809T113319_B08_10m.jp2")  # NIR
bands = [b2, b3, b4, b8]

# --- PARAMS ---
SCENE_W, SCENE_H = 2000, 10000           # Case A scene (20 km x 100 km at 10 m/px)
TILE_W, TILE_H   = 1024, 1024            # Tiles for sweeps (inside the scene)

# Refined offsets (measured on the scene)
HC_COL_OFF, HC_ROW_OFF = 300, 688
LC_COL_OFF, LC_ROW_OFF = 488, 7012


# --- HELPERS ---------------------------------------------------------------

def _as_path(p: str | Path) -> Path:
    return p if isinstance(p, Path) else Path(p)

def _assert_inputs_exist(paths):
    missing = [p for p in paths if not _as_path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing input(s): {', '.join(map(str, missing))}")

def write_window_stack(out_path: str | Path,
                       window_w: int,
                       window_h: int,
                       col_off: int | None = None,
                       row_off: int | None = None) -> Path:
    """
    Cut a window directly from the JP2 sources and save a stacked multiband GeoTIFF.
    If offsets are None, the window is centered inside the full image.
    """
    _assert_inputs_exist(bands)
    out_path = _as_path(out_path)

    with rasterio.open(bands[0]) as ref:
        W, H = ref.width, ref.height
        if col_off is None:
            col_off = max(0, (W - window_w) // 2)
        if row_off is None:
            row_off = max(0, (H - window_h) // 2)
        col_off = min(col_off, max(0, W - window_w))
        row_off = min(row_off, max(0, H - window_h))
        win = Window(col_off=col_off, row_off=row_off, width=window_w, height=window_h)
        transform = rasterio.windows.transform(win, ref.transform)

        # Basic consistency checks across bands
        for p in bands:
            with rasterio.open(p) as s:
                assert s.width == ref.width and s.height == ref.height, f"Different size in {p}"
                assert s.crs == ref.crs, f"Different CRS in {p}"
                assert s.transform == ref.transform, f"Different transform in {p}"

        meta = ref.meta.copy()
        meta.update(
            driver="GTiff",
            count=len(bands),
            dtype="uint16",
            height=window_h,
            width=window_w,
            transform=transform,
            tiled=True,
            blockxsize=512,
            blockysize=512,
            BIGTIFF="IF_SAFER",
            crs=ref.crs,
            nodata=ref.nodata,
            predictor=2,   # only effective if compression is used; keeping for consistency
        )
        meta.pop("compress", None)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **meta) as dst:
            for i, p in enumerate(bands, start=1):
                with rasterio.open(p) as src:
                    data = src.read(1, window=win).astype(np.uint16, copy=False)
                    dst.write(data, i)

    return out_path


def write_window_from_parent(parent_path: str | Path,
                             out_path: str | Path,
                             window_w: int,
                             window_h: int,
                             col_off: int,
                             row_off: int) -> Path:
    """
    Cut a window from an existing parent GeoTIFF (e.g., the scene) to guarantee tiles
    fall inside the scene footprint.
    """
    parent_path = _as_path(parent_path)
    out_path = _as_path(out_path)
    _assert_inputs_exist([parent_path])

    with rasterio.open(parent_path) as src:
        assert 0 <= col_off <= src.width  - window_w, "col_off outside the scene"
        assert 0 <= row_off <= src.height - window_h, "row_off outside the scene"

        win = Window(col_off=col_off, row_off=row_off, width=window_w, height=window_h)
        transform = rasterio.windows.transform(win, src.transform)

        meta = src.meta.copy()
        meta.update(
            driver="GTiff",
            height=window_h,
            width=window_w,
            transform=transform,
            tiled=True,
            blockxsize=512,
            blockysize=512,
            predictor=2,
        )
        meta.pop("compress", None)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(src.read(window=win))  # write all bands

    return out_path


def to_12in16(in_path: str | Path, out_path: str | Path) -> Path:
    """
    Convert a GeoTIFF to “12-in-16”: keep uint16 storage, keep only 12 effective bits.
    Implementation: round to the nearest 16 (drop 4 LSBs with rounding).
    """
    in_path = _as_path(in_path)
    out_path = _as_path(out_path)
    _assert_inputs_exist([in_path])

    with rasterio.open(in_path) as src:
        meta = src.meta.copy()
        meta.update(driver="GTiff", dtype="uint16")
        meta.pop("compress", None)

        # Keep tiling if present; otherwise pick a safe block size
        if getattr(src, "block_shapes", None):
            block_h, block_w = src.block_shapes[0]
        else:
            block_h, block_w = 1024, 1024

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **meta) as dst:
            for row in range(0, src.height, block_h):
                for col in range(0, src.width, block_w):
                    h = min(block_h, src.height - row)
                    w = min(block_w, src.width - col)
                    win = Window(col_off=col, row_off=row, width=w, height=h)
                    arr = src.read(window=win).astype(np.uint16, copy=False)

                    # Round to multiples of 16: (x + 8) >> 4 << 4
                    arr12 = (((arr.astype(np.uint16) + 8) >> 4) << 4).astype(np.uint16, copy=False)
                    dst.write(arr12, window=win)

    return out_path


def ql_rgb(src_path: str | Path) -> None:
    """Build 8-bit RGB quicklook using tools/quicklooks.py with a fixed stretch."""
    src_path = _as_path(src_path)
    ql_path = src_path.with_name(src_path.stem + "_RGB8.tif")
    print(f"  → Quicklook RGB: {ql_path.name}")
    subprocess.run(
        ["python", "tools/quicklooks.py", "--baseline", str(src_path), "--out", str(ql_path)],
        check=True,
    )


def ql_err(a_path: str | Path, b_path: str | Path) -> None:
    """Build 8-bit error map (max over bands) between a_path (baseline) and b_path (reference)."""
    a_path, b_path = _as_path(a_path), _as_path(b_path)
    out = a_path.with_name(a_path.stem + "_ERR8_0_15.tif")
    print(f"  → Quicklook ERR8: {out.name}")
    subprocess.run(
        [
            "python", "tools/quicklooks.py",
            "--baseline", str(a_path),
            "--error-against", str(b_path),
            "--err-max-global", "15",
            "--err-out-base", str(out),
        ],
        check=True,
    )


# --- MAIN ------------------------------------------------------------------

if __name__ == "__main__":
    OUTDIR = Path("data/baseline")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Saving baselines into {OUTDIR.resolve()}")

    # 1) SCENE: 16-bit (as-is), 12-in-16, RGB and ERR (vs 16-bit)
    scene_16 = OUTDIR / "caseA_scene_2k10k_16bit.tif"
    scene_12 = OUTDIR / "caseA_scene_2k10k_12in16.tif"

    print("[1/3] Building SCENE 16-bit from JP2…")
    write_window_stack(scene_16, SCENE_W, SCENE_H)

    print("[1/3] Converting SCENE to 12-in-16…")
    to_12in16(scene_16, scene_12)

    print("[1/3] Scene quicklooks…")
    ql_rgb(scene_12)           # RGB of 12-in-16 baseline (visual reference)
    ql_err(scene_12, scene_16) # ERR8: (12-in-16) vs (raw 16-bit)

    # 2) TILE HC (inside the SCENE) — remove 16-bit after conversion
    tile_HC_16 = OUTDIR / "caseA_tile_HC_1024_16bit.tif"
    tile_HC_12 = OUTDIR / "caseA_tile_HC_1024_12in16.tif"

    print("[2/3] Building HC tile within the SCENE…")
    write_window_from_parent(scene_16, tile_HC_16, TILE_W, TILE_H,
                             col_off=HC_COL_OFF, row_off=HC_ROW_OFF)
    to_12in16(tile_HC_16, tile_HC_12)
    try:
        os.remove(tile_HC_16)
    except FileNotFoundError:
        pass
    ql_rgb(tile_HC_12)

    # 3) TILE LC (inside the SCENE) — remove 16-bit after conversion
    tile_LC_16 = OUTDIR / "caseA_tile_LC_1024_16bit.tif"
    tile_LC_12 = OUTDIR / "caseA_tile_LC_1024_12in16.tif"

    print("[3/3] Building LC tile within the SCENE…")
    write_window_from_parent(scene_16, tile_LC_16, TILE_W, TILE_H,
                             col_off=LC_COL_OFF, row_off=LC_ROW_OFF)
    to_12in16(tile_LC_16, tile_LC_12)
    try:
        os.remove(tile_LC_16)
    except FileNotFoundError:
        pass
    ql_rgb(tile_LC_12)

    print("[DONE] Generated scene (16 + 12 + RGB + ERR) and tiles (12 + RGB). "
          "16-bit tile intermediates removed.")
