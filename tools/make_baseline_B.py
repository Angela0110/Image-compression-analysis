#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Case B — single-run pipeline from RAW hyperspectral tiles to 14‑in‑16 tiles with quicklooks and error maps.

Flow (short):
  1) Read metadata (wavelengths λ, bad bands, QUALITY_TESTFLAGS bit map) from XML.
  2) Subset each RAW tile to N=180 bands (skip bad bands when present).
  3) Build scene mosaics: SPECTRAL (Int16) + QUALITY_TESTFLAGS + PIXELMASK.
  4) Build final binary mask (1=valid, 0=invalid) from QUALITY + PIXELMASK + NoData.
  5) Annotate lambda_nm into band descriptions (if XML wavelengths exist).
  6) Scene quicklooks (RGB and False Color) using nearest-by-λ bands.
  7) k-LSB truncation → 14‑in‑16; scene quicklook from truncated scene.
  8) Scene error map (14‑in‑16 vs 16) in modes: max | mean | rms | p95 | count3.
  9) Crop LC/HC tiles from 14‑in‑16; write tile masks, tile RGB quicklooks, and per‑tile error (max) vs 16.
 10) Optional cleanup of intermediates (subsets + VRT).

Requirements: rasterio, numpy, Pillow, matplotlib, and GDAL CLI tools (gdalbuildvrt, gdal_translate) in PATH.
Example:
  python caseB_pipeline_one_shot.py \
    --input-raw data/raw --output data/baseline --dt DT0000156472 \
    --k 2 --err-mode mean --err-scale fixed --lc 580,5620 --hc 2000,1536
"""
from __future__ import annotations
import os, re, glob, subprocess, sys, argparse, shutil
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from PIL import Image
import matplotlib.pyplot as plt

# ========================= SETUP ========================= #
os.environ.setdefault("PYTHONHASHSEED", "0")
for k_env in [
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "GDAL_NUM_THREADS",
    "OPENJPEG_NUM_THREADS", "OPJ_NUM_THREADS"
]:
    os.environ.setdefault(k_env, "1")

# ========================= FS/CLI HELPERS ========================= #

def _natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]

def list_files(pattern: Path | str):
    return sorted(glob.glob(str(pattern)), key=_natural_key)

def run_cmd(args: list[str]):
    # Prefer EPSG in GeoTIFF SRS (reproducible)
    if args and args[0] in ("gdalbuildvrt", "gdal_translate", "gdalwarp"):
        args = [args[0], "--config", "GTIFF_SRS_SOURCE", "EPSG"] + args[1:]
    print(">>", " ".join(args))
    subprocess.run(args, check=True)

def spectral_tiles(input_dir: Path, dt: str):
    return list_files(input_dir / f"*{dt}*SPECTRAL_IMAGE_COG*.TIF")

def companion(spectral_path: str, old: str, new: str):
    base = os.path.basename(spectral_path).replace(old, new)
    direct = os.path.join(os.path.dirname(spectral_path), base)
    if os.path.exists(direct):
        return direct
    pats = list_files(os.path.join(os.path.dirname(spectral_path), base + "*"))
    return pats[0] if pats else None

# ========================= METADATA ========================= #

def parse_metadata(input_dir: Path, dt: str):
    patt = input_dir / f"*{dt}*METADATA*"
    metas = list_files(patt)
    if not metas:
        return None, None, {}
    try:
        root = ET.parse(metas[0]).getroot()
    except Exception:
        return None, None, {}

    lambdas, badband = [], []
    for band in root.iter():
        tag = band.tag.split('}')[-1].lower()
        if "band" in tag and list(band):
            lam = None; bad = False
            for ch in band:
                k = ch.tag.split('}')[-1].lower()
                v = (ch.text or "").strip()
                if not v:
                    continue
                if "center" in k and "wavelength" in k:
                    try:
                        lam = float(v)
                    except Exception:
                        pass
                if any(s in k for s in ["bad", "invalid", "artifact", "masked", "excluded"]):
                    if v.lower() in ("1", "true", "yes"):
                        bad = True
            if lam is not None:
                lambdas.append(lam); badband.append(bad)

    lambdas = np.array(lambdas, float) if lambdas else None
    badband = np.array(badband, bool) if badband else None

    bit_map = {}
    for el in root.iter():
        tag = el.tag.split('}')[-1].lower()
        if ("flag" in tag or "bit" in tag) and (el.attrib or el.text):
            idx = (el.attrib.get("index") or el.attrib.get("bit") or el.attrib.get("bit_index"))
            meaning = (el.attrib.get("meaning") or el.attrib.get("name") or (el.text or "")).strip()
            if idx is not None and meaning:
                try:
                    bit_map[int(idx)] = meaning.lower()
                except Exception:
                    pass
    return lambdas, badband, bit_map

# ========================= SUBSET to 180 ========================= #

def pick_180(count_common: int, lambdas: np.ndarray | None, badband: np.ndarray | None, target: int):
    all_idx = np.arange(1, count_common + 1)
    keep = np.ones(count_common, bool)
    if badband is not None and badband.size >= count_common:
        keep &= ~badband[:count_common]

    if lambdas is None or lambdas.size < count_common:
        if keep.sum() <= target:
            return all_idx[keep].tolist()
        pos = np.linspace(0, keep.sum() - 1, target).round().astype(int)
        return all_idx[keep][pos].tolist()

    lam_keep = lambdas[:count_common][keep]
    idx_keep = all_idx[keep]
    if lam_keep.size <= target:
        return idx_keep.tolist()

    targets = np.linspace(lam_keep.min(), lam_keep.max(), target)
    used = np.zeros(lam_keep.size, bool)
    sel = []
    for t in targets:
        j = int(np.argmin(np.abs(lam_keep - t)))
        if used[j]:
            left, right = j - 1, j + 1
            best = None
            if left >= 0 and not used[left]:
                best = left
            if right < lam_keep.size and not used[right]:
                if best is None or abs(lam_keep[right] - t) < abs(lam_keep[best] - t):
                    best = right
            if best is not None:
                j = best
        used[j] = True
        sel.append(int(idx_keep[j]))
    sel = sorted(set(sel))
    if len(sel) < target:
        extra = list(idx_keep[~used])[:(target - len(sel))]
        sel = sorted(sel + [int(x) for x in extra])
    return sel

# ========================= λ & QUICKLOOKS ========================= #

def lambdas_from_descriptions(ds):
    descs = getattr(ds, "descriptions", None)
    if not descs:
        return None
    vals = []
    for d in descs:
        if not d:
            vals.append(np.nan); continue
        m = re.search(r"lambda_nm\s*=\s*([0-9.]+)", d)
        vals.append(float(m.group(1)) if m else np.nan)
    arr = np.array(vals, float)
    return arr if np.isfinite(arr).any() else None

def nearest_band(lams, target_nm):
    return int(np.nanargmin(np.abs(lams - target_nm))) + 1

def _wb_whitepatch(R, G, B, valid=None, q=98):
    def qv(x):
        return np.percentile(x[valid], q) if (valid is not None and valid.any()) else np.percentile(x, q)
    rq, gq, bq = qv(R), qv(G), qv(B)
    t = (rq + gq + bq) / 3.0
    R = np.clip(R * (t / (rq + 1e-6)), 0, 1)
    G = np.clip(G * (t / (gq + 1e-6)), 0, 1)
    B = np.clip(B * (t / (bq + 1e-6)), 0, 1)
    return R, G, B

def _wb_grayworld(R, G, B, valid=None):
    if valid is not None and valid.any():
        rmed, gmed, bmed = np.median(R[valid]), np.median(G[valid]), np.median(B[valid])
    else:
        rmed, gmed, bmed = np.median(R), np.median(G), np.median(B)
    m = np.mean([rmed, gmed, bmed]) + 1e-6
    return (np.clip(R * (m/(rmed+1e-6)),0,1), np.clip(G * (m/(gmed+1e-6)),0,1), np.clip(B * (m/(bmed+1e-6)),0,1))

def rgb_joint(ds, bands_1based, valid=None, p=(1,99), gamma=1.0, wb="whitepatch", sample=6):
    bR, bG, bB = bands_1based
    nod = ds.nodata
    def to_float_mask_nodata(x):
        x = x.astype(np.float32)
        if nod is not None and np.isfinite(nod):
            x[x == nod] = np.nan
        return x
    h_s = max(1, ds.height // sample); w_s = max(1, ds.width // sample)
    R_s = to_float_mask_nodata(ds.read(bR, out_shape=(1,h_s,w_s), resampling=Resampling.nearest).squeeze())
    G_s = to_float_mask_nodata(ds.read(bG, out_shape=(1,h_s,w_s), resampling=Resampling.nearest).squeeze())
    B_s = to_float_mask_nodata(ds.read(bB, out_shape=(1,h_s,w_s), resampling=Resampling.nearest).squeeze())
    if valid is not None:
        Hc, Wc = h_s*sample, w_s*sample
        val_s = valid[:Hc,:Wc][::sample,::sample]
        sel = val_s & np.isfinite(R_s) & np.isfinite(G_s) & np.isfinite(B_s)
    else:
        sel = np.isfinite(R_s) & np.isfinite(G_s) & np.isfinite(B_s)
    flat = np.concatenate([R_s[sel],G_s[sel],B_s[sel]]) if np.any(sel) else np.array([])
    lo, hi = (np.percentile(flat, p) if flat.size else (0.0, 1.0))
    R = to_float_mask_nodata(ds.read(bR).squeeze())
    G = to_float_mask_nodata(ds.read(bG).squeeze())
    B = to_float_mask_nodata(ds.read(bB).squeeze())
    if valid is not None:
        H = min(valid.shape[0], R.shape[0]); W = min(valid.shape[1], R.shape[1])
        valid = valid[:H,:W]; R,G,B = R[:H,:W], G[:H,:W], B[:H,:W]
    rng = max(1e-6, (hi - lo))
    st = lambda x: np.clip((x - lo)/rng, 0, 1)
    R,G,B = st(R), st(G), st(B)
    if wb == "whitepatch":
        R,G,B = _wb_whitepatch(R,G,B,valid,q=98)
    elif wb == "gray":
        R,G,B = _wb_grayworld(R,G,B,valid)
    if gamma != 1.0:
        R,G,B = np.power(R,gamma), np.power(G,gamma), np.power(B,gamma)
    R,G,B = np.nan_to_num(R,0.0), np.nan_to_num(G,0.0), np.nan_to_num(B,0.0)
    return np.dstack([R,G,B])

def save_png(img, path: Path, valid=None, overlay=False, title=""):
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    if overlay and valid is not None:
        inv = ~valid
        ov = np.zeros((*inv.shape,4), float)
        ov[inv,0] = 1.0; ov[inv,3] = 0.25
        plt.imshow(ov)
    plt.axis("off"); plt.title(title); plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200); plt.close()
    print("🖼️", path)

# ========================= TILES ========================= #

def write_tile_from_scene(scene_path: Path, out_path: Path, col_off: int, row_off: int, size: int):
    with rasterio.open(scene_path) as src:
        assert 0 <= col_off <= src.width - size, "col_off out of bounds"
        assert 0 <= row_off <= src.height - size, "row_off out of bounds"
        win = Window(col_off=col_off, row_off=row_off, width=size, height=size)
        tr = rasterio.windows.transform(win, src.transform)
        meta = src.meta.copy()
        meta.update(height=size, width=size, transform=tr, tiled=True, blockxsize=512, blockysize=512, BIGTIFF="YES")
        with rasterio.open(out_path, "w", **meta) as dst:
            data = src.read(window=win)
            dst.write(data)
            if src.descriptions:
                for i in range(1, src.count + 1):
                    if src.descriptions[i - 1]:
                        dst.set_band_description(i, src.descriptions[i - 1])

def write_mask_tile_from_scene(mask_scene_path: Path, out_path: Path, col_off: int, row_off: int, size: int):
    with rasterio.open(mask_scene_path) as src:
        win = Window(col_off=col_off, row_off=row_off, width=size, height=size)
        tr = rasterio.windows.transform(win, src.transform)
        meta = src.meta.copy()
        meta.update(count=1, dtype="uint8", nodata=0, height=size, width=size, transform=tr,
                    tiled=True, blockxsize=512, blockysize=512, BIGTIFF="YES", compress="NONE")
        data = src.read(1, window=win)
        data = (data != 0).astype("uint8")
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(data, 1)

# ========================= TRUNCATION & ERROR ========================= #

def trunc_uint16(u16, k):
    if k <= 0:
        return u16
    return ((u16 >> k) << k).astype(np.uint16, copy=False)

def write_truncated_copy(input_path: Path, output_path: Path, k: int, tile: int = 512):
    with rasterio.open(input_path) as src:
        H, W = src.height, src.width
        nd_in = src.nodatavals[0] if src.nodatavals else None
        dtype_out = src.dtypes[0]
        profile = src.profile.copy()
        profile.update(dtype=dtype_out, count=src.count, tiled=True, blockxsize=tile, blockysize=tile,
                       compress=None, predictor=2, BIGTIFF="YES", nodata=nd_in)
        with rasterio.open(output_path, "w", **profile) as dst:
            for b in range(src.count):
                for r0 in range(0, H, tile):
                    r1 = min(H, r0 + tile)
                    for c0 in range(0, W, tile):
                        c1 = min(W, c0 + tile)
                        win = Window(c0, r0, c1 - c0, r1 - r0)
                        ref = src.read(b + 1, window=win)
                        if ref.dtype == np.int16:
                            u = ref.view(np.uint16)
                        elif ref.dtype == np.uint16:
                            u = ref
                        else:
                            u = ref.astype(np.uint16, copy=False)
                        ut_u16 = trunc_uint16(u, k)
                        if ref.dtype == np.int16:
                            out = ut_u16.view(np.int16)
                        else:
                            out = ut_u16.astype(ref.dtype, copy=False)
                        if nd_in is not None:
                            out[ref == nd_in] = nd_in
                        dst.write(out, b + 1, window=win)
    print(f"✅ wrote 14-in-16: {output_path} (k={k})")

def read_mask(mask_path: Path | None):
    if not mask_path or not Path(mask_path).exists():
        return None
    with rasterio.open(mask_path) as m:
        return (m.read(1) > 0)

def make_scene_error_map(ref16_path: Path, scene14_path: Path, mask_path: Path | None,
                         err_scale: str, k_bits: int, out_png: Path, err_mode: str = "mean"):
    with rasterio.open(ref16_path) as ref, rasterio.open(scene14_path) as cmp:
        assert (ref.width, ref.height, ref.count) == (cmp.width, cmp.height, cmp.count), \
            "ref16 and 14-in-16 must match in size and band count"
        valid = read_mask(mask_path)
        H,W = ref.height, ref.width
        tile = 512
        global_max = 0
        kmax = (1 << k_bits) - 1
        nbins = kmax + 1

        def pass_once(mode):
            nonlocal global_max
            for r0 in range(0,H,tile):
                r1=min(H,r0+tile); h=r1-r0
                if mode in ("mean","rms","count3"):
                    acc = np.zeros((h, W), dtype=np.float32)
                    cnt3 = np.zeros((h, W), dtype=np.uint16) if mode=="count3" else None
                    ssq = np.zeros((h, W), dtype=np.float32) if mode=="rms" else None
                elif mode == "max":
                    accmax = np.zeros((h, W), dtype=np.uint16)
                elif mode == "p95":
                    hist = np.zeros((h, W, nbins), dtype=np.uint32)
                for b in range(1, ref.count+1):
                    a = ref.read(b, window=Window(0,r0,W,h)).astype(np.int32)
                    c = cmp.read(b, window=Window(0,r0,W,h)).astype(np.int32)
                    d = np.abs(a - c)
                    if valid is not None: d[~valid[r0:r1, 0:W]] = 0
                    if mode == "mean":   acc += d
                    elif mode == "rms":  ssq += (d*d)
                    elif mode == "count3": cnt3 += (d == kmax)
                    elif mode == "max":  accmax = np.maximum(accmax, d.astype(np.uint16))
                    elif mode == "p95":
                        d_clip = np.clip(d, 0, kmax)
                        for k in range(nbins):
                            hist[..., k] += (d_clip == k)
                if mode == "mean":   out_tile = acc / ref.count
                elif mode == "rms":  out_tile = np.sqrt(ssq / ref.count)
                elif mode == "count3": out_tile = cnt3.astype(np.float32)
                elif mode == "max":  out_tile = accmax.astype(np.float32)
                elif mode == "p95":
                    cdf = np.cumsum(hist, axis=2)
                    thr = (cdf[..., -1] * 0.95).astype(np.uint32)
                    out_tile = np.zeros((h, W), dtype=np.float32)
                    for k in range(nbins):
                        mask = (cdf[..., k] >= thr) & (out_tile == 0)
                        out_tile[mask] = k
                global_max = max(global_max, float(out_tile.max()))
            return global_max

        pass_once(err_mode)
        if err_mode == "count3":
            emax = max(1, ref.count) if err_scale=="fixed" else max(1, int(global_max))
        else:
            emax = kmax if err_scale=="fixed" else max(1, int(np.ceil(global_max)))

        im = Image.new("L", (W, H))
        for r0 in range(0,H,tile):
            r1=min(H,r0+tile); h=r1-r0
            if err_mode in ("mean","rms","count3"):
                acc = np.zeros((h, W), dtype=np.float32)
                cnt3 = np.zeros((h, W), dtype=np.uint16) if err_mode=="count3" else None
                ssq = np.zeros((h, W), dtype=np.float32) if err_mode=="rms" else None
            elif err_mode == "max":
                accmax = np.zeros((h, W), dtype=np.uint16)
            elif err_mode == "p95":
                hist = np.zeros((h, W, nbins), dtype=np.uint32)
            for b in range(1, ref.count+1):
                a = ref.read(b, window=Window(0,r0,W,h)).astype(np.int32)
                c = cmp.read(b, window=Window(0,r0,W,h)).astype(np.int32)
                d = np.abs(a - c)
                if valid is not None: d[~valid[r0:r1, 0:W]] = 0
                if err_mode == "mean": acc += d
                elif err_mode == "rms": ssq += (d*d)
                elif err_mode == "count3": cnt3 += (d == kmax)
                elif err_mode == "max": accmax = np.maximum(accmax, d.astype(np.uint16))
                elif err_mode == "p95":
                    d_clip = np.clip(d, 0, kmax)
                    for k in range(nbins):
                        hist[..., k] += (d_clip == k)
            if err_mode == "mean": out_tile = acc / ref.count
            elif err_mode == "rms": out_tile = np.sqrt(ssq / ref.count)
            elif err_mode == "count3": out_tile = cnt3.astype(np.float32)
            elif err_mode == "max": out_tile = accmax.astype(np.float32)
            elif err_mode == "p95":
                cdf = np.cumsum(hist, axis=2)
                thr = (cdf[..., -1] * 0.95).astype(np.uint32)
                out_tile = np.zeros((h, W), dtype=np.float32)
                for k in range(nbins):
                    mask = (cdf[..., k] >= thr) & (out_tile == 0)
                    out_tile[mask] = k
            tile_u8 = (np.clip(out_tile, 0, emax) * (255.0/emax) + 0.5).astype(np.uint8)
            im.paste(Image.fromarray(tile_u8, mode="L"), (0, r0))
        im.save(out_png)
    print(f"🧮 SCENE error ({err_mode}) scale=0..{emax} DN → {out_png}")

# ========================= PIPELINE ========================= #

def main():
    ap = argparse.ArgumentParser(description="Case B: RAW → 14-in-16 tiles with quicklooks and error maps")
    ap.add_argument("--input-raw", required=True, help="Folder with RAW tiles")
    ap.add_argument("--output", required=True, help="Output folder")
    ap.add_argument("--dt", required=True, help="Datatake ID filter for filenames")
    ap.add_argument("--target-bands", type=int, default=180)
    ap.add_argument("--tile-size", type=int, default=512)
    ap.add_argument("--lc", default="580,5620", help="LC tile offset col,row (e.g., 580,5620)")
    ap.add_argument("--hc", default="2000,1536", help="HC tile offset col,row (e.g., 2000,1536)")
    ap.add_argument("--keep-subsets", action="store_true", help="Keep subsets_180b and VRT intermediates")

    # Quicklook params
    ap.add_argument("--stretch", default="1,99", help="Percentile stretch low,high")
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--wb", default="whitepatch", choices=["none","whitepatch","gray"])
    ap.add_argument("--rgb-nm", default="665.0,560.0,490.0")
    ap.add_argument("--false-nm", default="842.0,665.0,560.0")

    # Truncation + error
    ap.add_argument("--k", type=int, default=2, help="k LSBs to zero (14‑in‑16 → 2)")
    ap.add_argument("--err-mode", default="mean", choices=["max","mean","rms","p95","count3"],
                    help="Aggregation over bands for SCENE error (14‑in‑16 vs 16)")
    ap.add_argument("--err-scale", default="fixed", choices=["fixed","auto"], help="Error map scale for PNG")

    args = ap.parse_args()

    input_dir = Path(args.input_raw).resolve()
    out_dir = Path(args.output).resolve(); out_dir.mkdir(parents=True, exist_ok=True)
    dt = args.dt
    TILE_SIZE = args.tile_size
    LC_COL_OFF, LC_ROW_OFF = [int(x) for x in args.lc.split(",")]
    HC_COL_OFF, HC_ROW_OFF = [int(x) for x in args.hc.split(",")]
    P_LOW, P_HIGH = [float(v) for v in args.stretch.split(",")]
    RGB_TARGETS_NM = tuple(float(v) for v in args.rgb_nm.split(","))
    FALSE_TARGETS_NM = tuple(float(v) for v in args.false_nm.split(","))

    # 1) Subset → scene mosaic (180 bands)
    tiles = spectral_tiles(input_dir, dt)
    if not tiles:
        print(f"No *{dt}*SPECTRAL_IMAGE_COG*.TIF found in {input_dir}")
        sys.exit(1)

    counts = [rasterio.open(p).count for p in tiles]
    min_count = min(counts)

    lambdas, badband, bit_map = parse_metadata(input_dir, dt)
    idx_list = pick_180(min_count, lambdas, badband, args.target_bands)

    subset_dir = out_dir / "subsets_180b"; subset_dir.mkdir(exist_ok=True, parents=True)
    subset_files = []
    for sp in tiles:
        outp = subset_dir / os.path.basename(sp).replace("SPECTRAL_IMAGE_COG", "SUBSET_180B")
        subset_files.append(outp.as_posix())
        if outp.exists():
            print("OK (exists):", outp.name); continue
        with rasterio.open(sp) as src:
            prof = src.profile.copy()
            prof.update(count=len(idx_list), tiled=True, blockxsize=512, blockysize=512, compress='NONE', BIGTIFF='YES')
            with rasterio.open(outp, 'w', **prof) as dst:
                for i, b in enumerate(idx_list, 1):
                    dst.write(src.read(b), i)

    vrt = out_dir / f"{dt}_scene_180b.vrt"
    scene16 = out_dir / f"{dt}_scene_180b_int16.tif"
    run_cmd(['gdalbuildvrt', '-resolution', 'highest', vrt.as_posix()] + subset_files)
    run_cmd(['gdal_translate', vrt.as_posix(), scene16.as_posix(), '-co','BIGTIFF=YES','-co','TILED=YES','-co','BLOCKXSIZE=512','-co','BLOCKYSIZE=512','-co','COMPRESS=NONE'])
    print("✅ scene16:", scene16)

    # 2) QUALITY/PIXELMASK → final mask
    flags_tiles = [companion(p, "-SPECTRAL_IMAGE_COG", "-QL_QUALITY_TESTFLAGS_COG") for p in tiles]
    pixm_tiles  = [companion(p, "-SPECTRAL_IMAGE_COG", "-QL_PIXELMASK_COG") for p in tiles]
    flags_tiles = [p for p in flags_tiles if p and os.path.exists(p)]
    pixm_tiles  = [p for p in pixm_tiles  if p and os.path.exists(p)]

    flags_scene = None; pixm_scene = None
    if flags_tiles:
        vflags = out_dir / f"{dt}_scene_qualityflags.vrt"
        flags_scene = out_dir / f"{dt}_scene_qualityflags.tif"
        run_cmd(['gdalbuildvrt', '-resolution', 'highest', vflags.as_posix()] + flags_tiles)
        run_cmd(['gdal_translate', vflags.as_posix(), flags_scene.as_posix(), '-co','BIGTIFF=YES','-co','TILED=YES','-co','BLOCKXSIZE=512','-co','BLOCKYSIZE=512','-co','COMPRESS=NONE'])

    if pixm_tiles:
        vpix = out_dir / f"{dt}_scene_pixelmask.vrt"
        pixm_scene = out_dir / f"{dt}_scene_pixelmask.tif"
        run_cmd(['gdalbuildvrt', '-resolution', 'highest', vpix.as_posix()] + pixm_tiles)
        run_cmd(['gdal_translate', vpix.as_posix(), pixm_scene.as_posix(), '-co','BIGTIFF=YES','-co','TILED=YES','-co','BLOCKXSIZE=512','-co','BLOCKYSIZE=512','-co','COMPRESS=NONE'])

    with rasterio.open(scene16) as ref:
        h, w = ref.height, ref.width
        nodata = ref.nodata
        first = ref.read(1)
        nodamask = (first == nodata) if nodata is not None else np.zeros((h, w), bool)

    invalid = np.zeros((h, w), bool)

    def find_bit(substrs, bit_map):
        for b, name in bit_map.items():
            if all(ss in name for ss in substrs):
                return b
        return None

    used_bits = {}
    if flags_scene and bit_map:
        with rasterio.open(flags_scene) as f:
            fl = f.read(1).astype(np.uint32)
            b_cloud  = find_bit(['cloud'],  bit_map)
            b_shadow = find_bit(['shadow'], bit_map)
            b_cirrus = find_bit(['cirrus'], bit_map)
            b_defect = find_bit(['defect'], bit_map)
            if b_cloud  is not None: invalid |= (fl & (1 << b_cloud )) != 0; used_bits['cloud']  = b_cloud
            if b_shadow is not None: invalid |= (fl & (1 << b_shadow)) != 0; used_bits['shadow'] = b_shadow
            if b_cirrus is not None: invalid |= (fl & (1 << b_cirrus)) != 0; used_bits['cirrus'] = b_cirrus
            if b_defect is not None: invalid |= (fl & (1 << b_defect)) != 0; used_bits['defect'] = b_defect

    if pixm_scene:
        with rasterio.open(pixm_scene) as pm:
            pmv = pm.read(1)
            invalid |= (pmv != 0)

    invalid |= nodamask
    valid = (~invalid).astype(np.uint8)

    mask_final = out_dir / f"{dt}_scene_mask_uint8.tif"
    with rasterio.open(scene16) as ref:
        prof = ref.profile.copy()
        prof.update(driver='GTiff', count=1, dtype='uint8', nodata=0, compress='NONE', BIGTIFF='YES', tiled=True, blockxsize=512, blockysize=512)
        for k in ('nbits', 'scales', 'offsets'):
            prof.pop(k, None)
        with rasterio.open(mask_final, 'w', **prof) as dst:
            dst.write((valid > 0).astype(np.uint8), 1)
    print("✅ mask:", mask_final, "| used bits:", used_bits if used_bits else "{}")

    # 3) Annotate λ
    if lambdas is not None:
        with rasterio.open(scene16, "r+") as ds:
            for i, src_idx in enumerate(idx_list, 1):
                if src_idx-1 < len(lambdas):
                    ds.set_band_description(i, f"lambda_nm={lambdas[src_idx - 1]:.2f}")
        print("✅ band descriptions annotated with lambda_nm")

    # 4) Scene quicklooks (RGB/False Color)
    with rasterio.open(scene16) as ds:
        lams = lambdas_from_descriptions(ds)
        if (lams is None) and (lambdas is not None) and (len(lambdas) >= max(idx_list)):
            lams = lambdas[np.array(idx_list) - 1]
        if lams is None or not np.isfinite(lams).any():
            raise RuntimeError("Could not retrieve λ per band (no descriptions nor XML).")
        nb = lambda nm: int(np.nanargmin(np.abs(lams - nm))) + 1
        bands_rgb   = (nb(RGB_TARGETS_NM[0]), nb(RGB_TARGETS_NM[1]), nb(RGB_TARGETS_NM[2]))
        bands_false = (nb(FALSE_TARGETS_NM[0]), nb(FALSE_TARGETS_NM[1]), nb(FALSE_TARGETS_NM[2]))
        with rasterio.open(mask_final) as m:
            validm = (m.read(1) > 0)
        RGB   = rgb_joint(ds, bands_rgb,   valid=validm, p=(P_LOW,P_HIGH), gamma=args.gamma, wb=args.wb)
        FALSE = rgb_joint(ds, bands_false, valid=validm, p=(P_LOW,P_HIGH), gamma=args.gamma, wb=args.wb)
        save_png(RGB,   out_dir / f"{dt}_quicklook_rgb.png",         valid=validm, overlay=False, title="RGB (λ)")
        save_png(RGB,   out_dir / f"{dt}_quicklook_rgb_overlay.png", valid=validm, overlay=True,  title="RGB (λ)")
        save_png(FALSE, out_dir / f"{dt}_quicklook_false.png",       valid=validm, overlay=False, title="False Color (λ)")

    # 5) Truncation → 14‑in‑16
    scene14 = out_dir / f"{dt}_scene_180b_14in16.tif"
    write_truncated_copy(scene16, scene14, k=args.k, tile=512)

    # 6) Scene quicklook (14‑in‑16) + SCENE error vs 16
    with rasterio.open(scene14) as ds14:
        lams = lambdas_from_descriptions(ds14) or lambdas
        bands = (nearest_band(lams, RGB_TARGETS_NM[0]), nearest_band(lams, RGB_TARGETS_NM[1]), nearest_band(lams, RGB_TARGETS_NM[2]))
        validm = read_mask(mask_final)
        img14 = rgb_joint(ds14, bands, valid=validm, p=(P_LOW,P_HIGH), gamma=args.gamma, wb=args.wb)
        save_png(img14, scene14.with_suffix('.scene_RGB8.png'), valid=validm, overlay=False, title="Scene 14‑in‑16 RGB (λ)")
    make_scene_error_map(scene16, scene14, mask_final, args.err_scale, args.k, scene14.with_suffix(f".scene_ERR_{args.err_mode}.png"), err_mode=args.err_mode)

    # 7) Tiles from 14‑in‑16 + masks + quicklooks + per‑tile error (max)
    tile_LC_14 = out_dir / f"{dt}_tile_LC_{TILE_SIZE}_14in16bit.tif"
    tile_HC_14 = out_dir / f"{dt}_tile_HC_{TILE_SIZE}_14in16bit.tif"
    write_tile_from_scene(scene14, tile_LC_14, LC_COL_OFF, LC_ROW_OFF, TILE_SIZE)
    write_tile_from_scene(scene14, tile_HC_14, HC_COL_OFF, HC_ROW_OFF, TILE_SIZE)
    print(f"✅ tiles (14‑in‑16): LC=({LC_COL_OFF},{LC_ROW_OFF}) HC=({HC_COL_OFF},{HC_ROW_OFF})")

    tile_LC_mask = out_dir / f"{dt}_tile_LC_{TILE_SIZE}_mask.tif"
    tile_HC_mask = out_dir / f"{dt}_tile_HC_{TILE_SIZE}_mask.tif"
    write_mask_tile_from_scene(mask_final, tile_LC_mask, LC_COL_OFF, LC_ROW_OFF, TILE_SIZE)
    write_mask_tile_from_scene(mask_final, tile_HC_mask, HC_COL_OFF, HC_ROW_OFF, TILE_SIZE)

    for tpath, (cx, ry) in [(tile_LC_14, (LC_COL_OFF, LC_ROW_OFF)), (tile_HC_14, (HC_COL_OFF, HC_ROW_OFF))]:
        with rasterio.open(tpath) as ds:
            lams_tile = lambdas_from_descriptions(ds) or lambdas
            bands_tile = (nearest_band(lams_tile, RGB_TARGETS_NM[0]), nearest_band(lams_tile, RGB_TARGETS_NM[1]), nearest_band(lams_tile, RGB_TARGETS_NM[2]))
            with rasterio.open(mask_final) as m:
                win = Window(cx, ry, TILE_SIZE, TILE_SIZE)
                valid_tile = (m.read(1, window=win) > 0)
            imgT = rgb_joint(ds, bands_tile, valid=valid_tile, p=(P_LOW,P_HIGH), gamma=args.gamma, wb=args.wb)
        save_png(imgT, tpath.with_suffix('.RGB8.png'), valid=valid_tile, overlay=False, title="Tile RGB (λ)")

        out_err = tpath.with_suffix('.ERRmax_vs16.png')
        with rasterio.open(scene16) as R, rasterio.open(scene14) as C:
            tile_max = np.zeros((TILE_SIZE,TILE_SIZE), dtype=np.uint16)
            for b in range(1, R.count+1):
                a = R.read(b, window=Window(cx,ry,TILE_SIZE,TILE_SIZE)).astype(np.int32)
                c = C.read(b, window=Window(cx,ry,TILE_SIZE,TILE_SIZE)).astype(np.int32)
                diff = np.abs(a - c)
                diff[~valid_tile] = 0
                tile_max = np.maximum(tile_max, diff.astype(np.uint16))
            emax = (1<<args.k)-1 if args.err_scale=="fixed" else max(1, int(tile_max.max()))
            tile_u8 = (np.clip(tile_max,0,emax) * (255.0/emax) + 0.5).astype(np.uint8)
            Image.fromarray(tile_u8, mode="L").save(out_err)
            print(f"🧮 TILE error scale=0..{emax} DN → {out_err}")

    # 8) Cleanup
    if not args.keep_subsets:
        try:
            for p in subset_files:
                Path(p).unlink(missing_ok=True)
            shutil.rmtree(subset_dir, ignore_errors=True)
            Path(vrt).unlink(missing_ok=True)
        except Exception as e:
            print(f"⚠️ Could not remove intermediates: {e}")

    print("\n[DONE] Case B pipeline complete:", out_dir)

if __name__ == "__main__":
    main()
