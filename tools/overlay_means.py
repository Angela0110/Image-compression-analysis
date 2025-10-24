#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Overlay RD + Pareto + ISO bar plots, with optional interpolation.

What this script does:
- Loads and merges one or more metrics_mean.csv files.
- Normalizes columns used by the plots:
  * RD X: 'bpp' from 'bpp_mean'.
  * RD Y: PSNR or SSIM via --ymetric (psnr_global_rep / ssim_global_rep).
  * Anchors: 'near' (JPEG-LS), 'quality' (J2K), 'bpp_ctrl' (from rate_key=='bpp').
  * Time/Memory helpers: _tenc, _tdec, _mem (if present).
  * Tier normalization: tile_id -> {'LC','HC'} when possible.
- RD overlays per tile with optional piecewise-linear interpolation (--interp).
- Pareto plots: quality vs peak RAM / encode time / decode time.
- ISO-quality bars (target PSNR) and ISO-rate bars (PSNR at fixed CRs).
"""

import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CODEC_LABELS = {
    "ccsds122_ext":   "CCSDS-122",
    "ccsds121_ext":   "CCSDS-121",
    "ccsds123_ext":   "CCSDS-123",
    "j2k":            "JPEG 2000",
    "j2k_gdal":       "JPEG 2000",
    "jpegls":         "JPEG-LS",
    "jpegls_subproc": "JPEG-LS",
}
def _pretty_codec(name: str) -> str:
    s = str(name)
    return CODEC_LABELS.get(s, s)  # safe fallback


# -----------------------------
# I/O helpers
# -----------------------------

def _read_mean_csv(path: Path) -> pd.DataFrame:
    # Try auto-sep + decimal="," first; fallback to decimal="."
    try:
        df = pd.read_csv(path, sep=None, engine="python", decimal=",")
    except Exception:
        df = pd.read_csv(path, sep=None, engine="python", decimal=".")
    df["__source"] = path.as_posix()
    # normalize headers: spaces -> underscores
    df.columns = [re.sub(r"\s+", "_", c.strip()) for c in df.columns]
    return df

def load_and_merge(csv=None, inputs=None, glob_pat=None, dedup=False):
    files = []
    if csv: files.append(Path(csv))
    if inputs: files += [Path(x) for x in inputs]
    if glob_pat: files += list(Path(".").glob(glob_pat))
    files = [f for f in files if f and f.is_file()]
    if not files:
        raise SystemExit("No input CSVs. Pass --csv or --inputs or --glob.")
    dfs = []
    for f in files:
        try:
            dfs.append(_read_mean_csv(f))
            print(f"[OK] loaded {f}")
        except Exception as e:
            print(f"[WARN] skipping {f}: {e}")
    if not dfs:
        raise SystemExit("No valid CSVs loaded.")
    big = pd.concat(dfs, axis=0, ignore_index=True, sort=False)
    if dedup:
        key = ["case","asset","codec","encoder","rate_key","rate_value","tile_id","width","height","bands"]
        have = [k for k in key if k in big.columns]
        if have:
            big = big.sort_values("__source").drop_duplicates(subset=have, keep="last")
    return big


# -----------------------------
# Normalization
# -----------------------------

def _norm_tier(x: object) -> str:
    s = str(x).strip().upper()
    if s in ("LC", "LOW", "L"):  return "LC"
    if s in ("HC", "HIGH", "H"): return "HC"
    return s

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with numeric columns needed for plots and normalized tier labels."""
    d = df.copy()

    # x-axis (bpp from mean)
    if "bpp" not in d.columns and "bpp_mean" in d.columns:
        d["bpp"] = pd.to_numeric(d["bpp_mean"], errors="coerce")

    # y-axis choices
    if "psnr_global_rep" in d.columns:
        d["_psnr"] = pd.to_numeric(d["psnr_global_rep"], errors="coerce")
    if "ssim_global_rep" in d.columns:
        d["_ssim"] = pd.to_numeric(d["ssim_global_rep"], errors="coerce")

    # time / memory (means)
    if "t_comp_s_mean" in d.columns: d["_tenc"] = pd.to_numeric(d["t_comp_s_mean"], errors="coerce")
    if "t_dec_s_mean"  in d.columns: d["_tdec"] = pd.to_numeric(d["t_dec_s_mean"],  errors="coerce")
    if "mem_comp_peak_mb_mean" in d.columns:
        d["_mem"] = pd.to_numeric(d["mem_comp_peak_mb_mean"], errors="coerce")

    # anchors: JPEG-LS near, J2K quality, CCSDS bpp control
    if "nearlossless_eps" in d.columns:
        d["near"] = pd.to_numeric(d["nearlossless_eps"], errors="coerce")
    if "rate_key" in d.columns and "rate_value" in d.columns:
        rk = d["rate_key"].astype(str).str.lower()
        rv = pd.to_numeric(d["rate_value"], errors="coerce")
        d.loc[rk == "quality", "quality"] = rv
        d.loc[rk == "bpp",     "bpp_ctrl"] = rv

    # normalize tier
    if "tile_id" in d.columns:
        d["tile_id"] = d["tile_id"].apply(_norm_tier)

    # ensure numeric
    for c in ["bpp","_psnr","_ssim","quality","near","bpp_ctrl"]:
        if c in d.columns: d[c] = pd.to_numeric(d[c], errors="coerce")

    return d

def _sort_for_codec(dd: pd.DataFrame):
    """Sort points per codec by control param for nicer polylines."""
    if "near" in dd.columns and dd["near"].notna().any():      return dd.sort_values("near")
    if "quality" in dd.columns and dd["quality"].notna().any():return dd.sort_values("quality")
    if "bpp" in dd.columns:                                    return dd.sort_values("bpp")
    return dd


# -----------------------------
# Interpolation helpers
# -----------------------------

def interp_curve_xy(x, y, n=200):
    """Return (xi, yi) interpolating y(x) on a uniform grid within [min(x), max(x)]."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if len(x) < 2:
        return x, y
    idx = np.argsort(x)
    x, y = x[idx], y[idx]
    uniq_x, first_idx = np.unique(x, return_index=True)
    x, y = uniq_x, y[first_idx]
    xi = np.linspace(x.min(), x.max(), int(n))
    yi = np.interp(xi, x, y)
    return xi, yi

def interp_y_at_x(x, y, x_targets):
    """Return y(x_targets) via piecewise-linear interpolation; NaN outside range."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if len(x) < 2:
        return np.full(len(x_targets), np.nan, dtype=float)
    idx = np.argsort(x)
    x, y = x[idx], y[idx]
    uniq_x, first_idx = np.unique(x, return_index=True)
    x, y = uniq_x, y[first_idx]
    out = np.interp(x_targets, x, y)
    out = np.where((x_targets < x.min()) | (x_targets > x.max()), np.nan, out)
    return out

def interp_x_at_y(x, y, y_target):
    """Return x at target y (inverse interpolation). NaN if outside observed range."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if len(x) < 2:
        return np.nan
    idx = np.argsort(y)
    y, x = y[idx], x[idx]
    y_unique, first_idx = np.unique(y, return_index=True)
    y, x = y_unique, x[first_idx]
    if y_target < y.min() or y_target > y.max():
        return np.nan
    return float(np.interp(y_target, y, x))


# -----------------------------
# RD overlay
# -----------------------------

def overlay_rd(df: pd.DataFrame, out_prefix: Path, tiles=("HC","LC"), ymetric="psnr", anchors=None,
               interp=False, interp_points=200):
    ycol, ylab = ("_psnr","PSNR [dB]") if ymetric=="psnr" else ("_ssim","SSIM")

    for tile in tiles:
        dd = df[df["tile_id"] == tile] if "tile_id" in df.columns else df.copy()
        if dd.empty:
            print(f"[WARN] No data for tile {tile}")
            continue

        fig, ax = plt.subplots(figsize=(7.2,4.2))
        for codec, g in dd.groupby("codec"):
            gg = _sort_for_codec(g.copy())
            if "bpp" not in gg.columns:  # nothing to plot on X
                continue
            x = gg["bpp"].to_numpy(dtype=float)
            y = gg[ycol].to_numpy(dtype=float)
            label = str(_pretty_codec(codec))

            if interp and np.isfinite(x).sum() >= 2:
                xi, yi = interp_curve_xy(x, y, n=interp_points)
                ax.plot(xi, yi, "-", linewidth=1.6, label=label)
                ax.plot(x, y, "o", markersize=4, linestyle="None", label="_nolegend_")
            else:
                ax.plot(x, y, "-o", markersize=4, linewidth=1.5, label=label)

            # Anchors
            spec = anchors.get(str(codec)) if anchors else None
            if spec:
                try:
                    key, val = spec.split("="); key = key.strip().lower(); val = float(val)
                    m = None
                    if key in ("near","error") and "near" in gg.columns:
                        m = (gg["near"].astype(float) == val)
                    elif key in ("q","quality") and "quality" in gg.columns:
                        m = (gg["quality"].astype(float) == val)
                    elif key == "bpp":
                        src = gg["bpp_ctrl"].astype(float) if "bpp_ctrl" in gg.columns else gg["bpp"].astype(float)
                        m = np.isclose(src, val, rtol=0, atol=1e-12)
                    if m is not None and m.any():
                        xa = gg.loc[m, "bpp"].astype(float).iloc[0]
                        ya = gg.loc[m, ycol].astype(float).iloc[0]
                        ax.plot([xa], [ya], marker="*", markersize=14, linestyle="None", label="_nolegend_")
                except Exception:
                    pass

        ax.set_xlabel("bpp per band"); ax.set_ylabel(ylab)
        ax.set_title(f"RD overlay – {tile} ({ylab})")
        ax.grid(True, linewidth=0.3); ax.legend(title="Codec")
        out = out_prefix.parent / f"{out_prefix.name}_RD_{tile}_{ylab.replace(' ','_')}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(); fig.savefig(out, dpi=200)
        print(f"[OK] {out}")


# -----------------------------
# Pareto plots
# -----------------------------

def _maybe_anchor(ax, g, ycol, anchors, codec):
    spec = anchors.get(str(codec)) if anchors else None
    if not spec: return
    try:
        key, val = spec.split("="); key = key.strip().lower(); val = float(val)
        m = None
        if key in ("near","error") and "near" in g.columns: m = (g["near"].astype(float) == val)
        elif key in ("q","quality") and "quality" in g.columns: m = (g["quality"].astype(float) == val)
        elif key == "bpp":
            src = g["bpp_ctrl"].astype(float) if "bpp_ctrl" in g.columns else g["bpp"].astype(float)
            m = np.isclose(src, val, rtol=0, atol=1e-12)
        if m is not None and m.any():
            xa = g.iloc[m.values].iloc[0, 0]  # will be overridden by caller (x column)
            ya = g.loc[m, ycol].astype(float).iloc[0]
            return ya, m
    except Exception:
        pass
    return None, None

def pareto_plots(df: pd.DataFrame, out_prefix: Path, tile="HC", ymetric="psnr", anchors=None):
    ycol, ylab = ("_psnr","PSNR [dB]") if ymetric=="psnr" else ("_ssim","SSIM")
    dd = df[df["tile_id"] == tile].copy() if "tile_id" in df.columns else df.copy()
    if dd.empty:
        print(f"[WARN] No data for tile {tile}")
        return

    # Quality vs Peak RAM
    if "_mem" in dd.columns:
        fig, ax = plt.subplots(figsize=(6.6,4.2))
        for codec, g in dd.groupby("codec"):
            ax.plot(g["_mem"], g[ycol], "o", markersize=5, label=_pretty_codec(codec))
            # anchor star on RAM axis
            spec = anchors.get(str(codec)) if anchors else None
            if spec:
                try:
                    key, val = spec.split("="); key = key.strip().lower(); val = float(val)
                    m = None
                    if key in ("near","error") and "near" in g.columns: m = (g["near"].astype(float) == val)
                    elif key in ("q","quality") and "quality" in g.columns: m = (g["quality"].astype(float) == val)
                    elif key == "bpp":
                        src = g["bpp_ctrl"].astype(float) if "bpp_ctrl" in g.columns else g["bpp"].astype(float)
                        m = np.isclose(src, val, rtol=0, atol=1e-12)
                    if m is not None and m.any():
                        xa = g.loc[m, "_mem"].astype(float).iloc[0]
                        ya = g.loc[m, ycol].astype(float).iloc[0]
                        ax.plot([xa], [ya], marker="*", markersize=14, linestyle="None", label="_nolegend_")
                except Exception:
                    pass
        ax.set_xlabel("Peak RAM [MB]"); ax.set_ylabel(ylab)
        ax.set_title(f"Pareto – {tile}: {ylab} vs Peak RAM")
        ax.grid(True, linewidth=0.3); ax.legend(title="Codec")
        out = out_prefix.parent / f"{out_prefix.name}_Pareto_{tile}_{ylab.replace(' ','_')}_vs_RAM.png"
        fig.tight_layout(); fig.savefig(out, dpi=200)
        print(f"[OK] {out}")

    # Quality vs encode time
    if "_tenc" in dd.columns:
        fig, ax = plt.subplots(figsize=(6.6,4.2))
        for codec, g in dd.groupby("codec"):
            ax.plot(g["_tenc"], g[ycol], "o", markersize=5, label=_pretty_codec(codec))
            spec = anchors.get(str(codec)) if anchors else None
            if spec:
                try:
                    key, val = spec.split("="); key = key.strip().lower(); val = float(val)
                    m = None
                    if key in ("near","error") and "near" in g.columns: m = (g["near"].astype(float) == val)
                    elif key in ("q","quality") and "quality" in g.columns: m = (g["quality"].astype(float) == val)
                    elif key == "bpp":
                        src = g["bpp_ctrl"].astype(float) if "bpp_ctrl" in g.columns else g["bpp"].astype(float)
                        m = np.isclose(src, val, rtol=0, atol=1e-12)
                    if m is not None and m.any():
                        xa = g.loc[m, "_tenc"].astype(float).iloc[0]
                        ya = g.loc[m, ycol].astype(float).iloc[0]
                        ax.plot([xa], [ya], marker="*", markersize=14, linestyle="None", label="_nolegend_")
                except Exception:
                    pass
        ax.set_xlabel("Encode time [s]"); ax.set_ylabel(ylab)
        ax.set_title(f"Pareto – {tile}: {ylab} vs Encode time")
        ax.grid(True, linewidth=0.3); ax.legend(title="Codec")
        out = out_prefix.parent / f"{out_prefix.name}_Pareto_{tile}_{ylab.replace(' ','_')}_vs_EncodeTime.png"
        fig.tight_layout(); fig.savefig(out, dpi=200)
        print(f"[OK] {out}")

    # Quality vs decode time
    if "_tdec" in dd.columns:
        fig, ax = plt.subplots(figsize=(6.6,4.2))
        for codec, g in dd.groupby("codec"):
            ax.plot(g["_tdec"], g[ycol], "o", markersize=5, label=_pretty_codec(codec))
            spec = anchors.get(str(codec)) if anchors else None
            if spec:
                try:
                    key, val = spec.split("="); key = key.strip().lower(); val = float(val)
                    m = None
                    if key in ("near","error") and "near" in g.columns: m = (g["near"].astype(float) == val)
                    elif key in ("q","quality") and "quality" in g.columns: m = (g["quality"].astype(float) == val)
                    elif key == "bpp":
                        src = g["bpp_ctrl"].astype(float) if "bpp_ctrl" in g.columns else g["bpp"].astype(float)
                        m = np.isclose(src, val, rtol=0, atol=1e-12)
                    if m is not None and m.any():
                        xa = g.loc[m, "_tdec"].astype(float).iloc[0]
                        ya = g.loc[m, ycol].astype(float).iloc[0]
                        ax.plot([xa], [ya], marker="*", markersize=14, linestyle="None", label="_nolegend_")
                except Exception:
                    pass
        ax.set_xlabel("Decode time [s]"); ax.set_ylabel(ylab)
        ax.set_title(f"Pareto – {tile}: {ylab} vs Decode time")
        ax.grid(True, linewidth=0.3); ax.legend(title="Codec")
        out = out_prefix.parent / f"{out_prefix.name}_Pareto_{tile}_{ylab.replace(' ','_')}_vs_DecodeTime.png"
        fig.tight_layout(); fig.savefig(out, dpi=200)
        print(f"[OK] {out}")


# -----------------------------
# ISO bar plots
# -----------------------------

def ensure_cr_column(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "cr_mean" not in d.columns:
        need = {"in_bytes", "bitstream_bytes_mean"}
        if need.issubset(d.columns):
            num = pd.to_numeric(d["in_bytes"], errors="coerce")
            den = pd.to_numeric(d["bitstream_bytes_mean"], errors="coerce")
            d["cr_mean"] = num / den
        else:
            raise SystemExit("Need 'cr_mean' or (in_bytes & bitstream_bytes_mean) to compute CR.")
    d["cr_mean"] = pd.to_numeric(d["cr_mean"], errors="coerce")
    return d

def plot_iso_rate_psnr_bars(df: pd.DataFrame, out_prefix: Path, tile="HC", cr_list=(2,5,7)):
    """Grouped bars per codec: PSNR at fixed CR values (piecewise-linear interpolation on CR->PSNR)."""
    d = ensure_cr_column(df)
    if "tile_id" in d.columns:
        d = d[d["tile_id"] == tile]
    if d.empty:
        print(f"[WARN] No data for tile {tile}")
        return

    codecs = sorted(map(str, d["codec"].dropna().unique()))
    cr_list = list(cr_list)

    psnr_mat = np.full((len(codecs), len(cr_list)), np.nan, dtype=float)
    for i, codec in enumerate(codecs):
        g = d[d["codec"] == codec]
        cr = pd.to_numeric(g["cr_mean"], errors="coerce").to_numpy(dtype=float)
        ps = pd.to_numeric(g["psnr_global_rep"], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(cr).sum() >= 2 and np.isfinite(ps).sum() >= 2:
            ps_at_cr = interp_y_at_x(cr, ps, np.asarray(cr_list, dtype=float))
            psnr_mat[i, :] = ps_at_cr

    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    x = np.arange(len(codecs))
    width = 0.8 / max(1, len(cr_list))
    for j, crv in enumerate(cr_list):
        offs = x - 0.4 + width/2 + j*width
        vals = psnr_mat[:, j]
        bars = ax.bar(offs, np.nan_to_num(vals, nan=0.0), width, label=f"CR={crv}")
        for (bx, v) in zip(bars, vals):
            if np.isnan(v):
                bx.set_alpha(0.3)
                ax.text(bx.get_x()+bx.get_width()/2, 1.0, "N/A", ha="center", va="bottom", fontsize=8, rotation=90)
            else:
                ax.text(bx.get_x()+bx.get_width()/2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    pretty = [_pretty_codec(c) for c in codecs]
    ax.set_xticks(x)
    ax.set_xticklabels(pretty, ha="center", fontsize=11)
    ax.set_ylabel("PSNR [dB]")
    ax.set_title(f"Iso-rate: PSNR at fixed CR ({', '.join(map(str, cr_list))}) – {tile}")
    ax.legend(title="Fixed CR")
    ax.grid(axis="y", linewidth=0.3)
    ax.margins(x=0.05)
    plt.subplots_adjust(bottom=0.18)

    finite_vals = psnr_mat[np.isfinite(psnr_mat)]
    if finite_vals.size:
        ymin = max(0.0, np.floor(finite_vals.min() - 1))
        ymax = min(100.0, np.ceil(finite_vals.max() + 1))
        if ymin < ymax:
            ax.set_ylim(ymin, ymax)
    else:
        ax.set_ylim(0, 100)

    out = out_prefix.parent / f"{out_prefix.name}_IsoRate_{tile}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=200)
    print(f"[OK] {out}")


# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Overlay RD (+interp) + Pareto + ISO bar plots from metrics_mean.csv")
    # sources
    ap.add_argument("--csv", default=None, help="One metrics_mean.csv")
    ap.add_argument("--inputs", nargs="*", default=None, help="List of metrics_mean.csv to merge")
    ap.add_argument("--glob", default=None, help="Glob pattern, e.g., 'runs/**/metrics_mean.csv'")
    ap.add_argument("--dedup", action="store_true", help="Deduplicate by RD key (keep last)")
    ap.add_argument("--save-merged", default=None, help="Optional: save merged CSV here")

    # filters & output
    ap.add_argument("--case", default=None); ap.add_argument("--asset", default=None)
    ap.add_argument("--tiles", default="HC,LC", help="Comma-separated list (HC,LC)")
    ap.add_argument("--ymetric", choices=["psnr","ssim"], default="psnr")
    ap.add_argument("--out-prefix", default="fig/caseA/overlay_caseA")
    ap.add_argument("--codecs", nargs="*", default=None, help="Subset of codecs to include")
    ap.add_argument("--anchors", default=None,
                    help='JSON dict codec->spec, e.g. {"j2k_gdal":"quality=30","ccsds122_ext":"bpp=5","jpegls_subproc":"near=8"}')

    # interpolation for RD
    ap.add_argument("--interp", action="store_true", help="Draw piecewise-linear interpolated RD curves")
    ap.add_argument("--interp-points", type=int, default=200, help="Number of samples for interpolation")

    # ISO bars
    ap.add_argument("--iso-quality-psnr", type=float, default=65.0, help="Target PSNR for iso-quality bars")
    ap.add_argument("--iso-rate-cr", default="2,5,7", help="Comma-separated CR list for iso-rate bars")

    args = ap.parse_args()

    df = load_and_merge(csv=args.csv, inputs=args.inputs, glob_pat=args.glob, dedup=args.dedup)
    if args.save_merged:
        Path(args.save_merged).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.save_merged, sep=";", index=False, decimal=",")
        print(f"[OK] merged saved -> {args.save_merged}")

    # filters
    if args.case is not None:  df = df[df["case"] == args.case]
    if args.asset is not None: df = df[df["asset"] == args.asset]
    if args.codecs:            df = df[df["codec"].isin(args.codecs)]
    if df.empty: raise SystemExit("No rows after filters.")

    # normalize & plot
    d = normalize_df(df)
    tiles = [t.strip() for t in args.tiles.split(",") if t.strip()]
    anchors = {}
    if args.anchors:
        try:
            anchors = json.loads(args.anchors)
        except Exception as e:
            print(f"[WARN] Could not parse --anchors JSON: {e}. Ignoring.")
            anchors = {}

    out_prefix = Path(args.out_prefix)

    # RD overlay
    overlay_rd(d, out_prefix, tiles=tiles, ymetric=args.ymetric, anchors=anchors,
               interp=args.interp, interp_points=args.interp_points)

    # Pareto
    for t in tiles:
        pareto_plots(d, out_prefix, tile=t, ymetric=args.ymetric, anchors=anchors)

    # ISO bars
    try:
        cr_list = [float(x) for x in str(args.iso_rate_cr).replace(";",",").split(",") if x.strip()]
    except Exception:
        cr_list = [2,5,7]

    for t in tiles:
        plot_iso_rate_psnr_bars(d, out_prefix, tile=t, cr_list=cr_list)


if __name__ == "__main__":
    main()
