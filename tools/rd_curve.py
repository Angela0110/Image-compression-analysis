#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RD curves using only:
  X = bpp_mean  (mirrored to 'bpp' for plotting)
  Y = psnr_global_rep or ssim_global_rep (choose with --ymetric)

Extras:
- Optional filters: --case, --asset, --codec
- Tile-aware: --tile {HC|LC}; if omitted, draws combined HC vs LC and per-tile figures
- Anchors: --anchor-q (J2K QUALITY), --anchor-bpp (CCSDS target bpp from rate_key=bpp),
           --anchor-error (JPEG-LS NEAR). Anchors are star markers without legend entries.
- Optional piecewise-linear interpolation via --interp / --interp-points
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------- utilities -----------------------------

def _read_csv_smart(path: str) -> pd.DataFrame:
    """Try to read with auto-sep and decimal=','; fall back to decimal='.'."""
    try:
        return pd.read_csv(path, sep=None, engine="python", decimal=",")
    except Exception:
        return pd.read_csv(path, sep=None, engine="python", decimal=".")

def _norm_tile(s: object) -> str:
    """Normalize tile labels to {'HC','LC'} when possible."""
    t = str(s).strip().upper()
    if t in ("HC", "HIGH", "H"): return "HC"
    if t in ("LC", "LOW", "L"):  return "LC"
    return t

def _prep_df(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only what's needed for plotting; derive helpers for anchors and ordering."""
    d = df.copy()

    need = ["bpp_mean", "psnr_global_rep", "ssim_global_rep"]
    missing = [c for c in need if c not in d.columns]
    if missing:
        raise SystemExit("Missing required column(s): " + ", ".join(missing))

    for c in need:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d["bpp"] = pd.to_numeric(d["bpp_mean"], errors="coerce")

    if "rate_key" in d.columns and "rate_value" in d.columns:
        rk = d["rate_key"].astype(str).str.lower()
        rv = pd.to_numeric(d["rate_value"], errors="coerce")
        d.loc[rk == "quality", "quality"] = rv               # J2K
        d.loc[rk.isin(["nearlossless_eps","near","error","eps"]), "near"] = rv  # JPEG-LS
        d.loc[rk == "bpp", "bpp_ctrl"] = rv                  # CCSDS (control bpp)

    for c in ["quality","near","bpp_ctrl"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    if "tile_id" in d.columns:
        d["tile_id"] = d["tile_id"].apply(_norm_tile)

    return d

def _pick_y_column(df: pd.DataFrame, ymetric: str):
    if ymetric == "psnr":
        if "psnr_global_rep" in df.columns:
            return "psnr_global_rep", "PSNR [dB]", "PSNR"
        raise SystemExit("Missing 'psnr_global_rep'.")
    if ymetric == "ssim":
        if "ssim_global_rep" in df.columns:
            return "ssim_global_rep", "SSIM", "SSIM"
        raise SystemExit("Missing 'ssim_global_rep'.")
    raise ValueError("ymetric must be 'psnr' or 'ssim'.")

def _plot_curve(ax, x, y, label, interp=False, num_points=200):
    """Either connect measured points or draw a piecewise-linear interpolation."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if len(x) == 0:
        return
    idx = np.argsort(x)
    x, y = x[idx], y[idx]
    uniq_x, first_idx = np.unique(x, return_index=True)
    x, y = uniq_x, y[first_idx]

    if interp and len(x) >= 2:
        xi = np.linspace(x.min(), x.max(), int(num_points))
        yi = np.interp(xi, x, y)
        ax.plot(xi, yi, "-", linewidth=1.5, label=label)
        ax.plot(x, y, "o", markersize=4, linestyle="None", label="_nolegend_")
    else:
        ax.plot(x, y, "-o", markersize=4, linewidth=1.5, label=label)
    ax.grid(True, linewidth=0.3)

def _mark_anchor_exact(ax, x, y, anchor_value):
    if anchor_value is None: return False
    m = np.isclose(x, float(anchor_value), rtol=0, atol=1e-12)
    if m.any():
        ax.plot([x[m][0]], [y[m][0]], marker="*", markersize=14, linestyle="None", label="_nolegend_")
        return True
    return False

def _mark_anchor_by_array(ax, x, y, anchor_value, arr):
    if anchor_value is None or arr is None: return False
    arr = np.asarray(arr, dtype=float)
    m = np.isclose(arr, float(anchor_value), rtol=0, atol=1e-12)
    if m.any():
        ax.plot([x[m][0]], [y[m][0]], marker="*", markersize=14, linestyle="None", label="_nolegend_")
        return True
    return False


# ----------------------------- plotting -----------------------------

def _order_for_plot(dd: pd.DataFrame) -> pd.DataFrame:
    if "near" in dd.columns and dd["near"].notna().any():      return dd.sort_values("near")
    if "quality" in dd.columns and dd["quality"].notna().any():return dd.sort_values("quality")
    return dd.sort_values("bpp")

def plot_rd_single(d: pd.DataFrame, tile: str, anchor_q, anchor_bpp, out_prefix: str,
                   ymetric="psnr", codec_filter=None, anchor_near=None,
                   interp=False, interp_points=200):
    dd = d[d.get("tile_id") == tile].copy() if "tile_id" in d.columns else d.copy()
    if codec_filter is not None and "codec" in dd.columns:
        dd = dd[dd["codec"] == codec_filter]
    if dd.empty:
        raise SystemExit(f"No data for tile_id={tile}" + (f" and codec={codec_filter}" if codec_filter else ""))

    dd = _order_for_plot(dd)
    if "bpp" not in dd.columns:
        raise SystemExit("Missing 'bpp' (mirrored from 'bpp_mean').")

    ycol, ylabel, suf = _pick_y_column(dd, ymetric)
    x = dd["bpp"].to_numpy(dtype=float)
    y = dd[ycol].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(6, 4))
    _plot_curve(ax, x, y, label=f"{tile}", interp=interp, num_points=interp_points)

    if "quality" in dd.columns and dd["quality"].notna().any():
        qvals = dd["quality"].to_numpy(dtype=float)
        for xi, yi, qi in zip(x, y, qvals):
            if not np.isnan(qi):
                ax.annotate(str(int(qi)), (xi, yi), xytext=(3, 3), textcoords="offset points", fontsize=8)
        if anchor_q is not None:
            m = (qvals == float(anchor_q))
            if m.any():
                ax.plot([x[m][0]], [y[m][0]], marker="*", markersize=14, linestyle="None", label="_nolegend_")

    if "near" in dd.columns and dd["near"].notna().any():
        nvals = dd["near"].to_numpy(dtype=float)
        for xi, yi, ni in zip(x, y, nvals):
            if not np.isnan(ni):
                ax.annotate(str(int(ni)), (xi, yi), xytext=(3, 3), textcoords="offset points", fontsize=8)
        if anchor_near is not None:
            m = (nvals == float(anchor_near))
            if m.any():
                ax.plot([x[m][0]], [y[m][0]], marker="*", markersize=14, linestyle="None", label="_nolegend_")

    ctrl = dd["bpp_ctrl"].to_numpy(dtype=float) if "bpp_ctrl" in dd.columns else None
    if ctrl is not None:
        _mark_anchor_by_array(ax, x, y, anchor_bpp, ctrl)
    else:
        _mark_anchor_exact(ax, x, y, anchor_bpp)

    ax.set_xlabel("bpp per band")
    ax.set_ylabel(ylabel)
    ax.set_title(f"RD – {tile}")
    ax.legend()

    out = Path(f"{out_prefix}_RD_{tile}_{suf}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=200)
    print(f"[OK] Figure saved: {out}")

def plot_rd_both(d: pd.DataFrame, anchor_q, anchor_bpp, out_prefix: str,
                 ymetric="psnr", codec_filter=None, anchor_near=None,
                 interp=False, interp_points=200):
    if "tile_id" in d.columns:
        tiles_present = [t for t in ["HC","LC"] if t in d["tile_id"].dropna().unique()]
    else:
        tiles_present = ["ALL"]
    if not tiles_present:
        print("[WARN] No HC/LC tiles found for the combined figure.")
        return

    y_label, suf = None, None
    for tile in tiles_present:
        dd0 = d[(d.get("tile_id") == tile) if "tile_id" in d.columns else (d.index == d.index)]
        if codec_filter is not None and "codec" in d.columns:
            dd0 = dd0[dd0["codec"] == codec_filter]
        if dd0.empty or "bpp" not in dd0.columns:
            continue
        _, y_label, suf = _pick_y_column(dd0, ymetric)
        break
    if y_label is None:
        raise SystemExit("Could not determine Y metric for the combined figure.")

    fig, ax = plt.subplots(figsize=(6, 4))
    for tile in tiles_present:
        dd = d[(d.get("tile_id") == tile) if "tile_id" in d.columns else (d.index == d.index)].copy()
        if codec_filter is not None and "codec" in dd.columns:
            dd = dd[dd["codec"] == codec_filter]
        if dd.empty or "bpp" not in dd.columns:
            continue

        dd = _order_for_plot(dd)
        ycol, _, _ = _pick_y_column(dd, ymetric)
        x = dd["bpp"].to_numpy(dtype=float)
        y = dd[ycol].to_numpy(dtype=float)

        _plot_curve(ax, x, y, label=f"{tile}", interp=interp, num_points=interp_points)

        if anchor_q is not None and "quality" in dd.columns and dd["quality"].notna().any():
            qvals = dd["quality"].to_numpy(dtype=float)
            mq = (qvals == float(anchor_q))
            if mq.any():
                ax.plot([x[mq][0]], [y[mq][0]], marker="*", markersize=14, linestyle="None", label="_nolegend_")

        if "near" in dd.columns and dd["near"].notna().any():
            nvals = dd["near"].to_numpy(dtype=float)
            for xi, yi, ni in zip(x, y, nvals):
                if not np.isnan(ni):
                    ax.annotate(str(int(ni)), (xi, yi), xytext=(3, 3), textcoords="offset points", fontsize=8)
            if anchor_near is not None:
                mn = (nvals == float(anchor_near))
                if mn.any():
                    ax.plot([x[mn][0]], [y[mn][0]], marker="*", markersize=14, linestyle="None", label="_nolegend_")

        ctrl = dd["bpp_ctrl"].to_numpy(dtype=float) if "bpp_ctrl" in dd.columns else None
        if ctrl is not None:
            _mark_anchor_by_array(ax, x, y, anchor_bpp, ctrl)
        else:
            _mark_anchor_exact(ax, x, y, anchor_bpp)

    ax.set_xlabel("bpp per band")
    ax.set_ylabel(y_label)
    ax.set_title("RD – HC vs LC")
    ax.legend(title="Tile")

    out = Path(f"{out_prefix}_RD_HC_vs_LC_{suf}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=200)
    print(f"[OK] Figure saved: {out}")


# ----------------------------- main -----------------------------

def main():
    ap = argparse.ArgumentParser(description="RD curves using bpp_mean vs *_global_rep metrics, tile-aware.")
    ap.add_argument("--csv", required=True, help="metrics.csv")
    ap.add_argument("--case",  default=None)
    ap.add_argument("--asset", default=None)
    ap.add_argument("--tile",  default=None, help="HC or LC; if omitted, draw combined + individual figures")
    ap.add_argument("--codec", default=None)
    ap.add_argument("--anchor-q",     type=float, default=None, help="J2K QUALITY to highlight")
    ap.add_argument("--anchor-bpp",   type=float, default=None, help="CCSDS target bpp (from rate_key=bpp)")
    ap.add_argument("--anchor-error", type=float, default=None, help="JPEG-LS NEAR/ERROR to highlight")
    ap.add_argument("--out-prefix", default="fig/rd")
    ap.add_argument("--ymetric", choices=["psnr","ssim"], default="psnr")
    ap.add_argument("--interp", action="store_true", help="Enable piecewise-linear interpolation")
    ap.add_argument("--interp-points", type=int, default=200, help="Samples for interpolation")
    args = ap.parse_args()

    df = _read_csv_smart(args.csv)

    if args.case  is not None and "case"  in df.columns:  df = df[df["case"]  == args.case]
    if args.asset is not None and "asset" in df.columns:  df = df[df["asset"] == args.asset]
    if args.codec is not None and "codec" in df.columns:  df = df[df["codec"] == args.codec]
    if df.empty:
        raise SystemExit("No rows match the provided filters.")

    d = _prep_df(df)

    if args.tile:
        plot_rd_single(d, args.tile, args.anchor_q, args.anchor_bpp,
                       args.out_prefix, ymetric=args.ymetric,
                       codec_filter=args.codec, anchor_near=args.anchor_error,
                       interp=args.interp, interp_points=args.interp_points)
    else:
        plot_rd_both(d, args.anchor_q, args.anchor_bpp,
                     args.out_prefix, ymetric=args.ymetric,
                     codec_filter=args.codec, anchor_near=args.anchor_error,
                     interp=args.interp, interp_points=args.interp_points)
        if "tile_id" in d.columns:
            for t in sorted(d["tile_id"].dropna().unique()):
                plot_rd_single(d, t, args.anchor_q, args.anchor_bpp,
                               args.out_prefix, ymetric=args.ymetric,
                               codec_filter=args.codec, anchor_near=args.anchor_error,
                               interp=args.interp, interp_points=args.interp_points)

if __name__ == "__main__":
    main()
