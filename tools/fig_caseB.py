#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot summary bars from one or more CSV files (LC vs HC, per codec).
- Input: 1..N CSV files with columns like tile_id, codec, cr_mean, t_comp_s_mean, mem_*_peak_mb_mean
- Normalization: flexible column name matching and LC/HC tier normalization
- Output: three bar charts (CR, encoding time, peak memory) and PNG files
"""

import sys, re, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _read_csv_flex(path: str) -> pd.DataFrame:
    """
    Read CSV trying flexible separators/decimals.
    First try pandas' automatic sep detection with both decimal variants.
    """
    # 1) Try automatic sep and decimal=","
    try:
        return pd.read_csv(path, sep=None, engine="python", decimal=",")
    except Exception:
        pass
    # 2) Fallback: automatic sep and decimal="."
    return pd.read_csv(path, sep=None, engine="python", decimal=".")


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols:
            return cols[c.lower()]
    raise KeyError(f"None of {candidates} found. Columns present: {list(df.columns)}")


def _norm_tile(x: object) -> str:
    s = str(x).strip().upper()
    LC = ("LC", "LOW", "L")
    HC = ("HC", "HIGH", "H")
    if s in LC:
        return "LC"
    if s in HC:
        return "HC"
    return s


def _pivot_lc_hc(df: pd.DataFrame, col_codec: str, metric: str, codecs_order: list[str]) -> pd.DataFrame:
    p = (
        df.groupby([col_codec, "tier"], as_index=False)[metric]
          .mean()
          .pivot(index=col_codec, columns="tier", values=metric)
          .reindex(codecs_order)
    )
    for t in ("LC", "HC"):
        if t not in p.columns:
            p[t] = np.nan
    return p[["LC", "HC"]]


def _plot_bars(pvt: pd.DataFrame, title: str, ylabel: str, fmt: str = ":.2f", fname: str | None = None):
    ax = pvt.plot(kind="bar", rot=0, figsize=(8, 4.2))
    ax.set_title(title)
    ax.set_xlabel("Codec")
    ax.set_ylabel(ylabel)
    ax.legend(title="Tier")
    for cont in ax.containers:
        try:
            ax.bar_label(cont, fmt=fmt)
        except Exception:
            pass
    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=160)
        print("Saved:", fname)
    plt.show()


def main():
    ap = argparse.ArgumentParser(description="Plot LC vs HC bar charts from CSV experiment summaries.")
    ap.add_argument("csv_paths", nargs="+", help="CSV files")
    ap.add_argument("--max-codecs", type=int, default=3, help="Max number of codecs to display (default: 3)")
    ap.add_argument("--mem", choices=["enc", "dec"], default="enc",
                    help="Use encoder (enc) or decoder (dec) peak memory (default: enc)")
    args = ap.parse_args()

    # --- read and concat all CSVs ---
    dfs = []
    for p in args.csv_paths:
        df = _read_csv_flex(p)
        # normalize column names: strip + spaces -> underscore
        df.columns = [re.sub(r"\s+", "_", c.strip()) for c in df.columns]
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)

    # --- expected columns (flexible names) ---
    COL_TILE  = _find_col(df, ["tile_id", "tile", "tier", "profile"])
    COL_CODEC = _find_col(df, ["codec", "coder", "codec_name"])
    COL_CR    = _find_col(df, ["cr_mean", "cr", "compression_ratio", "ratio"])
    COL_TENC  = _find_col(df, ["t_comp_s_mean", "enc_time_mean", "encode_time_mean", "t_comp_s"])
    if args.mem == "enc":
        COL_MEM = _find_col(df, ["mem_comp_peak_mb_mean", "mem_comp_peak_mb"])
    else:
        COL_MEM = _find_col(df, ["mem_dec_peak_mb_mean", "mem_dec_peak_mb"])

    # --- normalize LC/HC labels and filter ---
    df["tier"] = df[COL_TILE].apply(_norm_tile)
    df = df[df["tier"].isin(["LC", "HC"])].copy()

    # --- ensure numeric metrics ---
    for col in [COL_CR, COL_TENC, COL_MEM]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- pick up to N codecs in stable order ---
    codecs = pd.Index(df[COL_CODEC].dropna().astype(str).unique())
    if len(codecs) > args.max_codecs:
        print(f"Note: {len(codecs)} codecs found; showing first {args.max_codecs}:", list(codecs[:args.max_codecs]))
    codecs = codecs[:args.max_codecs]
    df[COL_CODEC] = pd.Categorical(df[COL_CODEC].astype(str), categories=list(codecs), ordered=True)

    # --- pivots ---
    cr_pvt   = _pivot_lc_hc(df, COL_CODEC, COL_CR,   list(codecs))
    tenc_pvt = _pivot_lc_hc(df, COL_CODEC, COL_TENC, list(codecs))
    mem_pvt  = _pivot_lc_hc(df, COL_CODEC, COL_MEM,  list(codecs))

    # --- plots ---
    _plot_bars(cr_pvt,   "CR achieved (LC vs HC)",   "CR (ratio)", fname="fig_cr.png")
    _plot_bars(tenc_pvt, "Encoding time (LC vs HC)", "Time [s]", fname="fig_time.png")
    mem_title = "Peak memory (LC vs HC) [ENC]" if args.mem == "enc" else "Peak memory (LC vs HC) [DEC]"
    _plot_bars(mem_pvt,  mem_title,                  "Memory [MiB]", fname="fig_mem.png")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: python graficas.py file1.csv [file2.csv ...] [--max-codecs 3] [--mem enc|dec]")
        sys.exit(1)
    main()
