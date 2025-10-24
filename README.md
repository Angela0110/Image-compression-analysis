# 📌 Image Compression Analysis – Baselines, Codecs & Data Requirements

This repository **does not include datasets nor codec binaries**.  
To run the pipelines, you must **download the input imagery and provide your own codec executables** (CCSDS-121 / CCSDS-122 / CCSDS-123).

---

## ✅ 1) Environment Setup (recommended)

> It is recommended to use **conda-forge** to avoid GDAL/Rasterio dependency conflicts.

```bash
# Create and activate environment
conda create -n rs-baselines python=3.11 -y
conda activate rs-baselines

# Core GIS / numeric / utilities
conda install -c conda-forge \
  gdal rasterio numpy scipy pandas scikit-image pillow matplotlib \
  lxml tqdm shapely pyproj -y

# Extra tools via pip
pip install psutil rich typer
```

## ✅ 2) Input Data

### **Case A – Sentinel-2**

You need **4 JP2 bands at 10 m**: `B02`, `B03`, `B04`, `B08`.

- **Source product:** Sentinel-2B Level-2A (SAFE)
- **Product used:**  
  `S2B_MSIL2A_20250809T113319_N0511_R080_T29TNH_20250809T122256.SAFE`  
  (Tile **T29TNH**, acquired **9 August 2025**)
- **Download from:** https://browser.dataspace.copernicus.eu/?zoom=8&lat=41.96536&lng=-6.96863&themeId=DEFAULT-THEME&visualizationUrl=U2FsdGVkX1%2FfaUE5JaitMr8Ae%2Bd%2F7KEIcInonEHKqKWj%2ByA3hFDpmYHUWrNDn2tko6THqDLJr023YrMin5erbA%2BalIAcrbXb1tQvQIgvUSs4DG7vpJFcuLFxVv0hQn36&datasetId=S2_L2A_CDAS
- **Place it in:** data/raw/

> If you use a different tile/date, keep the same band naming or update the file paths at the top of `make_baseline_A.py`.

**Run Case A baseline:**
```bash
python make_baseline_A.py
```
> Check if runs/tile/index_caseA.json and runs/tile/index_caseA.json matches with the resulting output.

### **Case B – EnMAP**

This dataset requires **registration and license approval** from the provider.

- **Source:** EnMAP Level-2A SR  
- **Datatake used:** `DT0000156472` (first seven tiles only)  
- **Download from:** https://geoservice.dlr.de/eoc/ogc/stac/v1/search?f=text%2Fhtml&startIndex=40&limit=20  
- **Place the files in:** `data/raw/`

**Required files:**

| Type | Pattern |
|--------|---------|
| Spectral tiles | `*SPECTRAL_IMAGE_COG*.TIF` |
| (Optional) Quality flags | `*QL_QUALITY_TESTFLAGS_COG*.TIF` |
| (Optional) Pixel mask | `*QL_PIXELMASK_COG*.TIF` |
| Metadata XML (must include wavelengths) | `*METADATA*.xml` |

**Run baseline generation:**
```bash
python make_baseline_B.py \
  --input-raw data/raw \
  --output data/baseline \
  --dt DT0000156472 \
  --target-bands 180 \
  --k 2 \
  --lc 580,5620 \
  --hc 2000,1536 \
  --err-mode mean \
  --err-scale fixed
```
> Check if runs/tile/index_caseB.json and runs/tile/index_caseB.json matches with the resulting output.

## ✅ 3) External binaries

This project calls external codec executables. **Provide the paths to the binaries in the corresponding wrappers (or via environment variables, if supported by the wrapper).**

| Codec      | Standard     | Platform in this repo |
|------------|--------------|-----------------------|
| `ccsds121` | CCSDS 121.0  | **WSL (Linux)** — libBAEC |
| `ccsds122` | CCSDS 122.0  | **Windows** (native `.exe`) |
| `ccsds123` | CCSDS 123.0  | **WSL (Linux)** — CNES |

- Example (Windows): set `CCSDS122_BIN="C:\tools\ccsds\122\ccsds122.exe"`; for Linux-only codecs, call via WSL, e.g., `CCSDS121_BIN="/opt/ccsds/121/ccsds121"` and run with `wsl`.

## ✅ 4) Tests and Usage Examples

Before running the full pipelines, you may inspect each wrapper to understand its available arguments and capabilities. Below are some example commands that demonstrate how to execute compression runs with different codecs.

### 🔹 Example 1 — Sentinel-2 (Case A), JPEG2000

```bash
python tools/run_codec.py \
  --indices runs/tile/index_caseA.json \
  --codec jpeg2000 \
  --rate-key quality \
  --rates 1 2 4 6 8 10 15 20 25 30 35 40 60 100 \
  --reps 3 \
  --outdir runs/tile/caseA/j2k \
  --compressor-cmd "python tools/codecs/j2k/j2k_wrap.py" \
  --keep-bitstream \
  --case caseA \
  --ql-err-global 255 \
  --ql-err-zoom 32 \
  --ql-rgb
```
### 🔹 Example 2 — EnMAP (Case B), CCSDS-121 (lossless)
```bash
python tools/run_codec.py \
  --indices "runs\tile\index_caseB.json" `
  --codec ccsds121 `
  --outdir "runs\tile\caseB\ccsds121_anchor" `
  --compressor-cmd python tools\codecs\ccsds121\ccsds121_wrap.py `
  --keep-bitstream `
  --case caseB `
  --reps 3 `
  -- `
  --run-in-wsl `
  --preproc none `
  --nbit 16 `
  --interleave bip `
  --tile 512
```
## 📊 Generating figures and tables
Once compression tests are completed (so that metrics_mean.csv files exist for each codec), you can generate comparison figures and tables.

### Example 1 — Case B overlay figure:

```bash
python tools/fig_caseB.py \
  runs/tile/caseB/ccsds121_anchor/metrics_mean.csv \
  runs/tile/caseB/jpegls_lossless/metrics_mean.csv \
  runs/tile/caseB/ccsds123_lossless/metrics_mean.csv
```
### Example 2 — Case A metric overlay:

```bash
python tools/overlay_means.py \
  --inputs runs/tile/caseA/jpegls/metrics_mean.csv \
           runs/tile/caseA/ccsds/metrics_mean.csv \
           runs/tile/caseA/j2k/metrics_mean.csv \
  --case caseA \
  --asset tile_1024 \
  --tiles HC \
  --ymetric psnr \
  --interp \
  --interp-points 300 \
  --iso-quality-psnr 65 \
  --iso-rate-cr "3,5,8" \
  --out-prefix fig/caseA/overlay_caseA
```
