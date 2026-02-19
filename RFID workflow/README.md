# RFID Workflow

*Part of the [Squirrels in Town](../README.md) study project — Group 3: Jan Becker, Deekshita Ram, Darian Weiß*

---

## Overview

This project provides an end-to-end Jupyter Notebook workflow (`squirrel_rfid_workflow.ipynb`) that **synchronizes RFID sensor data with box camera footage** to study squirrel behavior. It computes:

- Time spent in the box per squirrel
- Number of nuts taken per squirrel (via YOLO object detection)
- A reviewable overlay video of relevant segments
- CSV tables for downstream analysis

### Pipeline Summary

1. Discover all video files in `input/` and downscale them
2. Detect **motion segments** (background subtraction) to skip idle frames
3. Run **YOLO** on motion segments — keep those with high-confidence squirrel detections
4. *(Optional)* **Manually confirm** each detected segment (`MANUAL_CONFIRM_SEGMENTS = True`)
5. Build a **time calibration model** per video via OCR of the on-video timestamp overlay
6. Merge all videos as one continuous recording
7. Match video segments to **RFID visits** from the Excel sheet
8. Aggregate totals and export results
9. Render an **overlay video** of kept segments

---

## Prerequisites

### Python packages

Install all dependencies by running the first code cell in the notebook, or manually:

```bash
pip install ultralytics opencv-python pandas openpyxl tqdm easyocr matplotlib
```

> No external Tesseract binary is required — `easyocr` handles all OCR internally.

### ffmpeg (optional but recommended)

If `ffmpeg` is on your PATH, it is used for faster and higher-quality video downscaling.  
Without it, OpenCV is used as a fallback (slower, slightly lower quality).

Download: https://ffmpeg.org/download.html

---

## Folder Structure

```
RFID workflow/
├── input/                          # Place your video files here (do NOT rename them!)
│   ├── YYYYMMDD_SITE_BOXNR_in.mp4  # Expected filename format (see below)
│   ├── antenna_master_sheet.xlsx   # RFID data file
│   └── resized/                    # Optional: pre-resized videos (skip downscaling)
├── models/
│   └── best.pt                     # YOLO weights (required)
├── outputs/                        # Auto-generated results
│   ├── motion_segments.json
│   ├── segments_kept.json
│   ├── segments_kept.csv
│   ├── summary_per_squirrel.csv
│   ├── overlay_kept_segments.mp4
│   └── downscaled/
├── squirrel_rfid_workflow.ipynb    # Main workflow notebook
└── README.md                       # This file
```

---

## How to Run

### 1. Prepare your inputs

1. **Place at least one video file** in the `input/` folder.
   - **Do not rename the video files.** The workflow automatically extracts metadata (study site, box number, date) from the original filename.
   - Expected filename format: `YYYYMMDD_SITE_BOXNR_in.mp4`  
     Example: `20241015_TrepS_02_in (8).mp4`
   - If the filename cannot be parsed, RFID filtering is skipped (a warning is shown).

2. **Place the RFID Excel file** at `input/antenna_master_sheet.xlsx`.  
   Required columns:

   | study_site | box_nr | ID         | name   | sex  | date       | time     | unit_nr |
   |------------|-------:|------------|--------|------|------------|----------|--------:|
   | trep_s     | 2      | 000815E3CB | Stitch | male | 15.10.2024 | 10:48:54 | 1       |

3. **Ensure the YOLO weights** are at `models/best.pt`.  
   The model must detect at least the class `squirrel`.  
   Optionally it can also detect `cup_full`, `cup_empty`, and/or `nut` for nut counting.

### 2. Open and run the notebook

Open `squirrel_rfid_workflow.ipynb` in VS Code or JupyterLab and run **all cells in order** (top to bottom).

> **Important:** Always run cells sequentially. Later cells depend on the state set up by earlier ones.

### 3. Key configuration options

Edit the configuration block in cell 3 (`## 2) Configuration`) before running:

| Parameter | Default | Description |
|---|---|---|
| `INPUT_DIR` | `input/` | Folder containing video files |
| `RFID_XLSX_PATH` | `input/antenna_master_sheet.xlsx` | Path to RFID Excel file |
| `YOLO_WEIGHTS_PATH` | `models/best.pt` | Path to YOLO model weights |
| `MANUAL_CONFIRM_SEGMENTS` | `True` | Show a thumbnail for each detected segment and confirm manually |
| `YOLO_DEVICE` | `"cpu"` | Set to `"0"` to use GPU (much faster) |
| `DOWNSCALE_WIDTH` | `960` | Target width for downscaling; set to `0` to skip |
| `YOLO_CONF_SQUIRREL` | `0.55` | Minimum confidence threshold to keep a squirrel segment |
| `TS_ROI_REL` | `(0.00, 0.88, 0.45, 1.00)` | Relative ROI for OCR of the timestamp overlay |

### 4. Speed tips

- **Use a GPU:** Set `YOLO_DEVICE = "0"` in the configuration cell.
- **Use pre-resized videos:** Place already-downscaled videos in `input/resized/` — the downscaling step is then skipped automatically.
- **Install ffmpeg:** If available, it is used instead of OpenCV for significantly faster downscaling.
- **Reduce OCR samples:** Lower `OCR_N_SAMPLES` (default: 18) if timestamp calibration is slow.

---

## Outputs

| File | Description |
|---|---|
| `outputs/motion_segments.json` | All motion segments detected per video |
| `outputs/segments_kept.json` | Squirrel segments after YOLO filtering (and optional manual confirm) |
| `outputs/segments_kept.csv` | Same as above in tabular form |
| `outputs/summary_per_squirrel.csv` | Total time and nut count aggregated per squirrel |
| `outputs/overlay_kept_segments.mp4` | Overlay video with squirrel info, cumulative time, and nut counts |
| `outputs/downscaled/` | Downscaled versions of the input videos |
| `outputs/ocr_debug/` | OCR debug images and results per video |

---

## Important Notes

- **Do not rename video files.** The notebook parses the original camera filename to extract the recording date, study site, and box number for RFID matching. If filenames are changed, RFID filtering falls back to no filtering (all entries used).
- **Multiple videos are treated as one continuous recording.** All `.mp4`/`.mov`/`.avi`/`.mkv` files found in `input/` are sorted alphabetically and concatenated virtually. Ensure your files sort in the correct chronological order.
- **Timestamp drift is corrected automatically.** Many cameras record variable frame timing. The notebook fits a linear model from multiple OCR samples per video to compensate.
- **OCR requires a visible timestamp overlay** in the video. If no timestamp is readable, a manual fallback is triggered. Adjust `TS_ROI_REL` in the configuration if the OCR region does not match your camera's overlay position.
- **The `outputs/` folder is created automatically** if it does not exist.
- **Re-runs are incremental:** motion segment detection and downscaling results are cached and skipped if already present.
