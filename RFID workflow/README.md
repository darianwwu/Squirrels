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

> **Important:** Part of this workflow is to show off the different analysis steps we took during the course.
> We are aware that some steps could be skipped or optimized, but we wanted to keep the notebook as a record of our process.

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

### 3. Configuration reference

All settings are in the configuration cell (`## 2) Configuration`). They are grouped by topic below.

---

#### Paths

| Parameter | Default | Description |
|---|---|---|
| `INPUT_DIR` | `input/` | Folder where the notebook looks for video files and the RFID Excel sheet |
| `RFID_XLSX_PATH` | `input/antenna_master_sheet.xlsx` | Path to the RFID Excel file |
| `YOLO_WEIGHTS_PATH` | `models/best.pt` | Path to the YOLO model weights file |
| `OUTPUT_DIR` | `outputs/` | All result files are written here (created automatically) |

---

#### Manual segment review

| Parameter | Default | Description |
|---|---|---|
| `MANUAL_CONFIRM_SEGMENTS` | `False` | When `True`, after YOLO detects a squirrel segment, a thumbnail of the best frame is shown and you are asked to confirm or reject it. Set to `False` to skip manual review and accept all YOLO-detected segments automatically. |

> **Recommendation:** The default is `False` for efficient batch processing. Switch to `True` for your first run on new footage or a new camera setup to verify that the detection thresholds are working correctly.

---

#### Video downscaling

| Parameter | Default | Description |
|---|---|---|
| `DOWNSCALE_WIDTH` | `960` | Target pixel width for downscaled videos. Height is scaled proportionally. Set to `0` to disable downscaling entirely (not recommended — very slow). Lower values (e.g. `640`) are faster but reduce OCR and detection accuracy. |

> If you place already-downscaled videos in `input/resized/`, this step is skipped automatically regardless of this setting.

---

#### Motion detection (Stage 1)

These parameters control the background subtraction pass that finds candidate segments where *something moves*. Tune these if you get too many false positives (empty segments) or miss real visits.

| Parameter | Default | Description |
|---|---|---|
| `MOTION_SAMPLE_FPS` | `6.0` | How many frames per second are sampled for motion. Lower values are faster but may miss brief movements. |
| `MIN_MOTION_PCT` | `0.005` | Fraction of pixels that must change per frame to count as motion (0.5%). **Increase** this if wind/lighting causes too many false triggers. **Decrease** if subtle squirrel movements are being missed. |
| `MOTION_GAP_S` | `3.0` | A gap of non-motion shorter than this (in seconds) is bridged — the segment is kept intact. **Increase** if visits are wrongly split into multiple segments due to brief stillness. |
| `MOTION_PRE_S` | `1.5` | Seconds of video added before the detected motion start. Increase to catch the squirrel arriving. |
| `MOTION_POST_S` | `1.5` | Seconds of video added after the detected motion end. Increase to catch the squirrel leaving. |
| `MOTION_MIN_LEN_S` | `2.0` | Segments shorter than this (in seconds) are discarded. Helps eliminate spurious motion flickers. |

---

#### YOLO squirrel detection (Stage 2)

These parameters control the YOLO inference pass that verifies whether a motion segment actually contains a squirrel.

| Parameter | Default | Description |
|---|---|---|
| `YOLO_CONF_SQUIRREL` | `0.70` | Minimum detection confidence to keep a segment. **Lower** this (e.g. `0.5`) if real visits are being missed. **Raise** it (e.g. `0.85`) if too many non-squirrel segments pass through. |
| `YOLO_SAMPLE_FPS` | `2.0` | How many frames per second are passed to YOLO within each motion segment. Higher values are more accurate but slower. |
| `YOLO_IMGSZ` | `640` | Input image size for YOLO inference. Larger values increase accuracy but slow down processing. |
| `YOLO_DEVICE` | `"cpu"` | Compute device for YOLO. Set to `"0"` to use the first GPU — this is **strongly recommended** for large datasets, as GPU inference is typically 10–50× faster. |

---

#### OCR & timestamp calibration

The notebook reads the on-video timestamp overlay to map frame numbers to real wall-clock times.

| Parameter | Default | Description |
|---|---|---|
| `TS_ROI_REL` | `(0.00, 0.88, 0.45, 1.00)` | Relative coordinates `(x0, y0, x1, y1)` of the timestamp overlay region, as fractions of frame width/height. **Adjust this first** if OCR fails — run the ROI preview cell to visually check the crop. |
| `OCR_N_SAMPLES` | `18` | Number of frames sampled across each video for timestamp OCR. More samples give a more accurate time calibration model, but take longer. Reduce to `6–10` for faster runs. |

> **If OCR fails entirely** for a video, a manual fallback is triggered: the notebook will ask you to enter the start time of the recording manually.

---

#### RFID matching

| Parameter | Default | Description |
|---|---|---|
| `RFID_GAP_S` | `60` | If two RFID triggers from the same squirrel are more than this many seconds apart, they are treated as separate visits. Decrease for boxes with very short visit gaps; increase if a single long visit is being split. |
| `RFID_TOL_S` | `5` | The matched video time window is extended by ±this many seconds around the RFID visit start/end. Accounts for small clock offsets between camera and RFID sensor. |
| `RFID_FILTER_STUDY_SITE` | auto | Automatically parsed from the video filename. Override manually if needed (e.g. `"trep_s"`). |
| `RFID_FILTER_BOX_NR` | auto | Automatically parsed from the video filename. Override manually if needed (e.g. `2`). |

---

### 4. Speed tips

- **Use a GPU:** Set `YOLO_DEVICE = "0"` in the configuration cell.
- **Use pre-resized videos:** Place already-downscaled videos in `input/resized/` — the downscaling step is then skipped automatically.
- **Install ffmpeg:** If available, it is used instead of OpenCV for significantly faster downscaling.
- **Reduce OCR samples:** Lower `OCR_N_SAMPLES` (default: `18`) if timestamp calibration is slow.
- **Disable manual review:** Set `MANUAL_CONFIRM_SEGMENTS = False` for unattended batch runs.
- **Raise `MIN_MOTION_PCT`:** Reducing the number of motion candidates directly speeds up the YOLO stage.

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
