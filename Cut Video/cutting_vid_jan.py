import cv2 as cv
import numpy as np
import time
import os
from pathlib import Path

## Function to cut videos  using background subtraction
def cut_video_by_motion(videoName, outName, motion_threshold=5000, scale=0.5):
    start_time = time.time()
    print(f"\nStart processing: {videoName}")

    # Open video
    cap = cv.VideoCapture(str(videoName))
    if not cap.isOpened():
        print("Error: Cannot open input video")
        return

    # Get video properties
    fps = cap.get(cv.CAP_PROP_FPS)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH) * scale)
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT) * scale)
    
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(str(outName), fourcc, fps, (width, height))

    # Ititalisation background subtractor
    backSub = cv.createBackgroundSubtractorMOG2(history=500, varThreshold=32)

    written_frames = 0
    total_frames = 0

    # Process video frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1

        if scale != 1.0:
            frame = cv.resize(frame, (width, height))

        fgMask = backSub.apply(frame)

        motion_pixels = np.sum(fgMask > 0)

        if motion_pixels > motion_threshold:
            out.write(frame)
            written_frames += 1

    cap.release()
    out.release()

    print(f"Total frames: {total_frames}")
    print(f"Frames written: {written_frames}")
    print(f"Saved cut video to: {outName}")
    print(f"--- {time.time() - start_time:.2f} seconds ---")

# Function to cut all videos in a folder
def cut_videos_in_folder(folder, motion_threshold=5000, scale=0.5):
    folder = Path(folder)
    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    videos = [f for f in folder.iterdir() if f.suffix.lower() in video_extensions]

    if not videos:
        print("Kein Video gefunden")
        return

    print(f"{len(videos)} Videos gefunden.\n")

    # Process every video in folder
    for video in videos:
        outName = video.with_name(video.stem + "_cut.mp4")

        if outName.exists() or video.stem.endswith("_cut.mp4"):
            print(f"Skip: {outName}")
            continue

        cut_video_by_motion(
            video,
            outName,
            motion_threshold=motion_threshold,
            scale=scale
        )


cut_videos_in_folder(
    r"C:\Users\job02\Documents\Squirrel_Videos",  
    motion_threshold=5000,
    scale=1.0
)
