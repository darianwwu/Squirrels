import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import time
import os

# Funcrion to read video, apply background subtraction and save output
def readVideo_BGSub(videoName):
    start_time = time.time()
    print("Start Reading Video")
    cap = cv.VideoCapture(videoName)

    if not cap.isOpened():
        print(f"Error: Could not open video '{videoName}'")
        return

    # Initialize background subtractor
    backSub_MOG2 = cv.createBackgroundSubtractorMOG2(history=500, varThreshold=32)

    scale = 0.5
    differences_mog2 = []
    frame_count = 0

    ret, frame = cap.read()
    if not ret:
        print("No frames found!")
        return

    # Apply scaling
    if scale != 1.0:
        frame = cv.resize(frame, (0, 0), fx=scale, fy=scale)

    height, width = frame.shape[:2]

    # Combined frame width = original + mask
    output_size = (width * 2, height)

    # Output video path
    base = os.path.splitext(os.path.basename(videoName))[0]
    out_path = f"{base}_combined_output.mp4"

    # VideoWriter setup
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    out = cv.VideoWriter(out_path, fourcc, 30, output_size)

    print(f"Exporting MP4 to: {out_path}")

    # Reset video stream to beginning
    cap.set(cv.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if scale != 1.0:
            frame = cv.resize(frame, (0, 0), fx=scale, fy=scale)

        fgMask_MOG2 = backSub_MOG2.apply(frame)
        count_mog2 = np.sum(fgMask_MOG2 > 0)
        differences_mog2.append(count_mog2)

        # Convert mask 
        fgMask_color = cv.cvtColor(fgMask_MOG2, cv.COLOR_GRAY2BGR)

        # side by side
        combined = np.hstack((frame, fgMask_color))

        #cv.imshow("Original + MOG2", combined)

        # write video
        out.write(combined)

        key = cv.waitKey(1)
        frame_count += 1

    cap.release()
    out.release()
    cv.destroyAllWindows()

    print(f"Total frames processed: {frame_count}")
    print(f"Saved MP4: {out_path}")
    print(f"--- {time.time() - start_time:.2f} seconds ---")

    return differences_mog2


# Run
diffs_mog2 = readVideo_BGSub(
    r"C:\Users\job02\Documents\Squirrel_Videos\outside\20241030_TrepN_04_out (8)_short.mp4"
)

# Plot results
plt.figure(figsize=(12, 5))
plt.plot(diffs_mog2, label='MOG2', color='red')
plt.xlabel('Frame Number')
plt.ylabel('Motion Pixels')
plt.title('MOG2 Background Subtraction')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
