import json
import os
import cv2
import random
# Script to convert Labelbox NDJSON annotations to YOLO format 
# paths
ndjson_file = r"C:\Users\job02\Documents\Hoernchen\Squirrels_in_town_annotations_12_12_2025.ndjson"
videos_path = r"C:\Users\job02\Documents\Hoernchen\study_project_ws25_26\study_project_ws25_26"
output_base = "yolo_dataset_from_labelbox_squirrel"

train_ratio = 0.8  # 80% train, 20% val

# create folders
train_images = os.path.join(output_base, "train", "images")
train_labels = os.path.join(output_base, "train", "labels")
val_images = os.path.join(output_base, "valid", "images")
val_labels = os.path.join(output_base, "valid", "labels")

for d in [train_images, train_labels, val_images, val_labels]:
    os.makedirs(d, exist_ok=True)

class_map = {}
current_class_id = 0

# read NDJSON
with open(ndjson_file, "r") as f:
    data = [json.loads(line) for line in f]

# Process each video and annotations
for item in data:
    video_name = os.path.splitext(item["data_row"]["external_id"])[0]
    video_file = os.path.join(videos_path, item["data_row"]["external_id"])
    media_width = item["media_attributes"]["width"]
    media_height = item["media_attributes"]["height"]

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Video could not be opened: {video_file}")
        continue
    
    print(f"Processing vid: {video_file}")
    
    # collect annotations
    frame_annotations = {}
    for project in item["projects"].values():
        for label in project["labels"]:
            frame_annotations.update(label["annotations"]["frames"])

    # Process video frames
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_id = str(frame_count)

        # train / valid split
        if random.random() < train_ratio:
            img_dir, lbl_dir = train_images, train_labels
        else:
            img_dir, lbl_dir = val_images, val_labels

        file_id = f"{video_name}_{frame_count:05d}"
        image_path = os.path.join(img_dir, f"{file_id}.jpg")
        label_path = os.path.join(lbl_dir, f"{file_id}.txt")

        cv2.imwrite(image_path, frame)

        # convert annotations to YOLO format
        yolo_lines = []
        if frame_id in frame_annotations:
            for obj in frame_annotations[frame_id]["objects"].values():
                class_name = obj["name"]
                if class_name not in class_map:
                    class_map[class_name] = current_class_id
                    current_class_id += 1

                bbox = obj["bounding_box"]
                x_center = (bbox["left"] + bbox["width"] / 2) / media_width
                y_center = (bbox["top"] + bbox["height"] / 2) / media_height
                w = bbox["width"] / media_width
                h = bbox["height"] / media_height

                yolo_lines.append(
                    f"{class_map[class_name]} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
                )

        with open(label_path, "w") as f:
            f.write("\n".join(yolo_lines))

    cap.release()

# save classes
with open(os.path.join(output_base, "classes.txt"), "w") as f:
    for name, idx in sorted(class_map.items(), key=lambda x: x[1]):
        f.write(f"{name}\n")
print("Done")
