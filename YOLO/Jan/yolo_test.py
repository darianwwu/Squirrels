from ultralytics import YOLO
import cv2 as cv

model = YOLO("yolo11n.pt")

# train the model on dataset
# model.train( 
# data=r"D:\squirrel.v1i.yolov11\data.yaml", 
# epochs=100, 
# imgsz=640, 
# batch=16, 
# name="squirrel_yolo11" 
# )

# train the model on dataset
model.train(
    data=r"yolo_dataset_from_labelbox_squirrel\data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    workers=8,
    cache=True,
    name="squirrel_model_big"
)

# use existing model to predict on video
model = YOLO(r"runs/detect/squirrel_model2/weights/squirrelrightclasses.pt")

# path declaration
video_path = r"C:\Users\job02\Documents\Hoernchen\study_project_ws25_26\study_project_ws25_26\v5.mp4"
#video_outside_path = r"C:\Users\job02\Documents\Squirrel_Videos\outside\20241030_TrepN_04_out (8)_short.mp4"
output_path = r"C:\Users\job02\Documents\Hoernchen\squirrel_yolo_output_dico_nuts.mp4"

# debugging class names
# print(model.names)
# model.model.names[0] = "cup_full"
# model.model.names[1] = "squirrel"
# model.model.names[2] = "nut"
# model.model.names[3] = "cup_empty"
# model.model.names[4] = "disco_ball"

#save new model with right class names
# model.save('runs\detect\squirrel_model2\weights\squirrelrightclasses.pt')

# Draw detections on video and save output
cap = cv.VideoCapture(video_path)
fps = cap.get(cv.CAP_PROP_FPS)
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

fourcc = cv.VideoWriter_fourcc(*"mp4v")
out = cv.VideoWriter(output_path, fourcc, fps, (width, height))

#max_frames = int(fps * 30)
frame_count = 0

#while frame_count < max_frames:
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.3)

    # Always start with the original frame
    output_frame = frame.copy()

    # Draw detections 
    if len(results) > 0:
        output_frame = results[0].plot()

    # Write every frame
    out.write(output_frame)

    # Debug:
    #cv.imshow("out", output_frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
out.release()
cv.destroyAllWindows()
