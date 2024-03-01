from ultralytics import YOLO
import numpy as np
import cv2 as cv
import torch.onnx

model = YOLO("/Users/melo/PycharmProjects/yolov8Test/runs/detect/train3/weights/best.pt")

#results = model.train(data="config.yaml", epochs=50)
#onnxModel = model.export(format="onnx", opset=10)


cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        tracking = model.track(frame, persist=True)
        frame_ = tracking[0].plot()
        cv.imshow("frame", frame_)
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        if cv.waitKey(1) == ord('q'):
            break


# When everything done, release the capture
cap.release()
cv.destroyAllWindows()