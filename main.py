from ultralytics import YOLO
import numpy as np
import cv2 as cv
import torch.onnx

model = YOLO("/Users/melo/Desktop/yolo_100/best.pt")

#results = model.train(data="config.yaml", epochs=50)
#onnxModel = model.export(format="onnx", opset=10)


results = model(['/Users/melo/Desktop/test image for ssd/product_824_jpg.rf.e701747c03c80df259a29c3a2f0bea05.jpg'])  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    result.show()  # display to screen
    result.save(filename='result.jpg')  # save to disk


'''cap = cv.VideoCapture('/Users/melo/Desktop/test_videos/IMG_1739.MOV')
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
cv.destroyAllWindows()'''