from ultralytics import YOLO

model = YOLO("yolov8n.pt")

for param in model.named_parameters():
    print(param[0])

results = model.train(data="config.yaml", epochs=50, device='mps', pretrained=True, freeze=22)