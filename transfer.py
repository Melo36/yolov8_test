

#model = YOLO("/Users/melo/PycharmProjects/yolov8Test/runs/detect/train2/weights/best.pt")  # load a pretrained model (recommended for training)

#results = model.train(data="config.yaml", epochs=50, device='mps')


'''i=0
freeze = 80
for k, v in model.named_parameters():
    v.requires_grad = True  # train all layers
    if i > freeze:
        break
    else:
        print(f'freezing {k}')
        v.requires_grad = False
        i += 1'''


import json

def update_ids(json_data):
    new_id = 166
    for image in json_data["annotations"]:
        image["id"] = new_id
        new_id += 1


input_file = '/Users/melo/PycharmProjects/yolov8Test/annotations2.json'
output_file = '/Users/melo/PycharmProjects/yolov8Test/annotations3.json'

with open(input_file, 'r') as file:
    data = json.load(file)

update_ids(data)

with open(output_file, 'w') as file:
    json.dump(data, file, indent=2)




