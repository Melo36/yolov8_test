import json
import os
from PIL import Image

# Set the paths for the input and output directories
image_dir = '/Users/melo/PycharmProjects/DataAugmentation/augmented_images/images'
annotations_dir = '/Users/melo/PycharmProjects/DataAugmentation/augmented_images/labels'
output_dir = '/Users/melo/PycharmProjects/DataAugmentation'

# Define the categories for the COCO dataset
categories = [{"id": 0, "name": "haferflocken_ja"}]

# Define the COCO dataset dictionary
coco_dataset = {
    "info": {},
    "licenses": [],
    "categories": categories,
    "images": [],
    "annotations": []
}

new_id = 0
# Loop through the images in the input directory
for image_file in os.listdir(image_dir):
    # Load the image and get its dimensions
    image_path = os.path.join(image_dir, image_file)
    split_path = image_path.split('/')
    file_name = split_path[len(split_path) - 1]

    if file_name.startswith('.'):
        continue

    image = Image.open(image_path)
    width, height = image.size

    # Add the image to the COCO dataset
    image_dict = {
        "id": new_id,
        "width": width,
        "height": height,
        "file_name": image_file
    }
    coco_dataset["images"].append(image_dict)



    # Load the bounding box annotations for the image
    with open(os.path.join(annotations_dir, f'{image_file.split(".")[0]}.txt')) as f:
        annotations = f.readlines()

    # Loop through the annotations and add them to the COCO dataset
    for ann in annotations:
        x, y, w, h = map(float, ann.strip().split()[1:])
        x_min, y_min = int((x - w / 2) * width), int((y - h / 2) * height)
        x_max, y_max = int((x + w / 2) * width), int((y + h / 2) * height)
        ann_dict = {
            "id": len(coco_dataset["annotations"]),
            "image_id": new_id,
            "category_id": 0,
            "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
            "area": (x_max - x_min) * (y_max - y_min),
            "iscrowd": 0
        }
        coco_dataset["annotations"].append(ann_dict)

    new_id = new_id + 1
# Save the COCO dataset to a JSON file
with open(os.path.join(output_dir, 'annotations_new2.json'), 'w') as f:
    json.dump(coco_dataset, f)