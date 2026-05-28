import os
import json
from PIL import Image

# Since this Python file is inside the same folder as images + json files
json_folder = "."
image_folder = "."
output_file = "annotations_coco.json"

classes = ["healthy_leaf", "healthy_stem", "diseased_leaf"]
category_id = {name: i + 1 for i, name in enumerate(classes)}

coco = {
    "images": [],
    "annotations": [],
    "categories": []
}

for name, cid in category_id.items():
    coco["categories"].append({
        "id": cid,
        "name": name,
        "supercategory": "plant"
    })

annotation_id = 1
image_id = 1

for file in os.listdir(json_folder):
    if not file.endswith(".json"):
        continue

    json_path = os.path.join(json_folder, file)

    with open(json_path, "r") as f:
        data = json.load(f)

    image_name = data.get("imagePath")

    if image_name is None:
        print(f"Skipping {file}: no imagePath found")
        continue

    image_path = os.path.join(image_folder, image_name)

    if os.path.exists(image_path):
        width, height = Image.open(image_path).size
    else:
        width = data.get("imageWidth")
        height = data.get("imageHeight")

    coco["images"].append({
        "id": image_id,
        "file_name": image_name,
        "width": width,
        "height": height
    })

    for shape in data.get("shapes", []):
        label = shape["label"]

        if label not in category_id:
            print(f"Skipping unknown label: {label}")
            continue

        points = shape["points"]

        segmentation = []
        for x, y in points:
            segmentation.extend([float(x), float(y)])

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        x_min = min(xs)
        y_min = min(ys)
        box_width = max(xs) - x_min
        box_height = max(ys) - y_min

        area = box_width * box_height

        coco["annotations"].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id[label],
            "segmentation": [segmentation],
            "bbox": [
                float(x_min),
                float(y_min),
                float(box_width),
                float(box_height)
            ],
            "area": float(area),
            "iscrowd": 0
        })

        annotation_id += 1

    image_id += 1

with open(output_file, "w") as f:
    json.dump(coco, f, indent=4)

print("COCO file created:", output_file)