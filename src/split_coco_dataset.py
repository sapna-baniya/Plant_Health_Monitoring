"""
Split one COCO annotation file into train/val/test JSON files and copy images.
Handles images stored either directly in data/images/all
or inside data/images/all/converted_images.
"""

import argparse
import json
import os
import random
import shutil


def subset(data, image_ids):
    image_ids = set(image_ids)
    return {
        "images": [img for img in data["images"] if img["id"] in image_ids],
        "annotations": [
            ann for ann in data["annotations"]
            if ann["image_id"] in image_ids
        ],
        "categories": data["categories"],
    }


def find_image_path(image_root, file_name):
    filename = os.path.basename(file_name)

    possible_paths = [
        os.path.join(image_root, file_name),
        os.path.join(image_root, filename),
        os.path.join(image_root, "converted_images", filename),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, help="Folder with all images")
    parser.add_argument("--annotations", required=True, help="Single COCO JSON file")
    parser.add_argument("--out", default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--val", type=float, default=0.1)
    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.annotations, "r") as f:
        data = json.load(f)

    images = list(data["images"])
    random.shuffle(images)

    n = len(images)
    n_train = int(n * args.train)
    n_val = int(n * args.val)

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:],
    }

    os.makedirs(os.path.join(args.out, "annotations"), exist_ok=True)

    for split, imgs in splits.items():
        split_image_dir = os.path.join(args.out, "images", split)
        os.makedirs(split_image_dir, exist_ok=True)

        cleaned_imgs = []
        missing = []

        for img in imgs:
            img = img.copy()

            original_file_name = img["file_name"]
            filename = os.path.basename(original_file_name)

            src = find_image_path(args.images, original_file_name)
            dst = os.path.join(split_image_dir, filename)

            if src:
                shutil.copy2(src, dst)
                img["file_name"] = filename
                cleaned_imgs.append(img)
            else:
                missing.append(original_file_name)
                print("Missing image:", original_file_name)

        ids = [img["id"] for img in cleaned_imgs]
        split_data = subset(data, ids)
        split_data["images"] = cleaned_imgs

        output_json = os.path.join(args.out, "annotations", f"{split}.json")

        with open(output_json, "w") as f:
            json.dump(split_data, f, indent=2)

        print("=" * 50)
        print(split)
        print("Images:", len(cleaned_imgs))
        print("Annotations:", len(split_data["annotations"]))
        print("Missing:", len(missing))
        print("Saved:", output_json)


if __name__ == "__main__":
    main()