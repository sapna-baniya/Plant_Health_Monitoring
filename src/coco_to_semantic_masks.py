import argparse
import os
import numpy as np
from PIL import Image
from pycocotools.coco import COCO


def safe_ann_to_mask(coco, ann, height, width):
    seg = ann.get("segmentation", [])

    use_segmentation = (
        isinstance(seg, list)
        and len(seg) > 0
        and isinstance(seg[0], list)
        and len(seg[0]) >= 6
    )

    if use_segmentation:
        try:
            return coco.annToMask(ann)
        except Exception:
            pass

    # fallback: create mask from bbox
    x, y, w, h = ann["bbox"]
    mask = np.zeros((height, width), dtype=np.uint8)

    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(width, int(x + w))
    y2 = min(height, int(y + h))

    mask[y1:y2, x1:x2] = 1
    return mask


def convert(annotation_file, output_dir, image_size=512):
    os.makedirs(output_dir, exist_ok=True)

    coco = COCO(annotation_file)

    for image_id in coco.getImgIds():
        img_info = coco.loadImgs(image_id)[0]

        width = img_info["width"]
        height = img_info["height"]

        semantic_mask = np.zeros((height, width), dtype=np.uint8)

        ann_ids = coco.getAnnIds(imgIds=image_id)
        anns = coco.loadAnns(ann_ids)

        for ann in anns:
            category_id = ann["category_id"]

            instance_mask = safe_ann_to_mask(coco, ann, height, width)

            semantic_mask[instance_mask > 0] = category_id

        semantic_mask = Image.fromarray(semantic_mask)
        semantic_mask = semantic_mask.resize((image_size, image_size), Image.NEAREST)

        file_name = os.path.splitext(os.path.basename(img_info["file_name"]))[0] + ".png"
        output_path = os.path.join(output_dir, file_name)

        semantic_mask.save(output_path)

    print(f"Saved semantic masks to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_size", type=int, default=512)

    args = parser.parse_args()

    convert(args.annotations, args.output_dir, args.image_size)